# tests/test_integration_basic.py
import json
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import app as appmod  # app.py in project root


@pytest.fixture(scope="function")
def db_session(monkeypatch):
    """Real SQLAlchemy stack, no ORM stubs; fresh schema per test."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    appmod.Base.metadata.create_all(bind=engine)

    TestingSessionLocal = scoped_session(sessionmaker(bind=engine, autoflush=False, autocommit=False))
    original = appmod.SessionLocal
    appmod.SessionLocal = TestingSessionLocal
    try:
        yield TestingSessionLocal
    finally:
        TestingSessionLocal.remove()
        appmod.SessionLocal = original


@pytest.fixture(scope="function")
def client(db_session, monkeypatch):
    """
    Flask test client with a real DB session & minimal LLM mocking.
    We only 'route' by model name; otherwise keep the stack intact.
    """
    app = appmod.app
    app.config["TESTING"] = True

    # ---- Minimal LLM mocks for integration ----
    class _Msg:
        def __init__(self, content): self.content = content
    class _Choice:
        def __init__(self, content): self.message = _Msg(content)
    class _Resp:
        def __init__(self, content): self.choices = [_Choice(content)]

    def fake_parse(**kwargs):
        # Return realistic parsed filters
        return _Resp(json.dumps({"intent": "search", "genre": "Action", "limit": 5, "sort_by": "rating"}))
    def fake_answer(**kwargs):
        return _Resp("Here are some action picks.")

    def router(**kwargs):
        model = kwargs.get("model", "")
        if model == appmod.OPENAI_MODEL_PARSE:
            return fake_parse(**kwargs)
        return fake_answer(**kwargs)

    monkeypatch.setattr(appmod.client.chat.completions, "create", router)

    # ---- Seed a small dataset into the *real* DB ----
    db = db_session()
    m1 = appmod.Movie(id=1, title="The Matrix", year=1999, overview="Neo discovers the truth.", rating=4.8)
    m2 = appmod.Movie(id=2, title="Mad Max: Fury Road", year=2015, overview="Wasteland chase.", rating=4.6)
    db.add_all([m1, m2])
    db.add_all([
        appmod.Genre(movie_id=1, genre="Action"),
        appmod.Genre(movie_id=1, genre="Sci-Fi"),
        appmod.Genre(movie_id=2, genre="Action"),
        appmod.CastDirector(movie_id=1, name="Keanu Reeves", role="Actor"),
        appmod.CastDirector(movie_id=1, name="Lana Wachowski", role="Director"),
        appmod.CastDirector(movie_id=2, name="Tom Hardy", role="Actor"),
        appmod.CastDirector(movie_id=2, name="George Miller", role="Director"),
    ])
    db.commit()
    db.close()

    with app.test_client() as c:
        yield c


def test_e2e_query_filters_and_results(client):
    r = client.post("/query", json={"query": "show me top action movies"})
    assert r.status_code == 200
    data = r.get_json()
    assert data["parsed_filters"]["intent"] == "search"
    assert data["parsed_filters"]["genre"] == "Action"
    assert isinstance(data["results"], list)
    # Should return at least one action movie
    titles = [m["title"] for m in data["results"]]
    assert any(t in titles for t in ["The Matrix", "Mad Max: Fury Road"])


def test_e2e_chat_persistence(client):
    # 1st message -> new conversation
    r1 = client.post("/chat", json={"message": "Recommend action movies"})
    assert r1.status_code == 200
    d1 = r1.get_json()
    conv_id = d1["conversation_id"]
    assert d1["assistant_message"]

    # 2nd message -> same conversation id
    r2 = client.post("/chat", json={"message": "How about something from 2015?" , "conversation_id": conv_id})
    assert r2.status_code == 200
    d2 = r2.get_json()
    assert d2["conversation_id"] == conv_id
    # Check that history exists in DB (2 user + 2 assistant = 4 total new messages)
    # We don’t read DB directly; just ensure assistant responded again
    assert d2["assistant_message"]


def test_chat_fallback_when_llm_errors(client, monkeypatch):
    # Make the OpenAI call raise an exception to hit your fallback
    def boom(**kwargs):
        raise RuntimeError("LLM down")
    monkeypatch.setattr(appmod.client.chat.completions, "create", boom)

    r = client.post("/chat", json={"message": "Any good sci-fi?"})
    assert r.status_code == 200
    msg = r.get_json()["assistant_message"]
    # Should be your graceful fallback text, possibly with grounded list
    assert "Sorry" in msg


def test_random_and_lists_smoke(client):
    # random
    r = client.get("/movies/random")
    assert r.status_code == 200
    assert r.get_json()["title"] in {"The Matrix", "Mad Max: Fury Road"}

    # people
    r = client.get("/people?role=Director")
    assert r.status_code == 200
    names = [p["name"] for p in r.get_json()]
    assert "Lana Wachowski" in names or "George Miller" in names
