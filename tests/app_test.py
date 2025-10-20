# tests/test_app_basic.py
import json
import pytest
import sys, os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session

# Ensure project root (where app.py lives) is on path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import app as appmod 


# ---------- Fixtures: test app + isolated DB + mocked OpenAI ----------

@pytest.fixture(scope="session")
def test_engine():
    # Single in-memory engine for the session (fast)
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    # Create tables once initially (we'll drop/create per test below)
    appmod.Base.metadata.create_all(bind=engine)
    return engine


@pytest.fixture(scope="function")
def db_session(test_engine, monkeypatch):
    """
    Creates a brand-new schema and session for EACH test.
    This avoids UNIQUE conflicts when we seed the same IDs in every test.
    """
    # Reset schema per test
    appmod.Base.metadata.drop_all(bind=test_engine)
    appmod.Base.metadata.create_all(bind=test_engine)

    TestingSessionLocal = scoped_session(sessionmaker(bind=test_engine, autoflush=False, autocommit=False))

    # Monkeypatch the app's SessionLocal so request.db uses our testing session
    original = appmod.SessionLocal
    appmod.SessionLocal = TestingSessionLocal
    try:
        yield TestingSessionLocal
    finally:
        TestingSessionLocal.remove()
        appmod.SessionLocal = original


@pytest.fixture(scope="function")
def client(db_session, monkeypatch):
    """Flask test client with mocked OpenAI calls, and fresh seed data per test."""
    app = appmod.app
    app.config["TESTING"] = True

    # ---- Mock OpenAI chat responses ----
    class _Msg:
        def __init__(self, content): self.content = content

    class _Choice:
        def __init__(self, content): self.message = _Msg(content)

    class _Resp:
        def __init__(self, content): self.choices = [_Choice(content)]

    def fake_create_parse(**kwargs):
        # Respond with a tiny valid JSON object for llm_parse_query
        return _Resp(json.dumps({"intent": "search", "genre": "Action", "limit": 5, "sort_by": "rating"}))

    def fake_create_answer(**kwargs):
        # Return a simple assistant message for llm_answer
        return _Resp("Here are some action movies you might like.")

    # Router by model
    def fake_create_router(**kwargs):
        model = kwargs.get("model", "")
        if model == appmod.OPENAI_MODEL_PARSE:
            return fake_create_parse(**kwargs)
        return fake_create_answer(**kwargs)

    monkeypatch.setattr(appmod.client.chat.completions, "create", fake_create_router)

    # ---- Seed minimal data into the testing DB (fresh each test) ----
    db = db_session()

    m1 = appmod.Movie(id=1, title="The Matrix", year=1999, overview="Neo discovers the truth.", rating=4.8)
    g1 = appmod.Genre(movie_id=1, genre="Action")
    g2 = appmod.Genre(movie_id=1, genre="Sci-Fi")
    a1 = appmod.CastDirector(movie_id=1, name="Keanu Reeves", role="Actor")
    d1 = appmod.CastDirector(movie_id=1, name="Lana Wachowski", role="Director")
    d2 = appmod.CastDirector(movie_id=1, name="Lilly Wachowski", role="Director")
    db.add_all([m1, g1, g2, a1, d1, d2])

    m2 = appmod.Movie(id=2, title="Mad Max: Fury Road", year=2015, overview="Wasteland chase.", rating=4.6)
    db.add(m2)
    db.add_all([
        appmod.Genre(movie_id=2, genre="Action"),
        appmod.CastDirector(movie_id=2, name="Tom Hardy", role="Actor"),
        appmod.CastDirector(movie_id=2, name="George Miller", role="Director"),
    ])

    db.commit()
    db.close()

    with app.test_client() as c:
        yield c


# ---------- Tests: quick smoke for each endpoint ----------

def test_home_ok(client):
    r = client.get("/")
    assert r.status_code == 200
    data = r.get_json()
    assert "endpoints" in data


def test_get_movies_ok(client):
    r = client.get("/movies")
    assert r.status_code == 200
    data = r.get_json()
    assert isinstance(data, list)
    assert any(m["title"] == "The Matrix" for m in data)


def test_get_movies_filters(client):
    r = client.get("/movies?genre=Action&min_rating=4.7")
    assert r.status_code == 200
    titles = [m["title"] for m in r.get_json()]
    assert "The Matrix" in titles
    assert "Mad Max: Fury Road" not in titles  # filtered out by rating


def test_get_movie_ok(client):
    r = client.get("/movies/1")
    assert r.status_code == 200
    data = r.get_json()
    assert data["title"] == "The Matrix"
    assert "Action" in data["genres"]
    assert "Lana Wachowski" in data["directors"]


def test_get_movie_404(client):
    r = client.get("/movies/9999")
    assert r.status_code == 404
    assert r.get_json().get("error") == "Movie not found"


def test_search_movies_requires_title(client):
    r = client.get("/movies/search")
    assert r.status_code == 400
    assert "error" in r.get_json()


def test_search_movies_ok(client):
    r = client.get("/movies/search?title=Matrix")
    assert r.status_code == 200
    titles = [m["title"] for m in r.get_json()]
    assert "The Matrix" in titles


def test_random_movie_ok(client):
    r = client.get("/movies/random")
    assert r.status_code == 200
    data = r.get_json()
    assert data["title"] in {"The Matrix", "Mad Max: Fury Road"}


def test_people_ok(client):
    r = client.get("/people?role=Director")
    assert r.status_code == 200
    names = [p["name"] for p in r.get_json()]
    assert "Lana Wachowski" in names
    assert "Lilly Wachowski" in names


def test_actors_ok(client):
    r = client.get("/actors/Keanu")
    assert r.status_code == 200
    titles = [m["title"] for m in r.get_json()]
    assert "The Matrix" in titles


def test_directors_ok(client):
    r = client.get("/directors/George Miller")
    assert r.status_code == 200
    titles = [m["title"] for m in r.get_json()]
    assert "Mad Max: Fury Road" in titles


def test_genres_ok(client):
    r = client.get("/genres")
    assert r.status_code == 200
    genres = r.get_json()
    assert "Action" in genres
    assert "Sci-Fi" in genres


def test_top_rated_ok(client):
    r = client.get("/top-rated?genre=Action&limit=1")
    assert r.status_code == 200
    data = r.get_json()
    assert len(data) == 1
    assert data[0]["title"] in {"The Matrix", "Mad Max: Fury Road"}


def test_query_llm_ok(client):
    r = client.post("/query", json={"query": "top action"})
    assert r.status_code == 200
    data = r.get_json()
    assert data["parsed_filters"]["intent"] == "search"
    assert data["parsed_filters"]["genre"] == "Action"
    assert isinstance(data["results"], list)


def test_chat_flow_ok(client):
    r = client.post("/chat", json={"message": "Recommend some action movies"})
    assert r.status_code == 200
    data = r.get_json()
    assert "conversation_id" in data and isinstance(data["conversation_id"], int)
    assert data["assistant_message"]
    assert "context" in data
