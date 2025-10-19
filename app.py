import os
import json
from datetime import datetime

from flask import Flask, request, jsonify
from flask_cors import CORS

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, ForeignKey, DateTime, Text,
    func, select
)
from sqlalchemy.orm import (
    declarative_base, relationship, sessionmaker, scoped_session
)
from sqlalchemy.exc import NoResultFound

from openai import OpenAI

# ----------------------------
# App + Config
# ----------------------------
app = Flask(__name__)
CORS(app)

DB_PATH = "./db/movies.db"
os.makedirs("./db", exist_ok=True)
DATABASE_URL = f"sqlite:///{DB_PATH}"


OPENAI_MODEL_PARSE = os.getenv("OPENAI_MODEL_PARSE", "gpt-4o-mini")
OPENAI_MODEL_REPLY = os.getenv("OPENAI_MODEL_REPLY", "gpt-4o-mini")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ----------------------------
# SQLAlchemy setup
# ----------------------------
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = scoped_session(sessionmaker(bind=engine, autoflush=False, autocommit=False))
Base = declarative_base()


# ----------------------------
# ORM Models (match existing schema)
# ----------------------------
class Movie(Base):
    __tablename__ = "movies"
    id = Column(Integer, primary_key=True)
    title = Column(String)
    year = Column(Integer, nullable=True)
    overview = Column(Text)
    rating = Column(Float)

    genres = relationship("Genre", back_populates="movie", cascade="all, delete-orphan")
    cast_directors = relationship("CastDirector", back_populates="movie", cascade="all, delete-orphan")


class Genre(Base):
    __tablename__ = "genres"
    movie_id = Column(Integer, ForeignKey("movies.id"), primary_key=True)
    genre = Column(String, primary_key=True)

    movie = relationship("Movie", back_populates="genres")


class CastDirector(Base):
    __tablename__ = "cast_directors"
    movie_id = Column(Integer, ForeignKey("movies.id"), primary_key=True)
    name = Column(String, primary_key=True)
    role = Column(String, primary_key=True)  # 'Actor' or 'Director'

    movie = relationship("Movie", back_populates="cast_directors")


class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"))
    role = Column(String)  # 'user' | 'assistant'
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(engine)


# ----------------------------
# Request-scoped session
# ----------------------------
@app.before_request
def _create_session():
    request.db = SessionLocal()

@app.teardown_request
def _remove_session(exc):
    db = getattr(request, "db", None)
    if db is not None:
        if exc:
            db.rollback()
        db.close()
    SessionLocal.remove()


# ----------------------------
# Helpers
# ----------------------------
def movie_brief_dict(m: Movie):
    return {
        "id": m.id, "title": m.title, "year": m.year, "rating": m.rating,
        "overview": m.overview
    }

def movie_full_dict(db, m: Movie):
    return {
        "id": m.id,
        "title": m.title,
        "year": m.year,
        "overview": m.overview,
        "rating": m.rating,
        "genres": [g.genre for g in m.genres],
        "cast": [cd.name for cd in m.cast_directors if cd.role == "Actor"],
        "directors": [cd.name for cd in m.cast_directors if cd.role == "Director"],
    }

def create_conversation(db):
    conv = Conversation()
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return conv.id

def add_message(db, conversation_id, role, content):
    db.add(Message(conversation_id=conversation_id, role=role, content=content))
    db.commit()

def get_history(db, conversation_id):
    rows = db.query(Message).filter(Message.conversation_id == conversation_id).order_by(Message.id.asc()).all()
    return [{"role": r.role, "content": r.content} for r in rows]


# ----------------------------
# LLM: parse + answer
# ----------------------------
def llm_parse_query(nl_query: str) -> dict:
    system = (
        "You convert movie-related user queries into STRICT JSON. "
        "Allowed keys: intent ('movie_info'|'recommend'|'search'|'unknown'), "
        "title, genre, year_after, year_before, min_rating, max_rating, actor, director, limit, sort_by "
        "('rating'|'year'|'title'). Omit keys not present. Output ONLY JSON."
    )
    user = f"Query: {nl_query}"

    resp = client.chat.completions.create(
        model=OPENAI_MODEL_PARSE,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    try:
        parsed = json.loads(resp.choices[0].message.content)
    except Exception:
        parsed = {"intent": "unknown"}

    out = {"intent": parsed.get("intent", "unknown")}
    for k in ("title", "genre", "actor", "director", "sort_by"):
        v = parsed.get(k)
        if isinstance(v, str) and v.strip():
            out[k] = v.strip()
    for k in ("year_after", "year_before", "limit"):
        v = parsed.get(k)
        if isinstance(v, int):
            out[k] = v
    for k in ("min_rating", "max_rating"):
        v = parsed.get(k)
        if isinstance(v, (int, float)):
            out[k] = float(v)
    if "limit" not in out:
        out["limit"] = 10
    return out


ASSISTANT_SYSTEM = (
    "You are a helpful movie assistant. Answer conversationally but ground your answers ONLY in the provided CONTEXT. "
    "If the user asks about movies not in context, say you don't have enough data and offer alternatives. "
    "Prefer concise, useful answers with small lists (3-10 items)."
)

def llm_answer(user_message: str, history: list, context: dict) -> str:
    messages = [{"role": "system", "content": ASSISTANT_SYSTEM}]
    for h in history[-6:]:
        messages.append({"role": h["role"], "content": h["content"]})
    context_text = json.dumps(context, ensure_ascii=False, indent=2)[:6000]
    messages.append({"role": "system", "content": f"CONTEXT:\n{context_text}"})
    messages.append({"role": "user", "content": user_message})

    resp = client.chat.completions.create(
        model=OPENAI_MODEL_REPLY,
        messages=messages,
        temperature=0.4,
    )
    return resp.choices[0].message.content.strip()


# ----------------------------
# Retrieval using SQLAlchemy
# ----------------------------
def retrieve_by_filters(db, f: dict):
    q = db.query(Movie).distinct() \
        .outerjoin(Genre, Genre.movie_id == Movie.id) \
        .outerjoin(CastDirector, CastDirector.movie_id == Movie.id)

    if f.get("title"):
        q = q.filter(Movie.title.ilike(f"%{f['title']}%"))
    if f.get("genre"):
        q = q.filter(Genre.genre.ilike(f"%{f['genre']}%"))
    if f.get("year_after"):
        q = q.filter(Movie.year >= f["year_after"])
    if f.get("year_before"):
        q = q.filter(Movie.year <= f["year_before"])
    if f.get("min_rating"):
        q = q.filter(Movie.rating >= f["min_rating"])
    if f.get("max_rating"):
        q = q.filter(Movie.rating <= f["max_rating"])
    if f.get("actor"):
        q = q.filter(CastDirector.role == "Actor", CastDirector.name.ilike(f"%{f['actor']}%"))
    if f.get("director"):
        q = q.filter(CastDirector.role == "Director", CastDirector.name.ilike(f"%{f['director']}%"))

    sort = f.get("sort_by", "rating")
    if sort == "year":
        q = q.order_by(Movie.year.desc())
    elif sort == "title":
        q = q.order_by(Movie.title.asc())
    else:
        q = q.order_by(Movie.rating.desc())

    q = q.limit(int(f.get("limit", 10)))
    return [movie_brief_dict(m) for m in q.all()]


# ----------------------------
# Routes
# ----------------------------
@app.route("/")
def home():
    return jsonify({
        "message": "🎬 TMDB Movies API (SQLAlchemy) is running",
        "endpoints": [
            "/movies", "/movies/<id>", "/movies/search?title=", "/movies/random",
            "/people?name=&role=", "/actors/<name>", "/directors/<name>",
            "/genres", "/top-rated?genre=&limit=", "/query (LLM filters)", "/chat (conversational)"
        ]
    })


@app.route("/movies", methods=["GET"])
def get_movies():
    db = request.db
    genre = request.args.get("genre")
    year = request.args.get("year")
    min_rating = request.args.get("min_rating", type=float, default=0.0)

    q = db.query(Movie).distinct().outerjoin(Genre, Genre.movie_id == Movie.id)
    if genre:
        q = q.filter(Genre.genre.ilike(f"%{genre}%"))
    if year:
        q = q.filter(Movie.year == int(year))
    if min_rating:
        q = q.filter(Movie.rating >= min_rating)
    q = q.order_by(Movie.rating.desc()).limit(50)
    return jsonify([movie_brief_dict(m) for m in q.all()])


@app.route("/movies/<int:movie_id>", methods=["GET"])
def get_movie(movie_id):
    db = request.db
    m = db.get(Movie, movie_id)
    if not m:
        return jsonify({"error": "Movie not found"}), 404
    return jsonify(movie_full_dict(db, m))


@app.route("/movies/search", methods=["GET"])
def search_movies():
    db = request.db
    title = request.args.get("title", "")
    if not title:
        return jsonify({"error": "Missing 'title'"}), 400
    q = db.query(Movie).filter(Movie.title.ilike(f"%{title}%")).order_by(Movie.rating.desc()).limit(50)
    return jsonify([{"id": m.id, "title": m.title, "year": m.year, "rating": m.rating} for m in q.all()])


@app.route("/movies/random", methods=["GET"])
def random_movie():
    db = request.db
    m = db.query(Movie).order_by(func.random()).limit(1).first()
    return jsonify({"error": "No movies found"} if not m else {
        "id": m.id, "title": m.title, "year": m.year, "rating": m.rating
    })


@app.route("/people", methods=["GET"])
def get_people():
    db = request.db
    name = request.args.get("name", "")
    role = request.args.get("role", "")

    q = db.query(CastDirector).distinct()
    if name:
        q = q.filter(CastDirector.name.ilike(f"%{name}%"))
    if role:
        q = q.filter(CastDirector.role == role)
    rows = q.all()
    return jsonify([{"name": r.name, "role": r.role, "movie_id": r.movie_id} for r in rows])


@app.route("/actors/<string:name>", methods=["GET"])
def get_actor_movies(name):
    db = request.db
    q = db.query(Movie).join(CastDirector, CastDirector.movie_id == Movie.id) \
        .filter(CastDirector.role == "Actor", CastDirector.name.ilike(f"%{name}%")) \
        .order_by(Movie.rating.desc())
    return jsonify([{"id": m.id, "title": m.title, "year": m.year, "rating": m.rating} for m in q.all()])


@app.route("/directors/<string:name>", methods=["GET"])
def get_director_movies(name):
    db = request.db
    q = db.query(Movie).join(CastDirector, CastDirector.movie_id == Movie.id) \
        .filter(CastDirector.role == "Director", CastDirector.name.ilike(f"%{name}%")) \
        .order_by(Movie.rating.desc())
    return jsonify([{"id": m.id, "title": m.title, "year": m.year, "rating": m.rating} for m in q.all()])


@app.route("/genres", methods=["GET"])
def get_genres():
    db = request.db
    rows = db.query(Genre.genre).distinct().order_by(Genre.genre.asc()).all()
    return jsonify([r[0] for r in rows])


@app.route("/top-rated", methods=["GET"])
def get_top_rated():
    db = request.db
    genre = request.args.get("genre")
    limit = request.args.get("limit", type=int, default=20)
    q = db.query(Movie).distinct().outerjoin(Genre, Genre.movie_id == Movie.id)
    if genre:
        q = q.filter(Genre.genre.ilike(f"%{genre}%"))
    q = q.order_by(Movie.rating.desc()).limit(limit)
    return jsonify([{"id": m.id, "title": m.title, "year": m.year, "rating": m.rating} for m in q.all()])


# -------- LLM /query (filters) ----------
@app.route("/query", methods=["POST"])
def query_llm():
    db = request.db
    data = request.get_json(force=True) or {}
    nl_query = (data.get("query") or "").strip()
    if not nl_query:
        return jsonify({"error": "Missing 'query'"}), 400
    filters = llm_parse_query(nl_query)
    results = retrieve_by_filters(db, filters)
    return jsonify({"parsed_filters": filters, "results": results})


# -------- LLM /chat (conversational) ----------
@app.route("/chat", methods=["POST"])
def chat():
    db = request.db
    body = request.get_json(force=True) or {}
    user_msg = (body.get("message") or "").strip()
    conv_id = body.get("conversation_id")

    if not user_msg:
        return jsonify({"error": "Missing 'message'"}), 400

    if not conv_id:
        conv_id = create_conversation(db)
    add_message(db, conv_id, "user", user_msg)

    filters = llm_parse_query(user_msg)
    results = retrieve_by_filters(db, filters) if filters.get("intent") != "unknown" else []

    # fallback exact-title for movie_info
    if filters.get("intent") == "movie_info" and not results and filters.get("title"):
        m = db.query(Movie).filter(func.lower(Movie.title) == filters["title"].lower()).first()
        if m:
            results = [movie_brief_dict(m)]

    history = get_history(db, conv_id)
    context = {"parsed_filters": filters, "results": results}
    assistant = llm_answer(user_msg, history, context)

    add_message(db, conv_id, "assistant", assistant)

    return jsonify({
        "conversation_id": conv_id,
        "assistant_message": assistant,
        "context": context
    })


# ----------------------------
# Run app
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)
