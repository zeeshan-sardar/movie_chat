import os
import json
import time
import logging
from datetime import datetime

from flask import Flask, request, jsonify
from flask_cors import CORS

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, ForeignKey, DateTime, Text,
    func
)
from sqlalchemy.orm import (
    declarative_base, relationship, sessionmaker, scoped_session
)

from openai import OpenAI

# ----------------------------
# App + Config
# ----------------------------
app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("movies-api")

DB_PATH = "./db/movies.db"
os.makedirs("./db", exist_ok=True)
DATABASE_URL = f"sqlite:///{DB_PATH}"

OPENAI_MODEL_PARSE = os.getenv("OPENAI_MODEL_PARSE", "gpt-4o-mini")
OPENAI_MODEL_REPLY = os.getenv("OPENAI_MODEL_REPLY", "gpt-4o-mini")
OPENAI_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT", "20"))          # seconds
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "2"))   # retries (total attempts = 1 + retries)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "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"))

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
# LLM: Wrappers
# ----------------------------
def _chat_with_retries(model, messages, response_format=None, temperature=0.0,
                       retries=OPENAI_MAX_RETRIES, timeout=OPENAI_TIMEOUT):
    """Minimal wrapper: retries + timeout + basic logging."""
    for attempt in range(1, retries + 2):  # 1 try + N retries
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format=response_format,
                temperature=temperature,
                timeout=timeout,
            )
            if not resp or not resp.choices or not resp.choices[0].message.content:
                raise ValueError("Empty or invalid OpenAI response")
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"OpenAI call failed (attempt {attempt}/{retries + 1}): {e}")
            if attempt > retries:
                raise
            time.sleep(1.5 * attempt)  # simple linear backoff


def llm_parse_query(nl_query: str) -> dict:
    """Parse NL query into filters via LLM, with sane defaults and clamps."""
    system = (
        "Convert movie-related user queries into STRICT JSON. "
        "Keys: intent('movie_info'|'recommend'|'search'|'unknown'), "
        'title, genre, year_after, year_before, min_rating, max_rating, actor, director, limit, '
        "sort_by('rating'|'year'|'title'). Output ONLY JSON."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Query: {nl_query}"},
    ]

    try:
        content = _chat_with_retries(
            model=OPENAI_MODEL_PARSE,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        data = json.loads(content) if content else {}
    except Exception:
        logger.info("llm_parse_query: fallback to unknown intent", exc_info=True)
        data = {}

    out = {"intent": data.get("intent", "unknown")}
    # Copy known string fields if present
    for k in ("title", "genre", "actor", "director", "sort_by"):
        v = data.get(k)
        if isinstance(v, str) and v.strip():
            out[k] = v.strip()

    # Copy numeric fields if correct type
    for k in ("year_after", "year_before", "limit"):
        v = data.get(k)
        if isinstance(v, int):
            out[k] = v
    for k in ("min_rating", "max_rating"):
        v = data.get(k)
        if isinstance(v, (int, float)):
            out[k] = float(v)

    # Normalize & clamp
    out["intent"] = out.get("intent") if out.get("intent") in {"movie_info", "recommend", "search"} else "unknown"
    if out.get("sort_by") not in {"rating", "year", "title"}:
        out["sort_by"] = "rating"
    limit = int(out.get("limit", 10))
    out["limit"] = max(1, min(limit, 100))

    return out


ASSISTANT_SYSTEM = (
    "You are a helpful movie assistant. Answer conversationally but ground your answers ONLY in the provided CONTEXT. "
    "If the user asks about movies not in context, say you don't have enough data and offer alternatives. "
    "Prefer concise, useful answers with small lists (3-10 items)."
)

def llm_answer(user_message: str, history: list, context: dict) -> str:
    """Generate a conversational answer grounded in provided context."""
    messages = [{"role": "system", "content": ASSISTANT_SYSTEM}]
    messages += history[-6:]  # keep it light
    ctx = json.dumps(context, ensure_ascii=False)[:6000]
    messages.append({"role": "system", "content": f"CONTEXT:\n{ctx}"})
    messages.append({"role": "user", "content": user_message})

    try:
        return _chat_with_retries(
            model=OPENAI_MODEL_REPLY,
            messages=messages,
            temperature=0.4,
        )
    except Exception:
        logger.info("llm_answer: returning minimal grounded fallback", exc_info=True)
        results = context.get("results") or []
        if results:
            lines = [
                f"- {m.get('title','?')} ({m.get('year','—')}) • ⭐ {m.get('rating','—')}"
                for m in results[:5]
            ]
            return "Sorry, I’m having trouble responding right now.\n\nHere are a few relevant titles:\n" + "\n".join(lines)
        return "Sorry, I’m having trouble responding right now. Please try again."


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
# Routes (with logging + error handling)
# ----------------------------
@app.route("/")
def home():
    try:
        logger.info("GET /")
        return jsonify({
            "message": "🎬 TMDB Movies API (SQLAlchemy) is running",
            "endpoints": [
                "/movies", "/movies/<id>", "/movies/search?title=", "/movies/random",
                "/people?name=&role=", "/actors/<name>", "/directors/<name>",
                "/genres", "/top-rated?genre=&limit=", "/query (LLM filters)", "/chat (conversational)"
            ]
        })
    except Exception as e:
        logger.exception(f"Error in /: {e}")
        return jsonify({"error": "Unexpected error"}), 500


@app.route("/movies", methods=["GET"])
def get_movies():
    try:
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

        results = [movie_brief_dict(m) for m in q.all()]
        logger.info(f"GET /movies - {len(results)} results (genre={genre}, year={year}, min_rating={min_rating})")
        return jsonify(results)
    except Exception as e:
        logger.exception(f"Error in /movies: {e}")
        return jsonify({"error": "Failed to fetch movies"}), 500


@app.route("/movies/<int:movie_id>", methods=["GET"])
def get_movie(movie_id):
    try:
        db = request.db
        m = db.get(Movie, movie_id)
        if not m:
            logger.warning(f"GET /movies/{movie_id} - not found")
            return jsonify({"error": "Movie not found"}), 404
        logger.info(f"GET /movies/{movie_id} - fetched")
        return jsonify(movie_full_dict(db, m))
    except Exception as e:
        logger.exception(f"Error in /movies/{movie_id}: {e}")
        return jsonify({"error": "Failed to fetch movie"}), 500


@app.route("/movies/search", methods=["GET"])
def search_movies():
    try:
        db = request.db
        title = request.args.get("title", "")
        if not title or not title.strip():
            return jsonify({"error": "Missing 'title'"}), 400
        q = db.query(Movie).filter(Movie.title.ilike(f"%{title}%")).order_by(Movie.rating.desc()).limit(50)
        results = [{"id": m.id, "title": m.title, "year": m.year, "rating": m.rating} for m in q.all()]
        logger.info(f"GET /movies/search - title='{title}', results={len(results)}")
        return jsonify(results)
    except Exception as e:
        logger.exception(f"Error in /movies/search: {e}")
        return jsonify({"error": "Search failed"}), 500


@app.route("/movies/random", methods=["GET"])
def random_movie():
    try:
        db = request.db
        m = db.query(Movie).order_by(func.random()).limit(1).first()
        if not m:
            logger.warning("GET /movies/random - no movies found")
            return jsonify({"error": "No movies found"})
        logger.info(f"GET /movies/random - {m.title} ({m.id})")
        return jsonify({"id": m.id, "title": m.title, "year": m.year, "rating": m.rating})
    except Exception as e:
        logger.exception(f"Error in /movies/random: {e}")
        return jsonify({"error": "Random fetch failed"}), 500


@app.route("/people", methods=["GET"])
def get_people():
    try:
        db = request.db
        name = request.args.get("name", "")
        role = request.args.get("role", "")
        q = db.query(CastDirector).distinct()
        if name:
            q = q.filter(CastDirector.name.ilike(f"%{name}%"))
        if role:
            q = q.filter(CastDirector.role == role)
        rows = q.all()
        results = [{"name": r.name, "role": r.role, "movie_id": r.movie_id} for r in rows]
        logger.info(f"GET /people - name='{name}', role='{role}', results={len(results)}")
        return jsonify(results)
    except Exception as e:
        logger.exception(f"Error in /people: {e}")
        return jsonify({"error": "Failed to fetch people"}), 500


@app.route("/actors/<string:name>", methods=["GET"])
def get_actor_movies(name):
    try:
        db = request.db
        q = db.query(Movie).join(CastDirector, CastDirector.movie_id == Movie.id) \
            .filter(CastDirector.role == "Actor", CastDirector.name.ilike(f"%{name}%")) \
            .order_by(Movie.rating.desc())
        results = [{"id": m.id, "title": m.title, "year": m.year, "rating": m.rating} for m in q.all()]
        logger.info(f"GET /actors/{name} - results={len(results)}")
        return jsonify(results)
    except Exception as e:
        logger.exception(f"Error in /actors/{name}: {e}")
        return jsonify({"error": "Failed to fetch actor movies"}), 500


@app.route("/directors/<string:name>", methods=["GET"])
def get_director_movies(name):
    try:
        db = request.db
        q = db.query(Movie).join(CastDirector, CastDirector.movie_id == Movie.id) \
            .filter(CastDirector.role == "Director", CastDirector.name.ilike(f"%{name}%")) \
            .order_by(Movie.rating.desc())
        results = [{"id": m.id, "title": m.title, "year": m.year, "rating": m.rating} for m in q.all()]
        logger.info(f"GET /directors/{name} - results={len(results)}")
        return jsonify(results)
    except Exception as e:
        logger.exception(f"Error in /directors/{name}: {e}")
        return jsonify({"error": "Failed to fetch director movies"}), 500


@app.route("/genres", methods=["GET"])
def get_genres():
    try:
        db = request.db
        rows = db.query(Genre.genre).distinct().order_by(Genre.genre.asc()).all()
        results = [r[0] for r in rows]
        logger.info(f"GET /genres - {len(results)} genres")
        return jsonify(results)
    except Exception as e:
        logger.exception(f"Error in /genres: {e}")
        return jsonify({"error": "Failed to fetch genres"}), 500


@app.route("/top-rated", methods=["GET"])
def get_top_rated():
    try:
        db = request.db
        genre = request.args.get("genre")
        limit = request.args.get("limit", type=int, default=20)
        q = db.query(Movie).distinct().outerjoin(Genre, Genre.movie_id == Movie.id)
        if genre:
            q = q.filter(Genre.genre.ilike(f"%{genre}%"))
        q = q.order_by(Movie.rating.desc()).limit(limit)
        results = [{"id": m.id, "title": m.title, "year": m.year, "rating": m.rating} for m in q.all()]
        logger.info(f"GET /top-rated - genre={genre}, limit={limit}, results={len(results)}")
        return jsonify(results)
    except Exception as e:
        logger.exception(f"Error in /top-rated: {e}")
        return jsonify({"error": "Failed to fetch top rated"}), 500


# -------- LLM /query (filters) ----------
@app.route("/query", methods=["POST"])
def query_llm():
    try:
        db = request.db
        data = request.get_json(force=True) or {}
        nl_query = (data.get("query") or "").strip()
        if not nl_query:
            return jsonify({"error": "Missing 'query'"}), 400

        filters = llm_parse_query(nl_query)
        results = retrieve_by_filters(db, filters)
        logger.info(f"POST /query - intent={filters.get('intent')} results={len(results)}")
        return jsonify({"parsed_filters": filters, "results": results})
    except Exception as e:
        logger.exception(f"Error in /query: {e}")
        return jsonify({"error": "Failed to process query"}), 500


# -------- LLM /chat (conversational) ----------
@app.route("/chat", methods=["POST"])
def chat():
    try:
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
        logger.info(f"parsed filters: {filters}, results count: {len(results)}")
        assistant = llm_answer(user_msg, history, context)

        add_message(db, conv_id, "assistant", assistant)

        logger.info(f"POST /chat - conv_id={conv_id} intent={filters.get('intent')} results={len(results)}")
        return jsonify({
            "conversation_id": conv_id,
            "assistant_message": assistant,
            "context": context
        })
    except Exception as e:
        logger.exception(f"Error in /chat: {e}")
        return jsonify({"error": "Chat failed"}), 500


# ----------------------------
# Global unexpected error handler (optional)
# ----------------------------
@app.errorhandler(Exception)
def handle_unexpected_error(e):
    logger.exception(f"Unhandled exception: {e}")
    return jsonify({"error": "Internal Server Error"}), 500


# ----------------------------
# Run app
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)
