import os
import json
import sqlite3
import pandas as pd
from datasets import load_dataset

# ----------------------------
# 1. Load dataset
# ----------------------------
dataset = load_dataset("AiresPucrs/tmdb-5000-movies", split="train")
df = pd.DataFrame(dataset)

# ----------------------------
# 2. Clean and prepare basic fields
# ----------------------------
df["title"] = df["title"].fillna("NA")
df["overview"] = df["overview"].fillna("NA")

# Normalize rating (0–10 → 0–5)
df["rating"] = df["vote_average"].fillna(0.0) / 2.0

# Extract release year (fill -1 for missing)
df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year.fillna(-1).astype(int)

# ----------------------------
# 3. Helper functions
# ----------------------------
def parse_list(value):
    """Safely convert lists, dicts, or strings to a Python list."""
    if value is None or pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else [parsed]
        except Exception:
            return [value]
    return [value]

def extract_genres(genre_data):
    genres = []
    for item in parse_list(genre_data):
        if isinstance(item, dict) and "name" in item:
            genres.append(item["name"])
        elif isinstance(item, str):
            genres.append(item)
    return genres

def extract_cast(cast_data):
    cast_list = []
    for item in parse_list(cast_data):
        if isinstance(item, dict) and "name" in item:
            cast_list.append(item["name"])
        elif isinstance(item, str):
            cast_list.append(item)
    return cast_list

def extract_directors(crew_data):
    directors = []
    for item in parse_list(crew_data):
        if isinstance(item, dict) and item.get("job") == "Director":
            directors.append(item["name"])
    return directors

# ----------------------------
# 4. Create SQLite DB and tables
# ----------------------------
os.makedirs("./db", exist_ok=True)
conn = sqlite3.connect("./db/movies.db")
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS movies (
    id INTEGER PRIMARY KEY,
    title TEXT,
    year INTEGER,
    overview TEXT,
    rating REAL
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS genres (
    movie_id INTEGER,
    genre TEXT,
    FOREIGN KEY (movie_id) REFERENCES movies(id)
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS cast_directors (
    movie_id INTEGER,
    name TEXT,
    role TEXT,
    FOREIGN KEY (movie_id) REFERENCES movies(id)
)
""")

# ----------------------------
# 5. Insert data
# ----------------------------
for _, row in df.iterrows():
    movie_id = int(row["id"])
    title = row["title"]
    overview = row["overview"]
    year = None if row["year"] == -1 else int(row["year"])
    rating = float(row["rating"])

    cur.execute(
        "INSERT OR IGNORE INTO movies (id, title, year, overview, rating) VALUES (?, ?, ?, ?, ?)",
        (movie_id, title, year, overview, rating),
    )

    for g in extract_genres(row.get("genres")):
        cur.execute("INSERT INTO genres (movie_id, genre) VALUES (?, ?)", (movie_id, g))

    for c in extract_cast(row.get("cast")):
        cur.execute("INSERT INTO cast_directors (movie_id, name, role) VALUES (?, ?, ?)", (movie_id, c, "Actor"))

    for d in extract_directors(row.get("crew")):
        cur.execute("INSERT INTO cast_directors (movie_id, name, role) VALUES (?, ?, ?)", (movie_id, d, "Director"))

# ----------------------------
# 6. Commit and close
# ----------------------------
conn.commit()
conn.close()

print("✅ Data successfully saved to ./db/movies.db")
