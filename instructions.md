


python app.py

streamlit run frontend.py



| Endpoint                   | Method | Description                                |
| -------------------------- | ------ | ------------------------------------------ |
| `/`                        | GET    | Health check and endpoint list             |
| `/movies`                  | GET    | Filter by `genre`, `year`, `min_rating`    |
| `/movies/<id>`             | GET    | Movie details with genres, cast, directors |
| `/movies/search?title=`    | GET    | Search movies by title                     |
| `/movies/random`           | GET    | Get one random movie                       |
| `/people?name=&role=`      | GET    | Search any person by name or role          |
| `/actors/<name>`           | GET    | Get all movies by actor                    |
| `/directors/<name>`        | GET    | Get all movies by director                 |
| `/genres`                  | GET    | List all unique genres                     |
| `/top-rated?genre=&limit=` | GET    | Get top-rated movies (optionally by genre) |


curl -X POST http://127.0.0.1:5000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"show top 8 sci fi movies after 2017 directed by villeneuve rating above 4"}'




{
  "parsed_filters": {
    "genre": "Sci-Fi",
    "year_after": 2017,
    "min_rating": 4.0,
    "director": "Denis Villeneuve",
    "limit": 8,
    "sort_by": "rating"
  },
  "results": [ ... ]
}
