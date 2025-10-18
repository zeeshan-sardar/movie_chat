# Movie Chat — Conversational Movie Assistant

![Tests](https://github.com/zeeshan-sardar/movie_chat/actions/workflows/tests.yml/badge.svg)

## Project Overview

**Movie Chat** is an intelligent conversational platform that allows users to explore and discover movies through natural language interaction. Instead of using rigid filters or dropdowns, users can simply *ask questions the way they talk* — such as “show me top-rated sci-fi movies after 2015” or “who directed Inception?” The backend, built with **Flask** and **SQLAlchemy**, manages movie data and conversational history stored in **SQLite**, while **OpenAI GPT models** handle natural language understanding and response generation. The **Streamlit frontend** provides an intuitive chat interface that maintains context across turns, creating a seamless, human-like conversation experience for exploring movie data.

This project demonstrates how **LLM-powered agents** can be integrated with structured databases to provide contextual, grounded, and interactive responses — bridging the gap between traditional search systems and conversational AI assistants.

![Alt text](./figures/arhchitecture.png)


## Tech Stack

- Backend: Flask, SQLAlchemy, OpenAI API
- Frontend: Streamlit
- Database: SQLite
- Testing: pytest, pytest-cov
- Language: Python 3.12+


## Setup & Installation
1. Clone the Repository
``` 
git clone https://github.com/zeeshan-sardar/movie_chat.git  

cd movie_chat 
```


2. Create a Virtual Environment
```
python -m venv venv
source venv/bin/activate  
```
3. Install Dependencies
```
pip install -r requirements.txt
```
4. Build the Database
```
mkdir db
python data_db.py
```
5. Environment Variables
Create a .env file in the project root with:
```
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL_PARSE=gpt-4o-mini
OPENAI_MODEL_REPLY=gpt-4o-mini
API_URL=http://127.0.0.1:5000
```
5. Run the Backend
```
python app.py
```
6. Run the Frontend
```
streamlit run frontend.py
```

## API Endpoints
| Endpoint                   | Method | Description                          |
| -------------------------- | ------ | ------------------------------------ |
| `/`                        | GET    | Health check                         |
| `/movies`                  | GET    | Filter by genre, year, or min_rating |
| `/movies/<id>`             | GET    | Get movie details                    |
| `/movies/search?title=`    | GET    | Search by title                      |
| `/movies/random`           | GET    | Random movie                         |
| `/people?name=&role=`      | GET    | List people                          |
| `/actors/<name>`           | GET    | Movies by actor                      |
| `/directors/<name>`        | GET    | Movies by director                   |
| `/genres`                  | GET    | List all genres                      |
| `/top-rated?genre=&limit=` | GET    | Top-rated movies                     |
| `/query`                   | POST   | Parse natural-language query         |
| `/chat`                    | POST   | Conversational chat endpoint         |

## Demo
![Alt text](./figures/demo.gif)