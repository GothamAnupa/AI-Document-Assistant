# AI Document Assistant

An AI-powered RAG app for students to ask factual questions from documents and college/school websites.

## Purpose

Built to help students quickly find answers about:
- notes and study material
- notices and circulars
- exam schedules and results
- admissions and course details
- college and school websites

## Features

- Upload `.txt`, `.md`, `.log`, `.csv`, `.pdf`, and `.docx` files
- Paste website links and query their content
- Ask follow-up questions in chat
- Get answers backed by source references
- Reset or rebuild the knowledge base anytime

## How It Works

1. You upload files or paste links.
2. The app extracts text and splits it into chunks.
3. Chunks are stored in a vector database.
4. Your question is matched against the indexed content.
5. The model answers using only the retrieved context.

## Project Structure

- `app.py` - Streamlit UI and chat flow
- `src/knowledge.py` - loading, scraping, chunking, and retrieval
- `src/guardrails.py` - basic input/output safety checks
- `src/config.py` - model and storage settings

## Run Locally

1. `pip install -r requirements.txt`
2. Create `.env` and set `GROQ_API_KEY=...`
3. `streamlit run app.py`

## Usage

1. Add documents or links in the sidebar.
2. Click `Build knowledge base`.
3. Ask a question in the chat box.

## Supported Sources

- Local documents: `.txt`, `.md`, `.log`, `.csv`, `.pdf`, `.docx`
- Web pages from pasted URLs

## Notes

- The app answers only from indexed sources.
- It is designed mainly for student and academic use cases.
- Streamlit Cloud uses a disabled file watcher to avoid transformer import noise.
