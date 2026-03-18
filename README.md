# 🏫  AI Document Assistant

A production-grade, modular Retrieval-Augmented Generation (RAG) system designed to provide factual, high-speed responses to student queries regarding curriculum, exam schedules, and academic calendars.

## 🚀 Technical Highlights
- **High-Speed Inference:** Powered by **Groq LPU** (Llama 3.1 8B) for sub-second response latency.
- **Privacy-First Embeddings:** Uses local **HuggingFace (all-MiniLM-L6-v2)** embeddings to ensure data privacy and zero API costs for vectorization.
- **Hallucination Control:** Implements strict RAG logic to ensure responses are grounded only in verified school documentation.
- **Automated Audit:** Includes a dedicated evaluation suite using the **RAGAS framework** to measure system accuracy.

## 📂 Modular Architecture
The project is built using a decoupled, modular design to ensure scalability and maintainability:
- `src/loader.py`: Document parsing and data loading.
- `src/splitter.py`: Recursive character splitting with contextual overlap (1000/150).
- `src/embedder.py`: Local vector embedding initialization.
- `src/database.py`: Vector persistence and retrieval logic via **ChromaDB**.
- `src/guardrails.py`: Security layer for input sanitization and output redaction.

## 🛡️ Security & AI Safety (Guardrails)
This project implements a multi-layered security firewall to protect the LLM and the user:
- **Prompt Injection Defense:** Detects and blocks adversarial instructions (e.g., "Ignore previous rules").
- **PII Redaction:** Scans AI responses for sensitive data (e.g., phone numbers) using Regex-based pattern matching.
- **Topic Filtering:** Restricts the bot to academic domains, blocking queries related to restricted or dangerous topics.
- **Input Validation:** Limits query length to prevent token-exhaustion attacks.

## 📊 Performance Evaluation (RAGAS)
I implemented an automated evaluation suite (`evaluate.py`) that uses **Llama 3.3 70B** as a "Judge" to score the system across three key metrics:
- **Faithfulness:** Measures if the answer is factually supported by the retrieved context (Zero Hallucination).
- **Answer Relevancy:** Measures how well the AI response addresses the user's specific query.
- **Context Precision:** Measures the quality of the vector search retrieval from the database.
*Results are automatically logged to `evaluation_report.csv` for continuous monitoring.*

## 🛠️ Setup Instructions

1.Clone & Install:
2.pip install -r requirements.txt
3.Create a .env file with your GROQ_API_KEY.
4.Run ingestion: python ingest_main.py.
5.Run Evaluation (Optional - Audits the system accuracy using RAGAS)
'python evaluate.py'
6.Start the bot: streamlit run app.py.
