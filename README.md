# 🎬 YouTube RAG Assistant

A **Retrieval-Augmented Generation (RAG)** application that lets you ask natural language questions about any YouTube video — powered by LangChain, HuggingFace, FAISS, FastAPI, and Streamlit.



## Overview

YouTube RAG Assistant fetches the transcript of any YouTube video, splits it into semantic chunks, indexes them using vector embeddings, and lets you query the content using a conversational LLM. Instead of watching a 2-hour video, you can ask:

> *"What does the speaker say about climate change?"*
> *"Summarize the main points of this video."*
> *"What examples did they give for machine learning?"*

The system retrieves only the most relevant chunks from the transcript and feeds them as context to the LLM, ensuring accurate, grounded answers.

---

### Request Flow

```
User enters Video ID
        │
        ▼
FastAPI /load
   Fetch transcript via YouTubeTranscriptApi
  
        │
        ▼
User asks a question
        │
        ▼
FastAPI /query
  1. Retrieve top-2 relevant chunks from FAISS
  2. Format into PromptTemplate (context + query)
  3. Return answer + retrieved chunks
        │
        ▼
Streamlit renders answer in chat UI
```

---

## Tech Stack

| Layer | Technology | Purpose |
| **Frontend** | Streamlit | Chat UI, sidebar controls, session state |
| **LLM** | HuggingFaceH4/zephyr-7b-beta | Answer generation |
| **Embeddings** | all-MiniLM-L6-v2 | Semantic vector embeddings |
| **Vector Store** | FAISS | Fast similarity search |
| **RAG Framework** | LangChain | Chain orchestration, prompt templates, retrievers |
| **Transcript** | youtube-transcript-api | Fetching YouTube subtitles |
| **Text Splitting** | RecursiveCharacterTextSplitter | Chunking long transcripts |

---

## Project Structure

```
youtube-rag-assistant/
│
├── frontend.py         # Streamlit app — chat UI, sidebar, session state
├── requirements.txt    # Python dependencies
├── .env                # Environment variables (not committed)
└── README.md
```


## How It Works

### 1. Transcript Fetching
The `youtube-transcript-api` library fetches the auto-generated or manually uploaded English subtitles for a given video ID. All text chunks are joined into a single transcript string.

### 2. Text Splitting
`RecursiveCharacterTextSplitter` breaks the transcript into overlapping chunks:
- **Chunk size:** 880 characters
- **Overlap:** 150 characters (ensures context continuity across chunk boundaries)

### 3. Embedding & Indexing
Each chunk is embedded using `all-MiniLM-L6-v2` (a lightweight, fast sentence transformer) and stored in a **FAISS** flat index. The index is cached in memory per video ID — loading the same video twice skips re-indexing.

### 4. Retrieval
On each query, FAISS retrieves the **top-2 most semantically similar chunks** using cosine similarity.

### 5. Generation (RAG Chain)
A LangChain `RunnableParallel` chain:
```
Query ──┬──► Retriever ──► Format Context ──┐
        │                                    ▼
        └────────────────────────► PromptTemplate ──► Zephyr-7B ──► StrOutputParser ──► Answer
```
The prompt instructs the LLM to answer only from the provided context, or say *"Out of scope"* if the answer isn't there.

## Features

- ✅ **No re-indexing** — FAISS index is cached per video ID during the session
- ✅ **Full chat history** — previous Q&A pairs persist within the Streamlit session
- ✅ **Context transparency** — each answer shows the retrieved transcript chunks used
- ✅ **Flexible input** — accepts both YouTube URLs and bare video IDs
- ✅ **Out-of-scope handling** — LLM is instructed to declare when the answer isn't in the transcript
- ✅ **Swagger UI** — FastAPI auto-generates interactive API docs at `/docs`
- ✅ **Pure Python UI** — no HTML, CSS, or JavaScript required

---

## Limitations

- **English transcripts only** — the transcript fetch is configured for `languages=["en"]`. Videos without English subtitles will fail to load.
- **In-memory storage** — the FAISS index is not persisted to disk; restarting the backend clears all loaded videos.
- **Top-2 retrieval** — only 2 chunks are retrieved per query. For very long or complex videos, this may miss relevant context.
- **HuggingFace rate limits** — the free HuggingFace Inference API has rate limits. For heavy usage, consider a dedicated endpoint or a local model.
- **No streaming** — the LLM response is returned in full after generation completes (no token streaming in the UI).
