Below is a minimal, **hours‑level** RAG demo plan: install steps + a single Python script using LangChain + Anthropic over a small CSV of research papers (e.g., arXiv subset). You can swap in your own CSV with `title` and `abstract` columns.

***

## 1. One‑time installation and setup

### 1.1. Python and virtualenv

```bash
# Check Python (>=3.10 recommended)
python --version

# Create and activate virtualenv (Linux/macOS)
python -m venv venv
source venv/bin/activate

# On Windows
# python -m venv venv
# venv\Scripts\activate
```

### 1.2. Install required packages

```bash
pip install --upgrade pip

# Core
pip install langchain langchain-community pandas

# Vector store (Chroma, simple local DB)
pip install chromadb

# Anthropic + LangChain integration
pip install anthropic langchain-anthropic

# Optional: python-dotenv if you want .env
pip install python-dotenv

pip install sentence-transformers
```

LangChain’s docs recommend installing `langchain` and integration packages separately. [docs.langchain](https://docs.langchain.com/oss/python/langchain/install)
Anthropic integration in LangChain uses `langchain-anthropic` and requires an `ANTHROPIC_API_KEY` env variable. [docs.langchain](https://docs.langchain.com/oss/python/integrations/chat/anthropic)

### 1.3. Set Anthropic API key

Get your key from Anthropic console and set environment variable (example: Linux/macOS):

```bash
export ANTHROPIC_API_KEY="your_api_key_here"
```

On Windows (PowerShell):

```powershell
$env:ANTHROPIC_API_KEY="your_api_key_here"
```

LangChain examples assume `ANTHROPIC_API_KEY` is set in the environment. [7x.mintlify](https://7x.mintlify.app/oss/python/integrations/chat/anthropic)

***

## 2. Prepare a tiny research‑paper CSV

For a fast demo, use a **small CSV** stored locally, e.g. `papers.csv` with at least:

```text
id,title,abstract
1,Quantum error correction in superconducting qubits,This paper discusses...
2,Variational quantum eigensolver for chemistry,We propose...
...
```

You can manually create 20–50 rows, or export a small subset from an arXiv Kaggle dataset into this format. [youtube](https://www.youtube.com/watch?v=krkS9u140tM)

Place `papers.csv` in the same folder as your script.

***

## 3. Single‑file RAG demo using LangChain + Anthropic

Save this as `rag_demo.py`. It:

- Loads `papers.csv`  
- Builds a Chroma vector store on titles + abstracts  
- Wraps Anthropic Claude via LangChain  
- Runs an interactive query loop

```python
import os
import pandas as pd

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document


# ---------- 0. CONFIG ----------
DATA_PATH = "papers.csv"
CHROMA_DIR = "chroma_db"
ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"  # change if needed


def load_papers(path: str):
    df = pd.read_csv(path)
    # Minimal schema check
    assert "title" in df.columns and "abstract" in df.columns, \
        "CSV must have 'title' and 'abstract' columns"
    docs = []
    for _, row in df.iterrows():
        content = f"Title: {row['title']}\nAbstract: {row['abstract']}"
        metadata = {
            "title": row["title"],
        }
        docs.append(Document(page_content=content, metadata=metadata))
    return docs


def build_vector_store(docs):
    # Simple splitter so long abstracts are chunked
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )
    split_docs = splitter.split_documents(docs)

    # Lightweight open-source embedding model (no external API)
    embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={"normalize_embeddings": True},
    )

    vectordb = Chroma.from_documents(
        split_docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    vectordb.persist()
    return vectordb


def load_vector_store():
    embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={"normalize_embeddings": True},
    )
    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    return vectordb


def get_llm():
    # Anthropic chat model via LangChain
    if "ANTHROPIC_API_KEY" not in os.environ:
        raise RuntimeError("Please set ANTHROPIC_API_KEY env variable.")
    llm = ChatAnthropic(
        model=ANTHROPIC_MODEL,
        temperature=0.2,
        max_tokens=512,
    )
    return llm


def answer_query(query: str, vectordb, llm, k: int = 4):
    # 1. Retrieve relevant chunks
    docs = vectordb.similarity_search(query, k=k)

    # 2. Build context string
    context_parts = []
    for i, d in enumerate(docs, start=1):
        context_parts.append(
            f"[Document {i}] {d.page_content}"
        )
    context = "\n\n".join(context_parts)

    # 3. Build prompt for Claude
    system_msg = SystemMessage(
        content=(
            "You are a helpful research assistant. "
            "Answer the user's question using ONLY the provided documents. "
            "Cite the document numbers like [Document 1], [Document 2] "
            "where relevant. If the answer is not in the documents, say so."
        )
    )
    human_msg = HumanMessage(
        content=(
            f"User question:\n{query}\n\n"
            f"Here are relevant research paper snippets:\n{context}"
        )
    )

    resp = llm.invoke([system_msg, human_msg])
    return resp.content, docs


def main():
    # 1. Build or load vector store
    if not os.path.exists(CHROMA_DIR):
        print("Building vector store from CSV...")
        docs = load_papers(DATA_PATH)
        vectordb = build_vector_store(docs)
        print("Vector store built.")
    else:
        print("Loading existing vector store...")
        vectordb = load_vector_store()

    # 2. Init Anthropic LLM
    llm = get_llm()

    # 3. Simple REPL
    print("RAG demo ready. Type your question (or 'exit'):")
    while True:
        query = input("\nQuestion: ")
        if query.lower() in ("exit", "quit"):
            break
        answer, docs = answer_query(query, vectordb, llm)
        print("\n--- Answer ---")
        print(answer)
        print("\n--- Top sources ---")
        for i, d in enumerate(docs, start=1):
            print(f"[Document {i}] {d.metadata.get('title')}")


if __name__ == "__main__":
    main()
```

- `ChatAnthropic` is the official LangChain integration for Anthropic chat models. [sj-langchain.readthedocs](https://sj-langchain.readthedocs.io/en/latest/llms/langchain.llms.anthropic.Anthropic.html)
- Chroma is a lightweight in‑process vector store; LangChain provides the `Chroma` wrapper. [latenode](https://latenode.com/blog/ai-frameworks-technical-infrastructure/langchain-setup-tools-agents-memory/how-to-install-langchain-complete-python-setup-guide-troubleshooting-2025)

***

## 4. How to run the demo

```bash
# Activate venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Ensure the CSV is present
ls papers.csv

# Run
python rag_demo.py
```

Then type example questions, e.g.:

- “What are common approaches to variational quantum algorithms?”  
- “Which papers discuss error correction in superconducting qubits?”  

The console will show:

- Answer generated by Claude  
- Top retrieved paper titles as “sources”

***

If you want, next step I can give you a version wrapped in a small FastAPI endpoint so you can call it from any UI later without changing the core RAG logic.


Here are concise, domain‑style questions you can use to showcase the RAG demo with research papers (assuming your `papers.csv` is from arXiv abstracts).

***

## General “overview” questions

- What are the main research topics covered in these papers?  
- Summarize the key themes that appear most frequently in these abstracts.  
- What problems are these papers trying to solve?  
- What methods or techniques are most commonly mentioned across the papers?  

***

## Methodology and techniques

- What machine learning methods are used in these papers, and for what purposes?  
- How do the papers describe their experimental setup or evaluation methods?  
- What optimization techniques are mentioned, and in what context?  
- Which types of models or algorithms appear most often in these abstracts?  

***

## Results, challenges, and limitations

- What kinds of results or performance improvements do these papers claim?  
- What challenges or limitations do the authors commonly report?  
- What open research problems or future work directions are highlighted?  
- Are there recurring bottlenecks or constraints mentioned across the papers?  

***

## Comparative / synthesis questions

- Compare the different approaches discussed for solving the same type of problem.  
- How do the papers differ in their assumptions or problem formulations?  
- What trends do you observe over time in the techniques or topics mentioned?  
- Which applications or domains are most frequently targeted in these papers?  

***

## “Act like a research assistant” questions

- I want to start researching this field. What should I read first from these papers?  
- Summarize these papers as if I am a beginner in this area.  
- Extract key terms and concepts from these abstracts and explain them briefly.  
- Create a short literature review paragraph based only on these papers.