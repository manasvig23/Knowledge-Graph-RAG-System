import os
import pandas as pd

from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document



# ---------- 0. CONFIG ----------
DATA_PATH = "papers.csv"      # created from your arxiv_data.csv
CHROMA_DIR = "chroma_db"      # folder for vector store persistence
ANTHROPIC_MODEL = "claude-sonnet-4-5"  # change to a valid model you have access to


def load_papers(path: str):
    df = pd.read_csv(path)
    # Expect columns created by your generator script
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
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )
    split_docs = splitter.split_documents(docs)

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
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Check your .env file.")

    llm = ChatAnthropic(
        model=ANTHROPIC_MODEL,
        temperature=0.2,
        max_tokens=512,
        api_key=api_key,
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
    # Load .env into environment
    load_dotenv()  # reads .env in current directory

    # Build or load vector store
    if not os.path.exists(CHROMA_DIR):
        print("Building vector store from CSV...")
        docs = load_papers(DATA_PATH)
        vectordb = build_vector_store(docs)
        print("Vector store built.")
    else:
        print("Loading existing vector store...")
        vectordb = load_vector_store()

    llm = get_llm()

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
