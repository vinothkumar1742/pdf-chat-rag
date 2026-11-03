import os
import time
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from google import genai
from google.genai import types

load_dotenv()

# ---------- Setup ----------
embedder = SentenceTransformer("all-MiniLM-L6-v2")   # free, runs locally
db = chromadb.PersistentClient(path="chroma_db")
collection = db.get_or_create_collection("pdf_chunks")

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(timeout=30000),   # 30 seconds
)

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.1-flash-lite")
print("Using model:", MODEL, "| fallback:", FALLBACK_MODEL)


# ---------- PDF processing ----------
def split_text(text, size=1500, overlap=200):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks

def clear_all():
    global collection
    db.delete_collection("pdf_chunks")
    collection = db.get_or_create_collection("pdf_chunks")

def add_pdf(path, doc_name):
    clear_all()          # single-PDF mode for now
    reader = PdfReader(path)
    ids, docs, metas = [], [], []

    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for j, chunk in enumerate(split_text(text)):
            if chunk.strip():
                ids.append(f"{doc_name}-{page_no}-{j}")
                docs.append(chunk)
                metas.append({"doc": doc_name, "page": page_no})

    if not docs:
        return 0   # scanned PDF or no text found

    embeddings = embedder.encode(docs, batch_size=32).tolist()
    collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
    return len(docs)


# ---------- LLM call with retry and fallback ----------
def call_llm(prompt):
    for model in [MODEL, FALLBACK_MODEL]:
        for attempt in range(3):
            try:
                response = client.models.generate_content(model=model, contents=prompt)
                return response.text
            except Exception as e:
                print(f"{model} attempt {attempt + 1} failed: {str(e)[:150]}")
                err = str(e).lower()
                if "503" in err or "429" in err or "timeout" in err or "timed out" in err:
                    time.sleep(2 * (attempt + 1))   # wait 2s, 4s, 6s
                    continue
                break   # other errors: skip to the fallback model
    return "Gemini is busy right now. Please try again in a minute."


# ---------- Ask a question ----------
def ask(question, k=6):
    if collection.count() == 0:
        return "Please upload a PDF first.", []

    t = time.time()
    q_emb = embedder.encode([question]).tolist()
    print("embed:", round(time.time() - t, 2), "s")

    t = time.time()
    res = collection.query(query_embeddings=q_emb, n_results=k)
    print("search:", round(time.time() - t, 2), "s")

    chunks, metas = res["documents"][0], res["metadatas"][0]
    context = "\n\n".join(f"[Page {m['page']}] {c}" for c, m in zip(chunks, metas))
    print("\n--- RETRIEVED ---")
    for c, m in zip(chunks, metas):
        print(f"[Page {m['page']}]", c[:200].replace("\n", " "))
    print("-----------------\n")

    prompt = f"""You are answering questions about a document.
Use the context below to answer. The context may be only part of the document,
so read it carefully and use every relevant detail.
Only if there is truly nothing relevant, say "I could not find this in the document."

Context:
{context}

Question: {question}
Answer:"""

    t = time.time()
    answer = call_llm(prompt)
    print("gemini:", round(time.time() - t, 2), "s")

    pages = sorted({m["page"] for m in metas})
    return answer, pages