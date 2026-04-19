import os
import re
import math
from flask import Flask, request, jsonify
from flask_cors import CORS
from pypdf import PdfReader
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from groq import Groq

# --- Environment Setup ---
load_dotenv()
os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'

app = Flask(__name__)
CORS(app)

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, "gato.pdf") if os.path.exists(os.path.join(BASE_DIR, "gato.pdf")) else os.path.join(os.path.dirname(BASE_DIR), "gato.pdf")
HTML_PATH = os.path.join(BASE_DIR, "index.html") if os.path.exists(os.path.join(BASE_DIR, "index.html")) else os.path.join(os.path.dirname(BASE_DIR), "index.html")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3

HF_API_KEY = os.environ.get("HUGGINGFACE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- Clients ---
hf_client = InferenceClient(token=HF_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# --- Retrieval Helpers ---
def get_hf_embeddings(texts):
    if not texts: return []
    try:
        embeddings = hf_client.feature_extraction(texts, model=EMBEDDING_MODEL)
        # Convert to list if it's a numpy array from InferenceClient
        if hasattr(embeddings, 'tolist'):
            return embeddings.tolist()
        return embeddings
    except Exception as e:
        print(f"Error en embeddings HF: {e}")
        return []

def dot_product(v1, v2):
    return sum(x * y for x, y in zip(v1, v2))

def magnitude(v):
    return math.sqrt(sum(x * x for x in v))

def cosine_similarity_pure(v1, v2):
    mag1 = magnitude(v1)
    mag2 = magnitude(v2)
    if mag1 == 0 or mag2 == 0: return 0
    return dot_product(v1, v2) / (mag1 * mag2)

# --- Text Extraction ---
def get_pdf_text(path):
    if not os.path.exists(path): return ""
    try:
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception: return ""

def get_html_text(path):
    if not os.path.exists(path): return ""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')
            main = soup.find('main')
            return main.get_text(separator=' ') if main else ""
    except Exception: return ""

def chunk_text(text, size, overlap):
    text = re.sub(r'\s+', ' ', text).strip()
    chunks = []
    for i in range(0, len(text), size - overlap):
        chunk = text[i:i + size]
        if len(chunk) > 50: chunks.append(chunk)
    return chunks

# --- Initialize Knowledge Base ---
print("Consolidando el conocimiento del Oráculo (Hugging Face + Groq)...")
all_chunks = []

pdf_text = get_pdf_text(PDF_PATH)
if pdf_text: all_chunks.extend(chunk_text(pdf_text, CHUNK_SIZE, CHUNK_OVERLAP))

html_text = get_html_text(HTML_PATH)
if html_text: all_chunks.extend(chunk_text(html_text, CHUNK_SIZE, CHUNK_OVERLAP))

CHUNK_VECTORS = []
if all_chunks:
    print(f"Generando embeddings para {len(all_chunks)} fragmentos...")
    embeddings = get_hf_embeddings(all_chunks)
    if embeddings:
        CHUNK_VECTORS = embeddings
        print("Vectores listos.")
    else:
        print("ERROR: Fallo al generar vectores.")

def retrieve_context(query, top_k=TOP_K):
    if not all_chunks or not CHUNK_VECTORS: return []
    query_vecs = get_hf_embeddings([query])
    if not query_vecs: return []
    query_vec = query_vecs[0]
    
    similarities = [cosine_similarity_pure(query_vec, doc_vec) for doc_vec in CHUNK_VECTORS]
    results = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)[:top_k]
    return [all_chunks[i] for i, sim in results if sim > 0.1]

# --- Generation ---
def ask_groq(query, context_chunks):
    if not groq_client: return "No se ha configurado Groq para las respuestas."
    context = "\n\n---\n\n".join(context_chunks)
    try:
        system_prompt = "Eres el Oráculo de Plutón, el espíritu del gato negro. Respondes con sabiduría oscura y literaria, basándote en el cuento de Poe."
        user_prompt = f"CONTEXTO REGISTRADO:\n{context}\n\nPREGUNTA DEL MORTAL: {query}"
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.6,
            max_tokens=800,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"El Oráculo se desvanece por un error: {str(e)}"

# --- Flask endpoint ---
@app.route('/api/chat', methods=['POST'])
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json or {}
    query = data.get('query', '').strip()
    if not query: return jsonify({"answer": "Habla, mortal..."})
    if not all_chunks: return jsonify({"answer": "Mi conocimiento está perdido."})
    context_chunks = retrieve_context(query)
    if not context_chunks:
        answer = "Las sombras no revelan nada sobre eso. Pregunta sobre el gato o el destino oscuro."
    else:
        answer = ask_groq(query, context_chunks)
    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(port=5000, debug=False)
