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
ROOT_DIR = os.path.dirname(BASE_DIR)

# Robust path finding for Vercel vs Local
def find_data_file(filename):
    paths_to_try = [
        os.path.join(BASE_DIR, filename),           # /api/filename
        os.path.join(ROOT_DIR, filename),           # /filename (Root)
        os.path.join(os.getcwd(), filename),        # CWD/filename
        os.path.join('/var/task', filename),        # Vercel task root
    ]
    for p in paths_to_try:
        if os.path.exists(p):
            print(f"DEBUG: Encontrado {filename} en {p}")
            return p
    return None

PDF_PATH = find_data_file("gato.pdf")
HTML_PATH = find_data_file("index.html")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3

HF_API_KEY = os.environ.get("HUGGINGFACE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LAST_ERROR = "Ninguno"

# --- Clients ---
hf_client = InferenceClient(token=HF_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

import requests
import json

# --- Retrieval Helpers ---
def get_hf_embeddings(texts):
    global LAST_ERROR
    if not HF_API_KEY: 
        LAST_ERROR = "Falta HF_API_KEY"
        return []
    if not texts: return []
    
    # Nueva URL estándar de Hugging Face (Router 2024)
    api_url = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2"
    headers = {"Authorization": f"Bearer {HF_API_KEY}", "x-wait-for-model": "true"}
    
    batch_size = 10
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            payload = {"inputs": batch, "options": {"wait_for_model": True}}
            response = requests.post(api_url, headers=headers, json=payload, timeout=25)
            
            if response.status_code == 200:
                batch_embeddings = response.json()
                if isinstance(batch_embeddings, list):
                    all_embeddings.extend(batch_embeddings)
                else:
                    LAST_ERROR = f"Error: Formato JSON inesperado"
            else:
                LAST_ERROR = f"API Error {response.status_code}: {response.text[:50]}"
                break
        except Exception as e:
            LAST_ERROR = f"Excepción: {str(e)}"
            break

    return all_embeddings

def dot_product(v1, v2): return sum(x * y for x, y in zip(v1, v2))
def magnitude(v): return math.sqrt(sum(x * x for x in v))
def cosine_similarity_pure(v1, v2):
    mag1, mag2 = magnitude(v1), magnitude(v2)
    return dot_product(v1, v2) / (mag1 * mag2) if mag1 > 0 and mag2 > 0 else 0

# --- Text Extraction ---
def get_pdf_text(path):
    if not path or not os.path.exists(path): return ""
    try:
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e: return ""

def get_html_text(path):
    if not path or not os.path.exists(path): return ""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')
            main = soup.find('main')
            return main.get_text(separator=' ') if main else ""
    except Exception as e: return ""

def chunk_text(text, size, overlap):
    text = re.sub(r'\s+', ' ', text).strip()
    chunks = []
    for i in range(0, len(text), size - overlap):
        chunk = text[i:i + size]
        if len(chunk) > 50: chunks.append(chunk)
    return chunks

# --- Initialize Knowledge Base ---
print("Consolidando el conocimiento del Oráculo...")
all_chunks = []
if PDF_PATH: all_chunks.extend(chunk_text(get_pdf_text(PDF_PATH), CHUNK_SIZE, CHUNK_OVERLAP))
if HTML_PATH: all_chunks.extend(chunk_text(get_html_text(HTML_PATH), CHUNK_SIZE, CHUNK_OVERLAP))

CHUNK_VECTORS = []
if all_chunks and HF_API_KEY:
    CHUNK_VECTORS = get_hf_embeddings(all_chunks)

def retrieve_context(query, top_k=TOP_K):
    # Intentar búsqueda vectorial (Avanzada)
    if all_chunks and CHUNK_VECTORS:
        query_vecs = get_hf_embeddings([query])
        if query_vecs:
            query_vec = query_vecs[0]
            sims = [cosine_similarity_pure(query_vec, doc_vec) for doc_vec in CHUNK_VECTORS]
            results = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[:top_k]
            return [all_chunks[i] for i, sim in results if sim > 0.1]
    
    # Fallback: Búsqueda por Palabras Clave (Básica)
    print("DEBUG: Usando búsqueda de palabras clave de reserva.")
    keywords = query.lower().split()
    scored_chunks = []
    for chunk in all_chunks:
        score = sum(1 for word in keywords if word in chunk.lower())
        if score > 0: scored_chunks.append((chunk, score))
    
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    return [chunk for chunk, score in scored_chunks[:top_k]]

# --- Generation ---
def ask_groq(query, context_chunks):
    if not GROQ_API_KEY: return "Error: Falta GROQ_API_KEY en Vercel."
    context = "\n\n---\n\n".join(context_chunks)
    try:
        system_prompt = "Eres el Oráculo de Plutón. Espíritu espectral del gato negro. Responde en español con tono literario gótico basado en el contexto."
        user_prompt = f"CONTEXTO POE:\n{context}\n\nPREGUNTA:\n{query}"
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.6,
            max_tokens=800,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Oscuridad en la respuesta: {str(e)}"

# --- Flask endpoint ---
@app.route('/api/chat', methods=['POST'])
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json or {}
    query = data.get('query', '').strip()
    
    if not query: return jsonify({"answer": "Habla, mortal..."})
    
    # Diagnostic Info
    diag = f"(PDF:{bool(PDF_PATH)}, Chunks:{len(all_chunks)}, Vectors:{len(CHUNK_VECTORS)}, API_HF:{bool(HF_API_KEY)}, Error:{LAST_ERROR})"

    if not all_chunks or not CHUNK_VECTORS:
        return jsonify({"answer": f"El Oráculo está mudo. Las sombras bloquean su visión. {diag}"})

    context_chunks = retrieve_context(query)
    if not context_chunks:
        answer = f"Las sombras no revelan nada sobre eso en mis crónicas. {diag}"
    else:
        answer = ask_groq(query, context_chunks)

    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(port=5000, debug=False)
