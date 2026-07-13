import os
import sys
import json
import numpy as np
from flask import Flask, jsonify, request, render_template, send_from_directory
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Add current directory to path just in case
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import load_model
from visualize import get_kmer_hydrophobicity, get_kmer_charge_class

app = Flask(__name__, template_folder='templates', static_folder='static')

# Global variables to hold model and cached projections
MODEL = None
MODEL_INFO = {}
PROJECTION_CACHE = {}
VOCAB = []
TOP_N_KMERS = 600

def get_fallback_model():
    """
    Generates a mock/dummy Word2Vec model if no trained model is found.
    This allows the dashboard to be runnable immediately even without full training.
    """
    from gensim.models import Word2Vec
    print("No pre-trained model found. Generating dummy corpus and model for demo purposes...")
    # Generate random amino acid sequences
    import random
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    dummy_corpus = []
    random.seed(42)
    for _ in range(200):
        seq = "".join(random.choices(amino_acids, k=random.randint(15, 30)))
        # Split into overlapping 3-mers
        kmers = [seq[i:i+3] for i in range(len(seq) - 2)]
        dummy_corpus.append(kmers)
    
    model = Word2Vec(sentences=dummy_corpus, vector_size=64, window=4, min_count=1, epochs=5, sg=1)
    return model

def init_app():
    global MODEL, MODEL_INFO, PROJECTION_CACHE, VOCAB
    
    model_path = os.path.join("output", "skipgram_model.model")
    if os.path.exists(model_path):
        try:
            MODEL = load_model(model_path)
            print("Successfully loaded pre-trained model.")
        except Exception as e:
            print(f"Error loading model: {e}")
            MODEL = get_fallback_model()
    else:
        MODEL = get_fallback_model()
        
    # Extract vocabulary
    VOCAB = list(MODEL.wv.key_to_index.keys())
    
    # Infer k-mer size
    sample_kmer = VOCAB[0] if VOCAB else "AAA"
    k_size = len(sample_kmer)
    
    # Setup model info metadata
    MODEL_INFO = {
        "status": "loaded",
        "vocab_size": len(VOCAB),
        "vector_size": MODEL.vector_size,
        "window": MODEL.window,
        "epochs": getattr(MODEL, 'epochs', 'N/A'),
        "k": k_size
    }
    
    # Precompute projections for the top N frequent k-mers
    precompute_projections()

def precompute_projections():
    global MODEL, VOCAB, PROJECTION_CACHE
    
    limit = min(TOP_N_KMERS, len(VOCAB))
    top_vocab = VOCAB[:limit]
    
    if limit == 0:
        return
        
    vectors = np.array([MODEL.wv[word] for word in top_vocab])
    
    print(f"Precomputing 2D/3D projections for the top {limit} k-mers...")
    
    # 2D PCA
    pca_2d = PCA(n_components=2, random_state=42)
    coords_pca_2d = pca_2d.fit_transform(vectors)
    
    # 3D PCA
    pca_3d = PCA(n_components=3, random_state=42)
    coords_pca_3d = pca_3d.fit_transform(vectors)
    
    # 2D t-SNE
    perplexity = min(30, max(5, limit - 1))
    tsne_2d = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    coords_tsne_2d = tsne_2d.fit_transform(vectors)
    
    # 3D t-SNE
    tsne_3d = TSNE(n_components=3, perplexity=perplexity, random_state=42, max_iter=1000)
    coords_tsne_3d = tsne_3d.fit_transform(vectors)
    
    # Cache all projection outputs
    PROJECTION_CACHE = {
        "pca_2d": coords_pca_2d,
        "pca_3d": coords_pca_3d,
        "tsne_2d": coords_tsne_2d,
        "tsne_3d": coords_tsne_3d,
        "vocab": top_vocab
    }
    print("Projection caching completed.")

# HTML Routes
@app.route('/')
def index():
    return render_template('index.html')

# API Routes
@app.route('/api/info')
def api_info():
    return jsonify(MODEL_INFO)

@app.route('/api/embeddings')
def api_embeddings():
    method = request.args.get('method', 'tsne').lower()
    dims = int(request.args.get('dims', 2))
    
    cache_key = f"{method}_{dims}d"
    if cache_key not in PROJECTION_CACHE:
        return jsonify({"error": f"Projection mapping '{cache_key}' is not available."}), 400
        
    coords = PROJECTION_CACHE[cache_key]
    vocab = PROJECTION_CACHE["vocab"]
    
    result = []
    for i, word in enumerate(vocab):
        hydro = get_kmer_hydrophobicity(word)
        charge_grp = get_kmer_charge_class(word)
        item = {
            "text": word,
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1]),
            "hydrophobicity": hydro,
            "group": charge_grp
        }
        if dims == 3:
            item["z"] = float(coords[i, 2])
        result.append(item)
        
    return jsonify(result)

@app.route('/api/similarity')
def api_similarity():
    query = request.args.get('query', '').upper().strip()
    top_n = int(request.args.get('top_n', 10))
    
    if not query:
        return jsonify({"error": "Query string is empty"}), 400
        
    if query not in MODEL.wv:
        return jsonify({"error": f"K-mer '{query}' is not in the model vocabulary."}), 404
        
    similar_items = MODEL.wv.most_similar(query, topn=top_n)
    
    results = []
    # Include query itself at similarity = 1.0
    query_hydro = get_kmer_hydrophobicity(query)
    query_grp = get_kmer_charge_class(query)
    results.append({
        "text": query,
        "similarity": 1.0,
        "hydrophobicity": query_hydro,
        "group": query_grp,
        "is_query": True
    })
    
    for word, score in similar_items:
        hydro = get_kmer_hydrophobicity(word)
        charge_grp = get_kmer_charge_class(word)
        results.append({
            "text": word,
            "similarity": float(score),
            "hydrophobicity": hydro,
            "group": charge_grp,
            "is_query": False
        })
        
    return jsonify(results)

@app.route('/api/sequence_profile')
def api_sequence_profile():
    seq = request.args.get('seq', '').upper().strip()
    if not seq:
        return jsonify({"error": "Sequence parameter is empty."}), 400
        
    k = MODEL_INFO.get('k', 3)
    if len(seq) < k:
        return jsonify({"error": f"Sequence is too short (must be at least {k} residues)."}), 400
        
    # Split into overlapping k-mers
    kmers = [seq[i:i+k] for i in range(len(seq) - k + 1)]
    
    profile = []
    for idx, kmer in enumerate(kmers):
        in_vocab = kmer in MODEL.wv
        hydro = get_kmer_hydrophobicity(kmer)
        charge_grp = get_kmer_charge_class(kmer)
        
        # Calculate vector magnitude as a proxy for frequency/confidence
        vector_norm = float(np.linalg.norm(MODEL.wv[kmer])) if in_vocab else 0.0
        
        # Calculate similarity with next k-mer if available
        next_sim = 0.0
        if in_vocab and idx < len(kmers) - 1 and kmers[idx+1] in MODEL.wv:
            next_sim = float(MODEL.wv.similarity(kmer, kmers[idx+1]))
            
        profile.append({
            "index": idx,
            "kmer": kmer,
            "in_vocab": in_vocab,
            "hydrophobicity": hydro,
            "group": charge_grp,
            "vector_magnitude": vector_norm,
            "next_similarity": next_sim
        })
        
    return jsonify(profile)

if __name__ == '__main__':
    # Initialize model loader and projections
    init_app()
    # Run server
    app.run(host='0.0.0.0', port=5000, debug=True)
else:
    init_app()
