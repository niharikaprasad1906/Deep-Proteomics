import os
import numpy as np

class NumpyWord2VecKeyVectors:
    """
    Pure Python and NumPy implementation mimicking Gensim's KeyedVectors interface.
    This enables cosine similarity and vector lookups without compiled packages.
    """
    def __init__(self, vocab, vectors):
        self.vocab = list(vocab)
        self.vectors = vectors
        self.key_to_index = {word: i for i, word in enumerate(self.vocab)}
        
        # Precompute normalized vectors for quick cosine similarity
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self.norm_vectors = vectors / np.where(norms == 0, 1.0, norms)
        
    def __contains__(self, key):
        return key in self.key_to_index
        
    def __getitem__(self, key):
        if key not in self.key_to_index:
            raise KeyError(f"Word '{key}' not in vocabulary")
        return self.vectors[self.key_to_index[key]]
        
    def similarity(self, w1, w2):
        if w1 not in self.key_to_index or w2 not in self.key_to_index:
            return 0.0
        i1 = self.key_to_index[w1]
        i2 = self.key_to_index[w2]
        return float(np.dot(self.norm_vectors[i1], self.norm_vectors[i2]))
        
    def most_similar(self, word, topn=10):
        if word not in self.key_to_index:
            raise KeyError(f"Word '{word}' not in vocabulary")
        idx = self.key_to_index[word]
        query_vec = self.norm_vectors[idx]
        sims = np.dot(self.norm_vectors, query_vec)
        
        # Sort indices descending
        top_indices = np.argsort(sims)[::-1]
        
        results = []
        for i in top_indices:
            w = self.vocab[i]
            if w == word:
                continue
            results.append((w, float(sims[i])))
            if len(results) >= topn:
                break
        return results
        
    @property
    def index_to_key(self):
        return self.vocab


class NumpyWord2Vec:
    """
    Pure Python and NumPy wrapper mimicking Gensim's Word2Vec model.
    """
    def __init__(self, npz_data):
        vocab = npz_data["vocab"]
        vectors = npz_data["vectors"]
        self.vector_size = int(npz_data["vector_size"])
        self.window = int(npz_data["window"])
        self.epochs = int(npz_data["epochs"])
        self.wv = NumpyWord2VecKeyVectors(vocab, vectors)


def train_skipgram_model(tokenized_corpus, vector_size=100, window=5, min_count=1, workers=4, epochs=10):
    """
    Trains a Gensim Word2Vec model using the Skip-gram architecture.
    Imports Word2Vec inline to avoid crash if gensim is not installed.
    """
    from gensim.models import Word2Vec
    print("Initializing Word2Vec Skip-gram model...")
    model = Word2Vec(
        sentences=tokenized_corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=1,
        epochs=epochs
    )
    print("Training completed.")
    return model


def save_model(model, filepath):
    """
    Saves the Word2Vec model to a file using gensim.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    model.save(filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath):
    """
    Loads a Word2Vec model. Prioritizes loading a NumPy compressed file (.npz)
    to avoid importing gensim, falling back to gensim load if necessary.
    """
    # Check if a NumPy npz version exists in same directory
    npz_path = os.path.join(os.path.dirname(filepath), "kmer_embeddings.npz")
    
    if filepath.endswith('.npz') or os.path.exists(npz_path):
        target_path = npz_path if os.path.exists(npz_path) else filepath
        print(f"Loading pure NumPy embeddings model from {target_path}...")
        data = np.load(target_path)
        return NumpyWord2Vec(data)
        
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found at: {filepath}")
        
    from gensim.models import Word2Vec
    model = Word2Vec.load(filepath)
    print(f"Model loaded via Gensim from {filepath}")
    return model


def find_similar_kmers(model, kmer, topn=5):
    """
    Finds the most similar k-mers in the embedding space.
    """
    try:
        similar = model.wv.most_similar(kmer, topn=topn)
        return similar
    except KeyError:
        print(f"K-mer '{kmer}' not found in the vocabulary.")
        return []


if __name__ == "__main__":
    # Test model training on dummy corpus
    test_corpus = [
        ["MTA", "TAI", "AII", "IIK", "IKE", "KEI", "EIV", "IVS"],
        ["MTA", "TAI", "AII", "IIK", "IKE", "KEI"],
        ["IKE", "KEI", "EIV", "IVS"]
    ]
    model = train_skipgram_model(test_corpus, vector_size=8, window=2, min_count=1, epochs=5)
    print("\nVocabulary:", list(model.wv.key_to_index.keys()))
    print("Embedding vector for 'MTA':", model.wv['MTA'])
    print("Similar to 'IKE':", find_similar_kmers(model, "IKE", topn=2))
