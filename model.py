import os
from gensim.models import Word2Vec

def train_skipgram_model(tokenized_corpus, vector_size=100, window=5, min_count=1, workers=4, epochs=10):
    """
    Trains a Gensim Word2Vec model using the Skip-gram architecture.
    
    Parameters:
    - tokenized_corpus (list of list of str): Corpus tokenized into k-mers.
    - vector_size (int): Dimension of the embedding vectors.
    - window (int): Context window size.
    - min_count (int): Ignores all words with total frequency lower than this.
    - workers (int): Number of worker threads to use for training.
    - epochs (int): Number of iterations (epochs) over the corpus.
    
    Returns:
    - Word2Vec: The trained Gensim Word2Vec model.
    """
    print("Initializing Word2Vec Skip-gram model...")
    # sg=1 specifies Skip-gram architecture (sg=0 would be CBOW)
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
    Saves the Word2Vec model to a file.
    
    Parameters:
    - model (Word2Vec): The trained model.
    - filepath (str): Output file path.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    model.save(filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath):
    """
    Loads a Word2Vec model from a file.
    
    Parameters:
    - filepath (str): Path to the model file.
    
    Returns:
    - Word2Vec: The loaded model.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found at: {filepath}")
    model = Word2Vec.load(filepath)
    print(f"Model loaded from {filepath}")
    return model

def find_similar_kmers(model, kmer, topn=5):
    """
    Finds the most similar k-mers in the embedding space.
    
    Parameters:
    - model (Word2Vec): The trained model.
    - kmer (str): The query k-mer.
    - topn (int): Number of similar k-mers to return.
    
    Returns:
    - list of tuples: Similar k-mers and their cosine similarities.
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
