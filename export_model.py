import os
import numpy as np
from gensim.models import Word2Vec

def main():
    model_path = os.path.join("output", "skipgram_model.model")
    if os.path.exists(model_path):
        print(f"Loading Word2Vec model from {model_path}...")
        model = Word2Vec.load(model_path)
        
        vocab = model.wv.index_to_key
        vectors = model.wv.vectors
        vector_size = int(model.vector_size)
        window = int(model.window)
        epochs = int(getattr(model, 'epochs', 10))
        k = int(len(vocab[0])) if vocab else 3
        
        output_path = os.path.join("output", "kmer_embeddings.npz")
        np.savez_compressed(
            output_path, 
            vocab=vocab, 
            vectors=vectors, 
            vector_size=vector_size,
            window=window,
            epochs=epochs,
            k=k
        )
        print(f"Successfully exported compressed embeddings to {output_path} (Size: {os.path.getsize(output_path)/1024:.1f} KB)")
    else:
        print(f"Error: Model not found at {model_path}")

if __name__ == "__main__":
    main()
