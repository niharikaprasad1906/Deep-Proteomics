import argparse
import os
import sys
from data_loader import load_all_sequences
from tokenizer import tokenize_corpus
from model import train_skipgram_model, save_model, find_similar_kmers
from visualize import create_embeddings_visualization

DEFAULT_FASTA = r"c:\Users\nihar\AppData\Local\Temp\eb927a48-d9a4-4137-be18-e8a3e4550ca7_uniprotkb_organism_id_9606_2026_07_13.fasta.gz.ca7\uniprotkb_organism_id_9606_2026_07_13.fasta"
DEFAULT_OUTPUT_DIR = "./output"

def main():
    parser = argparse.ArgumentParser(description="Deep Proteomics: Skip-gram Word Embeddings for Proteins")
    parser.add_argument("--fasta", type=str, default=DEFAULT_FASTA, help="Path to FASTA dataset")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of sequences to process (for testing)")
    parser.add_argument("--k", type=int, default=3, help="k-mer size")
    parser.add_argument("--stride", type=int, default=1, help="Stride size (1 for overlapping k-mers)")
    parser.add_argument("--dim", type=int, default=100, help="Embedding dimension")
    parser.add_argument("--window", type=int, default=5, help="Skip-gram window size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--visualize-limit", type=int, default=400, help="Limit of k-mers to plot")
    parser.add_argument("--method", type=str, default="tsne", choices=["tsne", "pca"], help="Dimension reduction method")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers for training")
    
    args = parser.parse_args()
    
    print("="*60)
    print("DEEP PROTEOMICS: EMBEDDING GENERATOR & DECODER")
    print("="*60)
    print(f"Dataset path:  {args.fasta}")
    print(f"Output folder: {args.output}")
    print(f"K-mer size:    {args.k}")
    print(f"Stride:        {args.stride}")
    print(f"Dimension:     {args.dim}")
    print(f"Window size:   {args.window}")
    print(f"Epochs:        {args.epochs}")
    print(f"Workers:       {args.workers}")
    if args.limit:
        print(f"Subsampling:   First {args.limit} sequences only")
    print("="*60)

    # Step 1: Load sequences
    if not os.path.exists(args.fasta):
        print(f"Error: FASTA dataset not found at {args.fasta}.")
        print("Please specify a valid path with the --fasta argument.")
        sys.exit(1)
        
    try:
        entries = load_all_sequences(args.fasta, limit=args.limit)
    except Exception as e:
        print(f"Error loading sequences: {e}")
        sys.exit(1)
        
    sequences = [entry['sequence'] for entry in entries]
    
    # Step 2: Tokenize into k-mers
    print("\nTokenizing protein sequences into k-mers...")
    tokenized_corpus = tokenize_corpus(sequences, k=args.k, stride=args.stride)
    print(f"Tokenization completed. Total 'sentences': {len(tokenized_corpus)}")
    
    # Show stats on k-mer corpus
    total_words = sum(len(sent) for sent in tokenized_corpus)
    print(f"Total k-mer tokens in corpus: {total_words}")
    
    # Step 3: Train skip-gram model
    model = train_skipgram_model(
        tokenized_corpus,
        vector_size=args.dim,
        window=args.window,
        min_count=2, # Require at least 2 occurrences
        workers=args.workers,
        epochs=args.epochs
    )
    
    # Save the model
    model_path = os.path.join(args.output, "skipgram_model.model")
    save_model(model, model_path)
    
    # Export as compressed NumPy embeddings for lightweight dashboard serving
    try:
        from export_model import main as export_embeddings
        export_embeddings()
    except Exception as e:
        print(f"Warning: Could not export compressed NumPy embeddings: {e}")
    
    # Step 4: Run queries/examples
    print("\nTesting Similarity Decoder:")
    vocab = list(model.wv.key_to_index.keys())
    print(f"Unique vocabulary size (k-mers with count >= 2): {len(vocab)}")
    
    # Pick a couple of standard k-mers to test if available
    test_queries = []
    if args.k == 3:
        # standard combinations of interest
        potential_queries = ["LAL", "VAI", "KRH", "EEE", "CYS", "MTA"]
        test_queries = [q for q in potential_queries if q in model.wv]
        
    if not test_queries and len(vocab) > 0:
        import random
        random.seed(42)
        test_queries = random.sample(vocab, min(3, len(vocab)))
        
    for q in test_queries:
        similar = find_similar_kmers(model, q, topn=4)
        print(f"  k-mer '{q}' is most similar to:")
        for w, score in similar:
            print(f"    - {w} (similarity: {score:.4f})")
            
    # Step 5: Visualize
    print("\nGenerating projections and plots...")
    create_embeddings_visualization(
        model, 
        output_dir=args.output, 
        top_n=min(args.visualize_limit, len(vocab)),
        method=args.method
    )
    
    print("\n" + "="*60)
    print("Pipeline Execution Completed Successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
