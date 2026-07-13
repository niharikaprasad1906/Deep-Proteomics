# Deep Proteomics: Skip-gram Neural Word Embeddings

This project treats protein sequences as biological sentences and overlapping k-mers (e.g., 3-mers) as words. By training a Skip-gram neural network (Word2Vec) on protein sequence data, it learns a dense vector representation where biologically similar k-mers are positioned near one another.

## Installation
Ensure you are using a Python environment (like Anaconda) with the required dependencies:
```bash
pip install -r requirements.txt
```

## Running the Pipeline
You can run the end-to-end pipeline using `main.py`. By default, it will load the human proteome FASTA file from your system and run training.

### 1. Test Run (Fast Verification)
Train on a subset of 1000 sequences to verify the setup:
```bash
python main.py --limit 1000
```

### 2. Full Run
Train on the entire human proteome dataset (may take some time depending on your CPU/workers):
```bash
python main.py
```

### 3. Custom Configurations
You can specify k-mer size, dimensions, reduction method, and more:
```bash
python main.py --k 3 --dim 128 --window 5 --epochs 15 --method tsne
```

## Output Artifacts
The results of the run will be saved in the `./output` directory:
* `skipgram_model.model`: The trained Word2Vec model, which can be loaded in python via `gensim.models.Word2Vec.load()`.
* `protein_embeddings.png`: Static t-SNE projection colored by Kyte-Doolittle hydrophobicity.
* `protein_embeddings_classes.png`: Static t-SNE projection colored by biochemical group (Acidic, Basic, Hydrophobic, Polar).
* `protein_embeddings.html`: An interactive, web-based viewer using Plotly.js where you can hover over k-mers to inspect their precise biochemical features.
