# Deep Proteomics: Skip-gram Neural Word Embeddings for Protein Analysis

🧬 **Treating Protein Sequences as Biological Sentences and k-mers as Words**

Deep Proteomics is a machine learning pipeline that applies Natural Language Processing (NLP) techniques to proteomics. By treating protein sequences as biological sentences and overlapping k-mers (e.g., 3-mers) as words, this project trains a Skip-gram neural network (Word2Vec) on protein sequence data. It learns a dense vector representation where biologically similar k-mers are positioned near one another in a low-dimensional embedding space.

The project features a complete command-line training pipeline, static visualizations, and a **premium, interactive web-based dashboard** for real-time embedding exploration, neighborhood similarity decoding, and protein sequence profiling.

---

## 🚀 Key Features

* **Biologically-Aware Tokenization**: Tokenize protein FASTA sequences into overlapping k-mers with customizable stride configurations.
* **Skip-Gram Embeddings**: Train a Skip-gram neural network to map k-mers to high-dimensional vectors, capturing localized context.
* **Dimensionality Reduction**: Project embeddings into 2D or 3D spaces using t-SNE or PCA.
* **Biophysical Overlays**: Automatically compute average Kyte-Doolittle hydrophobicity and categorize k-mers into charge groups (Hydrophobic, Polar, Acidic, Basic).
* **Interactive Web Explorer**: A dashboard styled with dark-mode glassmorphism containing:
  * **Embedding Map**: 2D/3D Plotly visualizer allowing you to filter by biochemical groups or thermal hydrophobicity scales.
  * **K-mer Decoder**: Search any k-mer to dynamically identify nearest neighbors in the embedding space with graphical connector lines.
  * **Sequence Profiler**: Paste a protein sequence to map its residues to embedding coherence metrics and plot localized properties.
* **Cloud-Ready Containerization**: Dockerfile configurations for instant local or cloud deployment.

---

## 📁 Repository Structure

```
├── data_loader.py       # FASTA file parsing utilities
├── tokenizer.py          # Overlapping k-mer tokenization logic
├── model.py              # Skip-gram (Word2Vec) training & similarity helpers
├── visualize.py          # Static t-SNE/PCA & matplotlib plotting pipelines
├── main.py               # End-to-end CLI training pipeline execution
├── server.py             # Flask Web Server & coordinate caching APIs
├── templates/
│   └── index.html        # Glassmorphic single-page web dashboard
├── static/
│   ├── app.js            # Plotly scatter-plot and AJAX client logic
│   └── style.css         # Premium CSS design tokens & animations
├── Dockerfile            # Container definition for WSGI Gunicorn deployment
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## ⚙️ Installation

### Prerequisites
* Python 3.10+
* Git

### Local Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/niharikaprasad1906/Deep-Proteomics.git
   cd Deep-Proteomics
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🏃 Running the Pipeline

You can run the end-to-end pipeline using `main.py`. By default, it will load the human proteome FASTA file and run training.

### 1. Test Run (Fast Verification)
Train on a subset of 1000 sequences to verify setup:
```bash
python main.py --limit 1000
```

### 2. Full Run
Train on the entire human proteome dataset (may take some time depending on hardware):
```bash
python main.py
```

### 3. Custom CLI Configurations
Modify parameters such as k-mer size, dimensionality, context window size, and reduction methods:
```bash
python main.py --k 3 --dim 128 --window 5 --epochs 15 --method tsne
```

**Training CLI Arguments:**
* `--fasta`: Path to input FASTA file.
* `--output`: Output folder path (default: `./output`).
* `--limit`: Limit number of entries to process.
* `--k`: Size of k-mer word segments (default: 3).
* `--stride`: Stride size for extraction (default: 1).
* `--dim`: Word2Vec vector dimensions (default: 100).
* `--window`: Context window size (default: 5).
* `--epochs`: Epoch iterations (default: 10).
* `--method`: Projection algorithm (`tsne` or `pca`).

---

## 🖥️ Running the Web Dashboard

The web dashboard is the ultimate tool for visual analysis. It caches 2D and 3D projections on startup, enabling lightning-fast client-side transitions.

1. Start the Flask server:
   ```bash
   python server.py
   ```
2. Open your browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

### Features Inside the Dashboard
* **Toggle Views**: Move between a 2D Scatter plot and a **3D Spatial Topology** visualization.
* **Interactive Neighbors**: Click a k-mer on the chart to inspect its 10 nearest context neighbors. The plot will draw dashed connections in the vector space.
* **Sequence Alignment Profiler**: Load a custom protein sequence to plot its localized hydrophobicity profile and coherence index (next-neighbor cosine similarity).

---

## 🐳 Docker Deployment

To run the application inside a container (serving the web dashboard via a production Gunicorn WSGI server):

1. Build the image:
   ```bash
   docker build -t deep-proteomics-dashboard .
   ```

2. Run the container:
   ```bash
   docker run -d -p 5000:5000 deep-proteomics-dashboard
   ```

---

## 📊 Output Artifacts

Running `main.py` saves output assets in the `./output` folder:
* `skipgram_model.model`: Trained Word2Vec model, loadable via `gensim.models.Word2Vec.load()`.
* `protein_embeddings.png`: Static t-SNE plot colored by Kyte-Doolittle hydrophobicity.
* `protein_embeddings_classes.png`: Static t-SNE plot colored by biochemical group.
* `protein_embeddings.html`: Interactive, web-based Plotly.js explorer page.

---

## 📚 References
* **Word2Vec (Skip-gram)**: Mikolov et al. *Efficient Estimation of Word Representations in Vector Space*, 2013.
* **Kyte-Doolittle Hydrophobicity**: Kyte J., Doolittle R.F. *A simple method for displaying the hydropathic character of a protein*, J. Mol. Biol. 1982.
