import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

# Kyte-Doolittle Hydrophobicity Scale
KYTE_DOOLITTLE = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

def get_kmer_hydrophobicity(kmer):
    """
    Computes the average hydrophobicity of a k-mer based on the Kyte-Doolittle scale.
    """
    scores = [KYTE_DOOLITTLE.get(aa, 0.0) for aa in kmer.upper()]
    return np.mean(scores) if scores else 0.0

def get_kmer_charge_class(kmer):
    """
    Categorizes the k-mer by its net charge/polarity profile.
    """
    kmer = kmer.upper()
    pos = sum(1 for aa in kmer if aa in 'KRH')
    neg = sum(1 for aa in kmer if aa in 'DE')
    
    if pos > neg:
        return 'Basic (Positive)'
    elif neg > pos:
        return 'Acidic (Negative)'
    
    # Check if hydrophobic or polar
    hydrophobic_count = sum(1 for aa in kmer if aa in 'AVILPMFWY')
    if hydrophobic_count >= len(kmer) / 2.0:
        return 'Hydrophobic'
    else:
        return 'Polar/Neutral'

def create_embeddings_visualization(model, output_dir, top_n=500, method='tsne'):
    """
    Projects top N k-mer embeddings to 2D and generates static and interactive plots.
    
    Parameters:
    - model (Word2Vec): Trained Word2Vec model.
    - output_dir (str): Directory where plots will be saved.
    - top_n (int): Number of most frequent k-mers to plot.
    - method (str): Dimensionality reduction method ('tsne' or 'pca').
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get most frequent k-mers
    vocab = list(model.wv.key_to_index.keys())[:top_n]
    if len(vocab) == 0:
        print("Vocabulary is empty. Cannot visualize.")
        return
        
    vectors = np.array([model.wv[word] for word in vocab])
    
    # Dimensionality reduction
    print(f"Running {method.upper()} on {len(vocab)} k-mer vectors...")
    if method.lower() == 'tsne':
        reducer = TSNE(n_components=2, perplexity=min(30, len(vocab)-1), random_state=42, max_iter=1000)
    else:
        reducer = PCA(n_components=2, random_state=42)
        
    coords = reducer.fit_transform(vectors)
    
    # Prepare properties for plotting
    hydrophobicities = [get_kmer_hydrophobicity(word) for word in vocab]
    charge_classes = [get_kmer_charge_class(word) for word in vocab]
    
    # 1. Generate Static Matplotlib Plot (Hydrophobicity scale)
    plt.figure(figsize=(12, 10))
    sc = plt.scatter(coords[:, 0], coords[:, 1], c=hydrophobicities, cmap='coolwarm', alpha=0.7, edgecolors='none', s=35)
    plt.colorbar(sc, label='Kyte-Doolittle Hydrophobicity (Average)')
    plt.title(f'Protein k-mer Embeddings ({method.upper()} Projection)\nColored by Hydrophobicity', fontsize=14)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Label a few random points to avoid overcrowding
    np.random.seed(42)
    label_indices = np.random.choice(len(vocab), min(40, len(vocab)), replace=False)
    for idx in label_indices:
        plt.annotate(vocab[idx], (coords[idx, 0], coords[idx, 1]), textcoords="offset points", 
                     xytext=(2,2), ha='center', fontsize=9, fontweight='bold', alpha=0.8)
                     
    static_plot_path = os.path.join(output_dir, 'protein_embeddings.png')
    plt.savefig(static_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Static plot saved to {static_plot_path}")
    
    # 2. Generate Static Matplotlib Plot (Discrete Charge/Polarity groups)
    plt.figure(figsize=(12, 10))
    classes = list(set(charge_classes))
    colors = ['#3182bd', '#e6550d', '#2ca02c', '#756bb1'] # custom palette
    color_map = {cls: colors[i % len(colors)] for i, cls in enumerate(classes)}
    
    for cls in classes:
        idx = [i for i, c in enumerate(charge_classes) if c == cls]
        plt.scatter(coords[idx, 0], coords[idx, 1], label=cls, color=color_map[cls], alpha=0.7, s=35)
        
    plt.title(f'Protein k-mer Embeddings ({method.upper()} Projection)\nColored by Biochemical Property Group', fontsize=14)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend(title='Property Group')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    for idx in label_indices:
        plt.annotate(vocab[idx], (coords[idx, 0], coords[idx, 1]), textcoords="offset points", 
                     xytext=(2,2), ha='center', fontsize=9, fontweight='bold', alpha=0.8)
                     
    class_plot_path = os.path.join(output_dir, 'protein_embeddings_classes.png')
    plt.savefig(class_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Property group plot saved to {class_plot_path}")
    
    # 3. Generate Interactive HTML Plot (Plotly representation via raw HTML/JS template to avoid large package installation overhead)
    # Using a simple HTML document with Chart.js or Plotly.js CDN
    html_path = os.path.join(output_dir, 'protein_embeddings.html')
    
    # Build JavaScript arrays
    js_data = []
    for i, word in enumerate(vocab):
        js_data.append({
            'x': float(coords[i, 0]),
            'y': float(coords[i, 1]),
            'text': word,
            'hydrophobicity': float(hydrophobicities[i]),
            'group': charge_classes[i]
        })
    
    import json
    data_json = json.dumps(js_data)
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Deep Proteomics: Protein Embeddings Space</title>
    <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
    <style>
        body {{
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            background-color: #121214;
            color: #e4e4e7;
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{
            color: #ffffff;
            margin-bottom: 5px;
            font-weight: 500;
        }}
        p.subtitle {{
            color: #a1a1aa;
            margin-top: 0;
            margin-bottom: 20px;
        }}
        #plot-container {{
            background-color: #18181b;
            border-radius: 12px;
            padding: 15px;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
            border: 1px solid #27272a;
        }}
        .footer {{
            margin-top: 20px;
            text-align: center;
            font-size: 12px;
            color: #71717a;
        }}
        .legend-info {{
            margin-top: 15px;
            display: flex;
            justify-content: space-around;
            background-color: #18181b;
            padding: 10px;
            border-radius: 8px;
            border: 1px solid #27272a;
            font-size: 14px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
        }}
        .legend-color {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Protein Embedding Space Decoder</h1>
        <p class="subtitle">Skip-gram (Word2Vec) representation of {len(vocab)} k-mer words projected to 2D via {method.upper()}</p>
        
        <div id="plot-container">
            <div id="plot"></div>
        </div>

        <div class="legend-info">
            <div class="legend-item"><div class="legend-color" style="background-color: #2ca02c;"></div>Hydrophobic</div>
            <div class="legend-item"><div class="legend-color" style="background-color: #756bb1;"></div>Polar/Neutral</div>
            <div class="legend-item"><div class="legend-color" style="background-color: #3182bd;"></div>Basic (Positive)</div>
            <div class="legend-item"><div class="legend-color" style="background-color: #e6550d;"></div>Acidic (Negative)</div>
        </div>
        
        <div class="footer">
            Deep Proteomics &bull; Neural Word Embeddings Analysis
        </div>
    </div>

    <script>
        const rawData = {data_json};
        
        // Define colors matching static plot
        const colors = {{
            'Hydrophobic': '#2ca02c',
            'Polar/Neutral': '#756bb1',
            'Basic (Positive)': '#3182bd',
            'Acidic (Negative)': '#e6550d'
        }};
        
        // Split data by groups for multiple traces
        const groups = [...new Set(rawData.map(d => d.group))];
        const traces = groups.map(group => {{
            const groupData = rawData.filter(d => d.group === group);
            return {{
                x: groupData.map(d => d.x),
                y: groupData.map(d => d.y),
                text: groupData.map(d => `${{d.text}}<br>Hydrophobicity: ${{d.hydrophobicity.toFixed(2)}}<br>Group: ${{d.group}}`),
                mode: 'markers+text',
                name: group,
                marker: {{
                    size: 8,
                    color: colors[group],
                    opacity: 0.8,
                    line: {{
                        color: '#121214',
                        width: 0.5
                    }}
                }},
                textposition: 'top center',
                textfont: {{
                    size: 8,
                    color: '#a1a1aa'
                }},
                type: 'scatter'
            }};
        }});

        const layout = {{
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            margin: {{ t: 30, r: 30, b: 50, l: 50 }},
            hovermode: 'closest',
            showlegend: true,
            legend: {{
                font: {{ color: '#e4e4e7' }}
            }},
            xaxis: {{
                title: 'Component 1',
                titlefont: {{ color: '#a1a1aa' }},
                tickfont: {{ color: '#71717a' }},
                gridcolor: '#27272a',
                zerolinecolor: '#27272a'
            }},
            yaxis: {{
                title: 'Component 2',
                titlefont: {{ color: '#a1a1aa' }},
                tickfont: {{ color: '#71717a' }},
                gridcolor: '#27272a',
                zerolinecolor: '#27272a'
            }}
        }};

        Plotly.newPlot('plot', traces, layout);
    </script>
</body>
</html>
    """
    
    with open(html_path, 'w', encoding='utf-8') as fh:
        fh.write(html_content)
    print(f"Interactive plot saved to {html_path}")
