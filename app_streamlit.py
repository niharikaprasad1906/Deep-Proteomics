import streamlit as st
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Page Config
st.set_page_config(
    page_title="Deep Proteomics: Skip-gram Embedding Space Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Mode and Glassmorphic themes)
st.markdown("""
<style>
    /* Main body customization */
    .stApp {
        background-color: #09090b;
        color: #f4f4f5;
    }
    /* Headers gradient and glow */
    h1, h2 {
        font-family: 'Outfit', sans-serif;
        background: linear-gradient(135deg, #ffffff 40%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(18, 18, 24, 0.75);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    /* Cards style */
    .glass-card {
        background: rgba(30, 30, 40, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 15px;
    }
    /* Residue block styling */
    .res-block {
        display: inline-flex;
        flex-direction: column;
        align-items: center;
        border-radius: 6px;
        padding: 6px 8px;
        margin: 3px;
        font-family: monospace;
        font-weight: bold;
        min-width: 35px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }
    .res-hydrophobic { background-color: rgba(44, 160, 44, 0.18); color: #81c784; border-color: rgba(44, 160, 44, 0.3); }
    .res-polar { background-color: rgba(138, 114, 184, 0.18); color: #b39ddb; border-color: rgba(138, 114, 184, 0.3); }
    .res-basic { background-color: rgba(49, 130, 189, 0.18); color: #64b5f6; border-color: rgba(49, 130, 189, 0.3); }
    .res-acidic { background-color: rgba(230, 85, 13, 0.18); color: #ffb74d; border-color: rgba(230, 85, 13, 0.3); }
    
    /* Rotating logo animation */
    @keyframes rotateSlow {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

# Biochemical Definitions
BIO_COLORS = {
    'Hydrophobic': '#2ca02c',
    'Polar/Neutral': '#8a72b8',
    'Basic (Positive)': '#3182bd',
    'Acidic (Negative)': '#e6550d'
}

KYTE_DOOLITTLE = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

def get_kmer_hydrophobicity(kmer):
    scores = [KYTE_DOOLITTLE.get(aa, 0.0) for aa in kmer.upper()]
    return np.mean(scores) if scores else 0.0

def get_kmer_charge_class(kmer):
    kmer = kmer.upper()
    pos = sum(1 for aa in kmer if aa in 'KRH')
    neg = sum(1 for aa in kmer if aa in 'DE')
    if pos > neg:
        return 'Basic (Positive)'
    elif neg > pos:
        return 'Acidic (Negative)'
    
    hydrophobic_count = sum(1 for aa in kmer if aa in 'AVILPMFWY')
    if hydrophobic_count >= len(kmer) / 2.0:
        return 'Hydrophobic'
    else:
        return 'Polar/Neutral'

# Cache Model Load
@st.cache_resource
def load_w2v_model():
    from gensim.models import Word2Vec
    model_path = os.path.join("output", "skipgram_model.model")
    if os.path.exists(model_path):
        try:
            model = Word2Vec.load(model_path)
            return model, "Active Pre-trained Model"
        except Exception as e:
            st.sidebar.error(f"Error loading model: {e}")
    
    # Fallback to dummy model
    import random
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    dummy_corpus = []
    random.seed(42)
    for _ in range(200):
        seq = "".join(random.choices(amino_acids, k=random.randint(15, 30)))
        kmers = [seq[i:i+3] for i in range(len(seq) - 2)]
        dummy_corpus.append(kmers)
    
    model = Word2Vec(sentences=dummy_corpus, vector_size=64, window=4, min_count=1, epochs=5, sg=1)
    return model, "Demo Fallback Model (No trained model found)"

# Load resources
model, model_status = load_w2v_model()
vocab = list(model.wv.key_to_index.keys())

# Cache Projection Coordinates
@st.cache_data
def get_coordinates(vocab_list, vector_size, method, dims):
    limit = min(600, len(vocab_list))
    top_vocab = vocab_list[:limit]
    
    if limit == 0:
        return pd.DataFrame()
        
    vectors = np.array([model.wv[word] for word in top_vocab])
    
    if method == "tsne":
        perplexity = min(30, max(5, limit - 1))
        reducer = TSNE(n_components=dims, perplexity=perplexity, random_state=42, max_iter=1000)
    else:
        reducer = PCA(n_components=dims, random_state=42)
        
    coords = reducer.fit_transform(vectors)
    
    df = pd.DataFrame({
        "text": top_vocab,
        "x": coords[:, 0],
        "y": coords[:, 1],
        "hydrophobicity": [get_kmer_hydrophobicity(w) for w in top_vocab],
        "group": [get_kmer_charge_class(w) for w in top_vocab]
    })
    
    if dims == 3:
        df["z"] = coords[:, 2]
        
    return df

# Sidebar Controls (Rotating Colorful Logo Header)
st.sidebar.markdown("""
<div style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px; padding-bottom: 12px; border-bottom: 1px solid rgba(255,255,255,0.08);">
    <div style="font-size: 28px; display: inline-block; animation: rotateSlow 10s linear infinite; filter: drop-shadow(0 0 8px rgba(139, 92, 246, 0.35));">🧬</div>
    <div>
        <h1 style="font-size: 20px; font-weight: 700; margin: 0; background: linear-gradient(135deg, #ffffff 30%, #a78bfa 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: -0.5px; line-height: 1.2;">Deep Proteomics</h1>
        <span style="font-size: 11px; color: #71717a; text-transform: uppercase; letter-spacing: 1px; font-weight: 500;">Word2Vec k-mer Explorer</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Model Info Card
st.sidebar.markdown(f"""
<div class="glass-card">
    <div style="font-size: 11px; text-transform: uppercase; color: #71717a;">Model Status</div>
    <div style="font-size: 14px; font-weight: bold; color: #10b981;">{model_status}</div>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 5px; margin-top: 10px; font-size: 12px;">
        <div><b>Vocab:</b> {len(vocab):,}</div>
        <div><b>Dims:</b> {model.vector_size}</div>
        <div><b>Window:</b> {model.window}</div>
        <div><b>k-mer:</b> {len(vocab[0]) if vocab else 3}</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.header("🎛️ Projection Configuration")
proj_method = st.sidebar.selectbox("Dimension Reduction", ["tsne", "pca"], index=0, format_func=lambda x: "t-SNE" if x=="tsne" else "PCA")
proj_dims = st.sidebar.selectbox("Projection Mode", [2, 3], index=0, format_func=lambda x: f"{x}D Space")
color_scheme = st.sidebar.selectbox("Color Mapping", ["group", "hydrophobicity"], index=0, format_func=lambda x: "Biochemical Groups" if x=="group" else "Kyte-Doolittle Hydrophobicity")

# Get Coordinates
df_coords = get_coordinates(vocab, model.vector_size, proj_method, proj_dims)

# Search Box
st.sidebar.header("🔍 K-mer Decoder")
search_query = st.sidebar.text_input("Search a k-mer", value="", max_chars=4, placeholder="e.g. LAL, KRH...").upper().strip()

# Main Application Tabs
tab_map, tab_profile, tab_docs = st.tabs(["🧬 Embedding Projection", "📊 Sequence Profiler", "📖 Method & Glossary"])

# Tab 1: Embedding Map
with tab_map:
    col_chart, col_ins = st.columns([3, 1])
    
    with col_ins:
        st.subheader("🧠 K-mer Inspector")
        inspect_word = ""
        
        # Check query
        if search_query:
            if search_query in model.wv:
                inspect_word = search_query
            else:
                st.error(f"'{search_query}' not found in vocabulary.")
                
        if not inspect_word:
            st.info("Search a k-mer in the sidebar or check its nearest neighbors details here.")
        else:
            # Query similarities
            neighbors = model.wv.most_similar(inspect_word, topn=10)
            hydro = get_kmer_hydrophobicity(inspect_word)
            grp = get_kmer_charge_class(inspect_word)
            
            # Badge HTML
            bg_colors = {
                'Hydrophobic': 'res-hydrophobic',
                'Polar/Neutral': 'res-polar',
                'Basic (Positive)': 'res-basic',
                'Acidic (Negative)': 'res-acidic'
            }
            badge_class = bg_colors.get(grp, 'res-polar')
            
            st.markdown(f"""
            <div class="glass-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 32px; font-weight: bold; font-family: monospace;">{inspect_word}</span>
                    <span class="res-block {badge_class}" style="margin: 0; font-size: 11px; padding: 4px 8px;">{grp}</span>
                </div>
                <hr style="margin: 10px 0; border-color: rgba(255,255,255,0.08);"/>
                <div style="font-size: 12px; color: #a1a1aa; text-transform: uppercase;">Kyte-Doolittle Hydrophobicity</div>
                <div style="font-size: 20px; font-weight: bold; font-family: monospace;">{hydro:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 🔗 Nearest Neighbors")
            for neigh, score in neighbors:
                neigh_grp = get_kmer_charge_class(neigh)
                dot_color = BIO_COLORS.get(neigh_grp, "#999")
                
                # Create a card layout for neighbors
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.08); padding: 8px 12px; border-radius: 6px; margin-bottom: 5px;">
                    <div>
                        <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background-color: {dot_color}; margin-right: 8px;"></span>
                        <span style="font-family: monospace; font-weight: bold;">{neigh}</span>
                    </div>
                    <span style="font-family: monospace; color: #14b8a6;">{score:.4f}</span>
                </div>
                """, unsafe_allow_html=True)

    with col_chart:
        st.subheader("✨ Embedding Projections Map")
        
        if not df_coords.empty:
            # Highlight Neighbors on Chart
            highlight_words = []
            if inspect_word:
                highlight_words = [inspect_word] + [n[0] for n in model.wv.most_similar(inspect_word, topn=6)]
            
            fig = go.Figure()
            
            # Map configurations
            is_3d = proj_dims == 3
            plot_type = "scatter3d" if is_3d else "scatter"
            
            # Create Plotly traces
            if color_scheme == "group":
                # Discrete Biochemical groups
                for grp_name, grp_color in BIO_COLORS.items():
                    df_grp = df_coords[df_coords["group"] == grp_name]
                    if df_grp.empty: continue
                    
                    marker_opts = dict(
                        size=6 if is_3d else 8,
                        color=grp_color,
                        opacity=0.3 if highlight_words else 0.75,
                        line=dict(color='#09090b', width=0.5)
                    )
                    
                    trace_args = dict(
                        x=df_grp["x"],
                        y=df_grp["y"],
                        text=df_grp["text"],
                        mode='markers',
                        name=grp_name,
                        marker=marker_opts,
                        hovertemplate="<b>%{text}</b><br>Group: " + grp_name + "<extra></extra>"
                    )
                    if is_3d:
                        trace_args["z"] = df_grp["z"]
                        fig.add_trace(go.Scatter3d(**trace_args))
                    else:
                        fig.add_trace(go.Scatter(**trace_args))
            else:
                # Continuous Hydrophobicity color scale
                marker_opts = dict(
                    size=6 if is_3d else 8,
                    color=df_coords["hydrophobicity"],
                    colorscale="Coolwarm",
                    showscale=True,
                    colorbar=dict(title=dict(text="Hydrophobicity", font=dict(color="#a1a1aa", size=10)), tickfont=dict(color="#71717a")),
                    opacity=0.3 if highlight_words else 0.8,
                    line=dict(color='#09090b', width=0.5)
                )
                
                trace_args = dict(
                    x=df_coords["x"],
                    y=df_coords["y"],
                    text=df_coords["text"],
                    mode='markers',
                    name='k-mers',
                    marker=marker_opts,
                    hovertemplate="<b>%{text}</b><br>Hydrophobicity: %{marker.color:.2f}<extra></extra>"
                )
                if is_3d:
                    trace_args["z"] = df_coords["z"]
                    fig.add_trace(go.Scatter3d(**trace_args))
                else:
                    fig.add_trace(go.Scatter(**trace_args))
            
            # Add highlighting trace if a search query is active
            if highlight_words:
                df_high = df_coords[df_coords["text"].isin(highlight_words)]
                df_target = df_coords[df_coords["text"] == inspect_word]
                
                # Line Connections
                if not df_target.empty:
                    tx, ty = df_target.iloc[0]["x"], df_target.iloc[0]["y"]
                    tz = df_target.iloc[0]["z"] if is_3d else None
                    
                    for idx, row in df_high.iterrows():
                        if row["text"] == inspect_word: continue
                        line_args = dict(
                            x=[tx, row["x"]],
                            y=[ty, row["y"]],
                            mode='lines',
                            showlegend=False,
                            line=dict(color='rgba(20, 184, 166, 0.4)', width=1.5, dash='dash'),
                            hoverinfo='none'
                        )
                        if is_3d:
                            line_args["z"] = [tz, row["z"]]
                            fig.add_trace(go.Scatter3d(**line_args))
                        else:
                            fig.add_trace(go.Scatter(**line_args))
                
                # Neighbors markers
                df_neighbors = df_high[df_high["text"] != inspect_word]
                neigh_args = dict(
                    x=df_neighbors["x"],
                    y=df_neighbors["y"],
                    text=df_neighbors["text"],
                    mode='markers+text',
                    name='Nearest Neighbors',
                    textposition='top center',
                    textfont=dict(size=9, color='#f4f4f5'),
                    marker=dict(
                        size=8 if is_3d else 10,
                        color='#14b8a6',
                        line=dict(color='#fff', width=1),
                        opacity=0.95
                    )
                )
                if is_3d:
                    neigh_args["z"] = df_neighbors["z"]
                    fig.add_trace(go.Scatter3d(**neigh_args))
                else:
                    fig.add_trace(go.Scatter(**neigh_args))
                
                # Target word marker
                if not df_target.empty:
                    target_args = dict(
                        x=df_target["x"],
                        y=df_target["y"],
                        text=df_target["text"],
                        mode='markers+text',
                        name='Query Target',
                        textposition='top center',
                        textfont=dict(size=12, color='#fff'),
                        marker=dict(
                            size=12 if is_3d else 14,
                            color='#8b5cf6',
                            line=dict(color='#fff', width=2),
                            opacity=1
                        )
                    )
                    if is_3d:
                        target_args["z"] = df_target["z"]
                        fig.add_trace(go.Scatter3d(**target_args))
                    else:
                        fig.add_trace(go.Scatter(**target_args))
            
            # Dark layout settings
            layout_args = dict(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=0, r=0, b=0, l=0),
                legend=dict(
                    font=dict(color='#e4e4e7'),
                    bgcolor='rgba(10, 10, 15, 0.6)',
                    bordercolor='rgba(255, 255, 255, 0.05)',
                    borderwidth=1
                ),
                xaxis=dict(gridcolor='rgba(255, 255, 255, 0.05)', zerolinecolor='rgba(255, 255, 255, 0.1)', tickfont=dict(color='#71717a')),
                yaxis=dict(gridcolor='rgba(255, 255, 255, 0.05)', zerolinecolor='rgba(255, 255, 255, 0.1)', tickfont=dict(color='#71717a'))
            )
            
            if is_3d:
                layout_args["scene"] = dict(
                    xaxis=dict(gridcolor='rgba(255, 255, 255, 0.05)', backgroundcolor='rgba(0,0,0,0)', zerolinecolor='rgba(255, 255, 255, 0.1)', color='#71717a'),
                    yaxis=dict(gridcolor='rgba(255, 255, 255, 0.05)', backgroundcolor='rgba(0,0,0,0)', zerolinecolor='rgba(255, 255, 255, 0.1)', color='#71717a'),
                    zaxis=dict(gridcolor='rgba(255, 255, 255, 0.05)', backgroundcolor='rgba(0,0,0,0)', zerolinecolor='rgba(255, 255, 255, 0.1)', color='#71717a')
                )
                
            fig.update_layout(**layout_args)
            st.plotly_chart(fig, use_container_width=True)

# Tab 2: Sequence Profiler
with tab_profile:
    st.subheader("📊 Protein Sequence Profiler")
    st.write("Paste an amino acid sequence to project its sub-segments and evaluate physical property changes.")
    
    sample_seq = "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN"
    
    seq_input = st.text_area("Amino Acid Sequence", value="", placeholder="Paste residues (e.g. MTAIIKE...)", height=100).upper().replace("\n", "").replace(" ", "")
    
    col_btns = st.columns([1, 1, 4])
    with col_btns[0]:
        btn_analyze = st.button("Analyze Sequence", type="primary")
    with col_btns[1]:
        btn_sample = st.button("Load Human Insulin")
        if btn_sample:
            seq_input = sample_seq
            st.rerun()
            
    if btn_analyze or seq_input:
        cleaned_seq = [aa for aa in seq_input if aa in "ACDEFGHIKLMNPQRSTVWY"]
        seq_str = "".join(cleaned_seq)
        
        k = len(vocab[0]) if vocab else 3
        if len(seq_str) < k:
            st.error(f"Sequence is too short (must be at least {k} residues).")
        else:
            # Segment k-mers
            kmers = [seq_str[i:i+k] for i in range(len(seq_str) - k + 1)]
            
            # Calculate metrics
            profile_data = []
            for idx, kmer in enumerate(kmers):
                in_vocab = kmer in model.wv
                hydro = get_kmer_hydrophobicity(kmer)
                grp = get_kmer_charge_class(kmer)
                vector_norm = float(np.linalg.norm(model.wv[kmer])) if in_vocab else 0.0
                
                next_sim = 0.0
                if in_vocab and idx < len(kmers) - 1 and kmers[idx+1] in model.wv:
                    next_sim = float(model.wv.similarity(kmer, kmers[idx+1]))
                    
                profile_data.append({
                    "index": idx,
                    "kmer": kmer,
                    "in_vocab": in_vocab,
                    "hydrophobicity": hydro,
                    "group": grp,
                    "vector_magnitude": vector_norm,
                    "next_similarity": next_sim
                })
            
            # General Metrics Dashboard
            st.write("### 📈 Sequence Statistics")
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("Sequence Length", len(seq_str))
            with col_m2:
                avg_hydro = np.mean([p["hydrophobicity"] for p in profile_data])
                st.metric("Mean Hydrophobicity", f"{avg_hydro:.2f}")
            with col_m3:
                coverage = sum(1 for p in profile_data if p["in_vocab"]) / len(profile_data)
                st.metric("Vocabulary Coverage", f"{coverage * 100:.1f}%")
                
            # Render Charts
            st.write("### 📊 Biophysical Profiling Plots")
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                # Hydrophobicity Plot
                fig_h = px.line(
                    x=[p["index"] for p in profile_data],
                    y=[p["hydrophobicity"] for p in profile_data],
                    labels={"x": "Sequence Index", "y": "Average Hydrophobicity (Kyte-Doolittle)"},
                    title="Localized Hydrophobicity Scale"
                )
                fig_h.update_traces(line_color="#10b981", mode="lines+markers", marker=dict(size=4))
                fig_h.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(gridcolor='rgba(255, 255, 255, 0.05)'), yaxis=dict(gridcolor='rgba(255, 255, 255, 0.05)'))
                st.plotly_chart(fig_h, use_container_width=True)
                
            with col_chart2:
                # Coherence Plot
                fig_s = px.bar(
                    x=[p["index"] for p in profile_data][:-1],
                    y=[p["next_similarity"] for p in profile_data][:-1],
                    labels={"x": "Sequence Index", "y": "Cosine Similarity"},
                    title="Embedding Transition Coherence"
                )
                fig_s.update_traces(marker_color="#6366f1", opacity=0.8)
                fig_s.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(gridcolor='rgba(255, 255, 255, 0.05)'), yaxis=dict(gridcolor='rgba(255, 255, 255, 0.05)'))
                st.plotly_chart(fig_s, use_container_width=True)
                
            # Colored Residues Grid
            st.write("### 🧩 Sequence Residue Decoder")
            st.caption("Clicking a residue segment searches its corresponding k-mer in the Explorer.")
            
            html_blocks = []
            bg_styles = {
                'Hydrophobic': 'res-hydrophobic',
                'Polar/Neutral': 'res-polar',
                'Basic (Positive)': 'res-basic',
                'Acidic (Negative)': 'res-acidic'
            }
            
            for i, aa in enumerate(seq_str):
                prof_idx = min(i, len(profile_data)-1)
                info = profile_data[prof_idx]
                grp = info["group"] if info else 'Polar/Neutral'
                class_style = bg_styles.get(grp, 'res-polar')
                
                # Block HTML
                block_title = f"Residue: {aa} | Position: {i+1}\\nGroup: {grp}\\nk-mer: {info['kmer']} (Hydro: {info['hydrophobicity']:.2f})"
                html_blocks.append(f'<span class="res-block {class_style}" title="{block_title}">{aa}<span style="font-size: 8px; font-weight: normal; opacity:0.6; display:block;">{i+1}</span></span>')
                
            st.markdown(f'<div style="background-color: rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.08); padding: 15px; border-radius: 8px; line-height: 2;">{"".join(html_blocks)}</div>', unsafe_allow_html=True)

# Tab 3: Docs
with tab_docs:
    st.subheader("📖 Biological Word Embeddings")
    
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.write("""
        ### 🔬 Methodology
        This project applies natural language processing algorithms (specifically Word2Vec / Skip-gram) to genomics/proteomics. In this framework:
        
        * **Protein Sequences** act as sentences.
        * **Overlapping k-mers** (sub-strings of length *k*) act as words.
        * **Skip-gram model** predicts surrounding context k-mers given a target k-mer.
        
        Through this training, k-mers that appear in similar biological environments (e.g. alpha-helices, hydrophobic cores, active sites) are mapped to close vectors in the dense multi-dimensional space.
        """)
        
    with col_d2:
        st.write("### 🧪 Biochemical Property Glossary")
        st.markdown(f"""
        * <span style="color:{BIO_COLORS['Hydrophobic']}; font-weight:bold;">Hydrophobic</span>: Non-polar side chains (A, V, I, L, P, M, F, W, Y) that hide from water, usually composing the inner core of folded proteins.
        * <span style="color:{BIO_COLORS['Polar/Neutral']}; font-weight:bold;">Polar/Neutral</span>: Uncharged polar side chains (S, T, C, N, Q, G) that form hydrogen bonds with water and other polar atoms.
        * <span style="color:{BIO_COLORS['Basic (Positive)']}; font-weight:bold;">Basic / Positive</span>: Positively charged side chains (K, R, H) at physiological pH, highly hydrophilic and active in electrostatic binding.
        * <span style="color:{BIO_COLORS['Acidic (Negative)']}; font-weight:bold;">Acidic / Negative</span>: Negatively charged side chains (D, E) that are soluble and often participate in salt-bridges or metal binding.
        """, unsafe_allow_html=True)
