// Deep Proteomics App State
const state = {
    modelInfo: null,
    embeddings: [],
    selectedKmer: null,
    activeMethod: 'tsne',
    activeDims: 2,
    activeColorScheme: 'group',
    isLoading: false
};

// UI Elements
const el = {
    valK: document.getElementById('val-k'),
    valDim: document.getElementById('val-dim'),
    valVocab: document.getElementById('val-vocab'),
    valWindow: document.getElementById('val-window'),
    modelStatusText: document.getElementById('model-status-text'),
    
    projectionMethod: document.getElementById('projection-method'),
    projectionDims: document.getElementById('projection-dims'),
    colorScheme: document.getElementById('color-scheme'),
    
    kmerSearchInput: document.getElementById('kmer-search-input'),
    kmerSearchBtn: document.getElementById('kmer-search-btn'),
    searchFeedback: document.getElementById('search-feedback'),
    
    plotLoader: document.getElementById('plot-loader'),
    plotlyChart: document.getElementById('plotly-chart'),
    vizTitle: document.getElementById('viz-title'),
    
    inspectorPlaceholder: document.getElementById('inspector-placeholder'),
    inspectorContent: document.getElementById('inspector-content'),
    insKmer: document.getElementById('ins-kmer'),
    insGroup: document.getElementById('ins-group'),
    insHydro: document.getElementById('ins-hydro'),
    insNeighbors: document.getElementById('ins-neighbors'),
    
    tabBtns: document.querySelectorAll('.tab-btn'),
    tabContents: document.querySelectorAll('.tab-content'),
    
    sequenceInput: document.getElementById('sequence-input'),
    seqAnalyzeBtn: document.getElementById('seq-analyze-btn'),
    seqSampleBtn: document.getElementById('seq-sample-btn'),
    seqError: document.getElementById('seq-error'),
    seqResults: document.getElementById('seq-results'),
    
    seqLenVal: document.getElementById('seq-len-val'),
    seqHydroVal: document.getElementById('seq-hydro-val'),
    seqCoverageVal: document.getElementById('seq-coverage-val'),
    seqHydroChart: document.getElementById('seq-hydro-chart'),
    seqSimChart: document.getElementById('seq-sim-chart'),
    seqResiduesMap: document.getElementById('sequence-residues-map')
};

// Biochemical Groups Color Scheme
const BIO_COLORS = {
    'Hydrophobic': '#2ca02c',
    'Polar/Neutral': '#8a72b8',
    'Basic (Positive)': '#3182bd',
    'Acidic (Negative)': '#e6550d'
};

const BIO_DESCRIPTIONS = {
    'Hydrophobic': 'Lipophilic side chains. Typically buried in protein cores.',
    'Polar/Neutral': 'Hydrophilic uncharged side chains. Often found on surfaces forming hydrogen bonds.',
    'Basic (Positive)': 'Positively charged side chains (Lys, Arg, His). Key in electrostatic interactions.',
    'Acidic (Negative)': 'Negatively charged side chains (Asp, Glu). Soluble, involved in salt-bridges.'
};

// Sample Sequence for demo (Human Serum Albumin segment)
const SAMPLE_SEQUENCE = "MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKSLHTLFGDKLCTVATLRETYGEMADCCAKQEPERNECFLQHKDDNPNLPRLVRPEVDVMCTAFHDNEETFLKKYLYEIARRHPYFYAPELLFFAKRYKAAFTECCQAADKAACLLPKLDELRDEGKASSAKQRLKCASLQKFGERAFKAWAVARLSQRFPKAEFAEVSKLVTDLTKVHTECCHGDLLECADDRADLAKYICENQDSISSKLKECCEKPLLEKSHCIAEVENDEMPADLPSLAADFVESKDVCKNYAEAKDVFLGMFLYEYARRHPDYSVVLLLRLAKTYETTLEKCCAAADPHECYAKVFDEFKPLVEEPQNLIKQNCELFEQLGEYKFQNALLVRYTKKVPQVSTPTLVEVSRNLGKVGSKCCKHPEAKRMPCAEDYLSVVLNQLCVLHEKTPVSDRVTKCCTESLVNRRPCFSALEVDETYVPKEFNAETFTFHADICTLSEKERQIKKQTALVELVKHKPKATKEQLKAVMDDFAAFVEKCCKADDKETCFAEEGKKLVAASQAALGL";

// Initialize App
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    fetchModelInfo();
    fetchEmbeddings();
    initEventListeners();
});

// Setup navigation tabs
function initTabs() {
    el.tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.getAttribute('data-tab');
            
            // Toggle active buttons
            el.tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // Toggle active views
            el.tabContents.forEach(content => {
                content.classList.remove('active');
                if (content.getAttribute('id') === tabId) {
                    content.classList.add('active');
                }
            });
            
            // Replot if switching back to embedding map (fixes rendering dimensions bugs)
            if (tabId === 'tab-map') {
                plotEmbeddings();
            }
        });
    });
}

// Bind interactive event listeners
function initEventListeners() {
    el.projectionMethod.addEventListener('change', (e) => {
        state.activeMethod = e.target.value;
        fetchEmbeddings();
    });
    
    el.projectionDims.addEventListener('change', (e) => {
        state.activeDims = parseInt(e.target.value);
        fetchEmbeddings();
    });
    
    el.colorScheme.addEventListener('change', (e) => {
        state.activeColorScheme = e.target.value;
        plotEmbeddings();
    });
    
    el.kmerSearchBtn.addEventListener('click', () => {
        handleSearch();
    });
    
    el.kmerSearchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            handleSearch();
        }
    });
    
    el.seqAnalyzeBtn.addEventListener('click', () => {
        analyzeSequence();
    });
    
    el.seqSampleBtn.addEventListener('click', () => {
        el.sequenceInput.value = SAMPLE_SEQUENCE;
        analyzeSequence();
    });
}

// Show/Hide loaders
function setLoader(loading) {
    state.isLoading = loading;
    el.plotLoader.style.display = loading ? 'block' : 'none';
}

// Load Model metadata
async function fetchModelInfo() {
    try {
        const response = await fetch('/api/info');
        const data = await response.json();
        state.modelInfo = data;
        
        // Update view
        el.valK.innerText = data.k;
        el.valDim.innerText = data.vector_size;
        el.valVocab.innerText = data.vocab_size.toLocaleString();
        el.valWindow.innerText = data.window;
        
        el.modelStatusText.innerText = "Model Active";
    } catch (err) {
        console.error("Failed to load model info:", err);
        el.modelStatusText.innerText = "Error loading model";
        el.modelStatusText.parentElement.style.background = 'rgba(239, 68, 68, 0.1)';
        el.modelStatusText.parentElement.style.borderColor = 'rgba(239, 68, 68, 0.3)';
        el.modelStatusText.previousElementSibling.style.backgroundColor = '#ef4444';
    }
}

// Load coordinates based on filters
async function fetchEmbeddings() {
    setLoader(true);
    try {
        const url = `/api/embeddings?method=${state.activeMethod}&dims=${state.activeDims}`;
        const response = await fetch(url);
        const data = await response.json();
        state.embeddings = data;
        
        // Dynamic title
        el.vizTitle.innerText = `${state.activeMethod.toUpperCase()} ${state.activeDims}D Projection`;
        
        plotEmbeddings();
    } catch (err) {
        console.error("Failed to fetch coordinates:", err);
    } finally {
        setLoader(false);
    }
}

// Plot embeddings using Plotly.js
function plotEmbeddings(highlightNeighbors = null) {
    if (!state.embeddings || state.embeddings.length === 0) return;
    
    const is3D = state.activeDims === 3;
    const isColorByGroup = state.activeColorScheme === 'group';
    
    let traces = [];
    
    if (isColorByGroup) {
        // Group points by biochemical classes for distinct legend toggling
        const groups = [...new Set(state.embeddings.map(d => d.group))];
        
        groups.forEach(groupName => {
            const groupData = state.embeddings.filter(d => d.group === groupName);
            
            const trace = {
                x: groupData.map(d => d.x),
                y: groupData.map(d => d.y),
                text: groupData.map(d => d.text),
                customdata: groupData.map(d => ({ hydro: d.hydrophobicity, grp: d.group })),
                hovertemplate: '<b>%{text}</b><br>' +
                              'Hydrophobicity: %{customdata.hydro:.2f}<br>' +
                              'Group: %{customdata.grp}<extra></extra>',
                mode: 'markers',
                name: groupName,
                marker: {
                    size: is3D ? 5 : 7,
                    color: BIO_COLORS[groupName] || '#999',
                    opacity: highlightNeighbors ? 0.25 : 0.7,
                    line: {
                        color: '#09090b',
                        width: 0.5
                    }
                },
                type: is3D ? 'scatter3d' : 'scatter'
            };
            if (is3D) trace.z = groupData.map(d => d.z);
            traces.push(trace);
        });
    } else {
        // Continuous Kyte-Doolittle colorbar mapping
        const trace = {
            x: state.embeddings.map(d => d.x),
            y: state.embeddings.map(d => d.y),
            text: state.embeddings.map(d => d.text),
            customdata: state.embeddings.map(d => ({ hydro: d.hydrophobicity, grp: d.group })),
            hovertemplate: '<b>%{text}</b><br>' +
                          'Hydrophobicity: %{customdata.hydro:.2f}<br>' +
                          'Group: %{customdata.grp}<extra></extra>',
            mode: 'markers',
            name: 'k-mers',
            marker: {
                size: is3D ? 5 : 7,
                color: state.embeddings.map(d => d.hydrophobicity),
                colorscale: 'RdBu_r',
                showscale: true,
                colorbar: {
                    title: 'Kyte-Doolittle',
                    titlefont: { color: '#a1a1aa', size: 10 },
                    tickfont: { color: '#71717a' }
                },
                opacity: highlightNeighbors ? 0.25 : 0.75,
                line: {
                    color: '#09090b',
                    width: 0.5
                }
            },
            type: is3D ? 'scatter3d' : 'scatter'
        };
        if (is3D) trace.z = state.embeddings.map(d => d.z);
        traces.push(trace);
    }
    
    // Highlight specific k-mer & neighbors if queried
    if (highlightNeighbors && highlightNeighbors.length > 0) {
        const queryText = highlightNeighbors[0].text;
        const neighborTexts = highlightNeighbors.map(n => n.text);
        
        // Find coordinates for highlight items
        const highlightedCoords = state.embeddings.filter(d => neighborTexts.includes(d.text));
        const queryCoord = highlightedCoords.find(c => c.text === queryText);
        const otherNeighbors = highlightedCoords.filter(c => c.text !== queryText);
        
        // Trace for neighbors
        if (otherNeighbors.length > 0) {
            const neighborTrace = {
                x: otherNeighbors.map(d => d.x),
                y: otherNeighbors.map(d => d.y),
                text: otherNeighbors.map(d => d.text),
                customdata: otherNeighbors.map(d => {
                    const match = highlightNeighbors.find(n => n.text === d.text);
                    return { hydro: d.hydrophobicity, score: match ? match.similarity : 0.0 };
                }),
                hovertemplate: '<b>%{text}</b> (Neighbor)<br>' +
                              'Cosine Similarity: %{customdata.score:.4f}<br>' +
                              'Hydrophobicity: %{customdata.hydro:.2f}<extra></extra>',
                mode: 'markers+text',
                name: 'Nearest Neighbors',
                textposition: 'top center',
                textfont: { size: 9, color: '#f4f4f5', family: 'Outfit' },
                marker: {
                    size: is3D ? 8 : 10,
                    color: '#14b8a6', // Teal glow
                    line: {
                        color: '#fff',
                        width: 1
                    },
                    opacity: 0.95
                },
                type: is3D ? 'scatter3d' : 'scatter'
            };
            if (is3D) neighborTrace.z = otherNeighbors.map(d => d.z);
            traces.push(neighborTrace);
            
            // Draw connection lines from query to neighbors
            if (queryCoord) {
                otherNeighbors.forEach(neigh => {
                    const lineTrace = {
                        x: [queryCoord.x, neigh.x],
                        y: [queryCoord.y, neigh.y],
                        mode: 'lines',
                        showlegend: false,
                        hoverinfo: 'none',
                        line: {
                            color: 'rgba(20, 184, 166, 0.4)',
                            width: 1.5,
                            dash: 'dash'
                        },
                        type: is3D ? 'scatter3d' : 'scatter'
                    };
                    if (is3D) lineTrace.z = [queryCoord.z, neigh.z];
                    traces.push(lineTrace);
                });
            }
        }
        
        // Trace for target query k-mer
        if (queryCoord) {
            const targetTrace = {
                x: [queryCoord.x],
                y: [queryCoord.y],
                text: [queryCoord.text],
                customdata: [{ hydro: queryCoord.hydrophobicity }],
                hovertemplate: '<b>%{text} (Query Target)</b><br>Hydrophobicity: %{customdata.hydro:.2f}<extra></extra>',
                mode: 'markers+text',
                name: 'Query Target',
                textposition: 'top center',
                textfont: { size: 12, color: '#fff', weight: 'bold', family: 'Outfit' },
                marker: {
                    size: is3D ? 12 : 14,
                    color: '#8b5cf6', // Violet pulse
                    line: {
                        color: '#fff',
                        width: 2
                    },
                    opacity: 1
                },
                type: is3D ? 'scatter3d' : 'scatter'
            };
            if (is3D) targetTrace.z = [queryCoord.z];
            traces.push(targetTrace);
        }
    }
    
    // Layout configurations for Plotly
    const layout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        margin: { t: 20, r: 20, b: 30, l: 30 },
        hovermode: 'closest',
        showlegend: !highlightNeighbors,
        legend: {
            font: { color: '#e4e4e7', family: 'Outfit', size: 11 },
            bgcolor: 'rgba(10, 10, 15, 0.6)',
            bordercolor: 'rgba(255, 255, 255, 0.05)',
            borderwidth: 1
        },
        xaxis: {
            gridcolor: 'rgba(255, 255, 255, 0.05)',
            zerolinecolor: 'rgba(255, 255, 255, 0.1)',
            tickfont: { color: '#71717a', family: 'Outfit' }
        },
        yaxis: {
            gridcolor: 'rgba(255, 255, 255, 0.05)',
            zerolinecolor: 'rgba(255, 255, 255, 0.1)',
            tickfont: { color: '#71717a', family: 'Outfit' }
        }
    };
    
    if (is3D) {
        layout.scene = {
            xaxis: {
                gridcolor: 'rgba(255, 255, 255, 0.05)',
                backgroundcolor: 'rgba(0,0,0,0)',
                zerolinecolor: 'rgba(255, 255, 255, 0.1)',
                color: '#71717a'
            },
            yaxis: {
                gridcolor: 'rgba(255, 255, 255, 0.05)',
                backgroundcolor: 'rgba(0,0,0,0)',
                zerolinecolor: 'rgba(255, 255, 255, 0.1)',
                color: '#71717a'
            },
            zaxis: {
                gridcolor: 'rgba(255, 255, 255, 0.05)',
                backgroundcolor: 'rgba(0,0,0,0)',
                zerolinecolor: 'rgba(255, 255, 255, 0.1)',
                color: '#71717a'
            },
            camera: {
                eye: { x: 1.5, y: 1.5, z: 1.2 }
            }
        };
    }
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['select2d', 'lasso2d']
    };
    
    Plotly.newPlot(el.plotlyChart, traces, layout, config);
    
    // Bind points click handler
    el.plotlyChart.on('plotly_click', (data) => {
        if (data.points && data.points.length > 0) {
            const kmer = data.points[0].text;
            if (kmer) {
                inspectKmer(kmer);
            }
        }
    });
}

// Search handler
async function handleSearch() {
    const query = el.kmerSearchInput.value.toUpperCase().trim();
    if (!query) return;
    
    el.searchFeedback.innerText = "";
    try {
        await inspectKmer(query);
    } catch (err) {
        // handled in inspectKmer
    }
}

// Load detail info for specific k-mer
async function inspectKmer(kmer) {
    try {
        const response = await fetch(`/api/similarity?query=${kmer}`);
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || "Failed similarity fetch");
        }
        
        const neighbors = await response.json();
        
        // The first item in results is the query target
        const queryObj = neighbors[0];
        const neighborItems = neighbors.slice(1);
        
        state.selectedKmer = kmer;
        
        // Update UI
        el.inspectorPlaceholder.classList.add('hidden');
        el.inspectorContent.classList.remove('hidden');
        
        el.insKmer.innerText = queryObj.text;
        el.insHydro.innerText = queryObj.hydrophobicity.toFixed(2);
        
        // Group Class Badge styling
        el.insGroup.innerText = queryObj.group;
        el.insGroup.className = "ins-badge"; // reset classes
        if (queryObj.group === 'Hydrophobic') el.insGroup.classList.add('badge-hydrophobic');
        else if (queryObj.group === 'Polar/Neutral') el.insGroup.classList.add('badge-polar');
        else if (queryObj.group === 'Basic (Positive)') el.insGroup.classList.add('badge-basic');
        else if (queryObj.group === 'Acidic (Negative)') el.insGroup.classList.add('badge-acidic');
        
        // Set description
        document.getElementById('ins-description').innerText = BIO_DESCRIPTIONS[queryObj.group] || "";
        
        // Render similarity neighbor cards
        el.insNeighbors.innerHTML = '';
        neighborItems.forEach(n => {
            const card = document.createElement('div');
            card.className = 'neighbor-card';
            
            // Dot color
            const dotColor = BIO_COLORS[n.group] || '#71717a';
            
            card.innerHTML = `
                <div class="neighbor-info">
                    <span class="neighbor-class-dot" style="background-color: ${dotColor}"></span>
                    <span class="neighbor-word">${n.text}</span>
                </div>
                <span class="neighbor-score">${n.similarity.toFixed(4)}</span>
            `;
            
            card.addEventListener('click', () => {
                inspectKmer(n.text);
            });
            
            el.insNeighbors.appendChild(card);
        });
        
        // Sync search input box
        el.kmerSearchInput.value = kmer;
        
        // Replot to draw highlights
        plotEmbeddings(neighbors);
        
    } catch (err) {
        console.error(err);
        el.searchFeedback.innerText = err.message || "K-mer not found in model vocabulary.";
        el.searchFeedback.style.color = '#ef4444';
    }
}

// Sequence Profiler Analysis
async function analyzeSequence() {
    const rawSeq = el.sequenceInput.value.toUpperCase().replace(/[^A-Z]/g, '').trim();
    el.seqError.innerText = "";
    
    if (!rawSeq) {
        el.seqError.innerText = "Please enter a valid protein sequence.";
        return;
    }
    
    const k = state.modelInfo ? state.modelInfo.k : 3;
    if (rawSeq.length < k) {
        el.seqError.innerText = `Sequence length must be at least ${k} (the size of k-mer words in current model).`;
        return;
    }
    
    try {
        const response = await fetch(`/api/sequence_profile?seq=${rawSeq}`);
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || "Failed profile processing");
        }
        
        const profile = await response.json();
        
        // Update general metrics
        el.seqLenVal.innerText = rawSeq.length;
        
        const coverageCount = profile.filter(p => p.in_vocab).length;
        const coveragePercent = Math.round((coverageCount / profile.length) * 100);
        el.seqCoverageVal.innerText = `${coveragePercent}%`;
        
        const avgHydro = profile.reduce((sum, p) => sum + p.hydrophobicity, 0) / profile.length;
        el.seqHydroVal.innerText = avgHydro.toFixed(2);
        
        // Render charts & visualization grid
        el.seqResults.classList.remove('hidden');
        
        plotSequenceCharts(profile);
        plotResidueGrid(rawSeq, profile);
        
    } catch (err) {
        console.error(err);
        el.seqError.innerText = err.message || "Failed to analyze sequence.";
    }
}

// Plot sequence line charts using Plotly
function plotSequenceCharts(profile) {
    const xCoords = profile.map(p => p.index);
    const kmerTexts = profile.map(p => p.kmer);
    
    // Hydrophobicity plot
    const traceHydro = {
        x: xCoords,
        y: profile.map(p => p.hydrophobicity),
        text: kmerTexts,
        hovertemplate: 'Index: %{x}<br>k-mer: <b>%{text}</b><br>Hydrophobicity: %{y:.2f}<extra></extra>',
        type: 'scatter',
        mode: 'lines+markers',
        line: { color: '#10b981', width: 2 },
        marker: { size: 5, color: '#10b981' }
    };
    
    const layoutHydro = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        margin: { t: 10, r: 10, b: 30, l: 35 },
        xaxis: { gridcolor: 'rgba(255, 255, 255, 0.05)', tickfont: { color: '#71717a' } },
        yaxis: { gridcolor: 'rgba(255, 255, 255, 0.05)', tickfont: { color: '#71717a' }, title: 'Kyte-Doolittle Score', titlefont: { color: '#71717a', size: 11 } }
    };
    
    Plotly.newPlot(el.seqHydroChart, [traceHydro], layoutHydro, { responsive: true, displayModeBar: false });
    
    // Embedding Coherence (Similarity with next k-mer)
    const traceSim = {
        x: xCoords.slice(0, -1),
        y: profile.map(p => p.next_similarity).slice(0, -1),
        text: kmerTexts.slice(0, -1).map((k, i) => `${k} ➜ ${kmerTexts[i+1]}`),
        hovertemplate: 'Transition: <b>%{text}</b><br>Cosine Similarity: %{y:.4f}<extra></extra>',
        type: 'bar',
        marker: {
            color: profile.map(p => p.next_similarity).slice(0, -1).map(s => s > 0.5 ? '#8b5cf6' : '#6366f1'),
            opacity: 0.8
        }
    };
    
    const layoutSim = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        margin: { t: 10, r: 10, b: 30, l: 35 },
        xaxis: { gridcolor: 'rgba(255, 255, 255, 0.05)', tickfont: { color: '#71717a' } },
        yaxis: { gridcolor: 'rgba(255, 255, 255, 0.05)', tickfont: { color: '#71717a' }, title: 'Similarity [0 - 1]', titlefont: { color: '#71717a', size: 11 } }
    };
    
    Plotly.newPlot(el.seqSimChart, [traceSim], layoutSim, { responsive: true, displayModeBar: false });
}

// Render dynamic sequence blocks colored by biochem properties
function plotResidueGrid(rawSeq, profile) {
    el.seqResiduesMap.innerHTML = '';
    
    // We map residues 1-to-1 to their primary k-mer properties.
    // For overlap stride=1, residue index i corresponds to the properties of the k-mer at index i.
    // (For the final k-1 residues which don't start a full k-mer, they share the property of the last k-mer)
    
    for (let i = 0; i < rawSeq.length; i++) {
        const residue = rawSeq[i];
        
        // Find corresponding k-mer profile item
        let profileIndex = i;
        if (profileIndex >= profile.length) {
            profileIndex = profile.length - 1;
        }
        
        const info = profile[profileIndex];
        const group = info ? info.group : 'Polar/Neutral';
        
        const block = document.createElement('div');
        block.className = 'residue-block';
        
        // Set dynamic class matching group properties
        if (group === 'Hydrophobic') block.classList.add('residue-hydrophobic');
        else if (group === 'Polar/Neutral') block.classList.add('residue-polar');
        else if (group === 'Basic (Positive)') block.classList.add('residue-basic');
        else if (group === 'Acidic (Negative)') block.classList.add('residue-acidic');
        
        block.innerHTML = `
            <span class="res-char">${residue}</span>
            <span class="res-index">${i + 1}</span>
        `;
        
        // Add hover tooltip details
        const tooltipKmer = info ? info.kmer : '';
        const tooltipHydro = info ? info.hydrophobicity.toFixed(2) : '0.00';
        block.title = `Residue: ${residue} | Position: ${i + 1}\nGroup: ${group}\nk-mer: ${tooltipKmer} (Hydrophobicity: ${tooltipHydro})`;
        
        // Clicking a block searches its corresponding k-mer
        if (info && info.in_vocab) {
            block.style.cursor = 'pointer';
            block.addEventListener('click', () => {
                // Switch to dashboard map tab
                document.querySelector('[data-tab="tab-map"]').click();
                // Inspect k-mer
                inspectKmer(info.kmer);
            });
        }
        
        el.seqResiduesMap.appendChild(block);
    }
}
