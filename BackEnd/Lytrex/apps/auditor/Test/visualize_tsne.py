import os
import sys
import subprocess

# ==========================================
# AUTO-INSTALLER: Fixes Environment Mismatches
# ==========================================
def install_packages():
    packages = ["plotly", "pandas", "scikit-learn", "numpy", "dash"]
    print("\n[!] Missing libraries detected. Auto-installing into the correct environment...")
    for pkg in packages:
        print(f" -> Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet"])
    print("[+] Installation complete! Loading visualizer...\n")

try:
    import numpy as np
    import pandas as pd
    import plotly.express as px
    from sklearn.manifold import TSNE
    from dash import Dash, dcc, html, Input, Output
except ImportError:
    install_packages()
    import numpy as np
    import pandas as pd
    import plotly.express as px
    from sklearn.manifold import TSNE
    from dash import Dash, dcc, html, Input, Output

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

# ---------------------------------------------------------
# DUMMY EMBEDDINGS: Bypasses the need for an OpenAI API Key
# ---------------------------------------------------------
class DummyEmbeddings(Embeddings):
    def embed_documents(self, texts): return []
    def embed_query(self, text): return []

def load_and_visualize_tsne_3d():
    print("="*60)
    print(" LYTREX 3D T-SNE DASHBOARD (OFFLINE MODE) ")
    print("="*60)
    
    # 1. Get User Input
    print("\nSelect Framework Database to Visualize (NCA, ECC, SAMA, or ALL):")
    framework_name = input("Framework: ").strip().upper() or "ALL"
    
    # 2. Setup Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, "LytrexDB_OpenAI", framework_name)
    
    if not os.path.exists(db_path):
        print(f"\n[ERROR] Database not found at: {db_path}")
        print("Please run an audit on this framework first to generate the database.")
        return

    print(f"\n[1/4] Loading FAISS Database for {framework_name} (Offline)...")
    dummy_embeddings = DummyEmbeddings()
    vectorstore = FAISS.load_local(db_path, dummy_embeddings, allow_dangerous_deserialization=True)
    
    # 3. Extract Vectors and Text Metadata
    print("[2/4] Extracting mathematical vectors and text chunks...")
    num_vectors = vectorstore.index.ntotal
    raw_vectors = vectorstore.index.reconstruct_n(0, num_vectors)
    
    docstore_dict = vectorstore.docstore._dict
    index_to_id = vectorstore.index_to_docstore_id
    
    hover_texts = []
    full_texts = []
    parent_texts = []
    
    for i in range(num_vectors):
        doc_id = index_to_id[i]
        doc = docstore_dict[doc_id]
        
        # Format short text for hover
        clean_text = doc.page_content.replace("\n", " ")
        short_text = clean_text[:100] + "..." if len(clean_text) > 100 else clean_text
        hover_texts.append(short_text)
        
        # Keep the exact full text for the click event
        full_texts.append(doc.page_content)
        parent_texts.append("Framework Chunk")

    X = np.array(raw_vectors)
    print(f"      -> Extracted {X.shape[0]} chunks with {X.shape[1]} dimensions.")

    # 4. Perform 3D t-SNE Dimensionality Reduction
    print("[3/4] Compressing 3072 dimensions to 3D using t-SNE (This will take a moment)...")
    safe_perplexity = min(30, max(5, X.shape[0] - 1)) 
    tsne = TSNE(n_components=3, perplexity=safe_perplexity, random_state=42, init='pca', learning_rate='auto')
    X_3d = tsne.fit_transform(X)

    # 5. Build the DataFrame
    print("[4/4] Generating Interactive Dashboard...")
    df = pd.DataFrame({
        'X Axis': X_3d[:, 0],
        'Y Axis': X_3d[:, 1],
        'Z Axis': X_3d[:, 2],
        'Hover Text': hover_texts,
        'Full Text': full_texts, # Hidden data used for clicking
        'Category': parent_texts
    })

    # Build the 3D Scatter Plot (Attaching Full Text to custom_data so the click event can read it)
    fig = px.scatter_3d(
        df, x='X Axis', y='Y Axis', z='Z Axis', color='Category',
        custom_data=['Full Text'], # Magic line that allows Dash to read the hidden full text
        hover_data={'X Axis': False, 'Y Axis': False, 'Z Axis': False, 'Category': False, 'Full Text': False, 'Hover Text': True},
        template="plotly_dark" 
    )

    fig.update_traces(marker=dict(size=5, opacity=0.8, line=dict(width=0)))
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0), # Removes borders
        scene=dict(
            xaxis=dict(showticklabels=False, title='', showgrid=False, zeroline=False),
            yaxis=dict(showticklabels=False, title='', showgrid=False, zeroline=False),
            zaxis=dict(showticklabels=False, title='', showgrid=False, zeroline=False)
        )
    )

    # ---------------------------------------------------------
    # LAUNCH DASH WEB APP
    # ---------------------------------------------------------
    app = Dash(__name__)
    
    # The UI Layout (Dark Mode Enterprise Styling)
    app.layout = html.Div([
        html.H2(f"Lytrex Vector Universe: {framework_name}", style={'margin': '0 0 20px 0', 'fontFamily': 'sans-serif', 'fontWeight': '300'}),
        
        # Container for Graph and Text Box
        html.Div([
            
            # Left Side: The 3D Graph
            html.Div([
                dcc.Graph(id='3d-scatter', figure=fig, style={'height': '80vh'})
            ], style={'width': '65%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            
            # Right Side: The Document Viewer
            html.Div([
                html.Div("SELECTED FRAMEWORK RULE", style={'fontSize': '12px', 'letterSpacing': '2px', 'color': '#888', 'marginBottom': '10px'}),
                html.Pre(id='click-output', children="Click any dot in the 3D space to read the full framework rule here...", 
                         style={
                             'whiteSpace': 'pre-wrap', 
                             'backgroundColor': '#1e1e1e', 
                             'color': '#00ffcc', # Cyperpunk Green/Blue text
                             'padding': '20px', 
                             'borderRadius': '8px',
                             'height': '75vh', 
                             'overflowY': 'auto',
                             'fontFamily': 'monospace',
                             'fontSize': '14px',
                             'border': '1px solid #333'
                         })
            ], style={'width': '33%', 'display': 'inline-block', 'marginLeft': '2%', 'verticalAlign': 'top'})
            
        ])
    ], style={'backgroundColor': '#0a0a0a', 'color': 'white', 'padding': '20px', 'height': '100vh', 'boxSizing': 'border-box'})

    # The Logic: What happens when you click a dot
    @app.callback(
        Output('click-output', 'children'),
        Input('3d-scatter', 'clickData')
    )
    def display_click_data(clickData):
        if clickData is None:
            return "Click any dot in the 3D space to read the full framework rule here..."
        
        # Extract the hidden Full Text from the dot the user clicked
        full_text = clickData['points'][0]['customdata'][0]
        return full_text

    print("\n" + "="*50)
    print(" Lytrex Server Running! Click the link below: ")
    print(" -> http://127.0.0.1:8050/ ")
    print("="*50 + "\n")
    
    app.run(debug=False)

if __name__ == "__main__":
    load_and_visualize_tsne_3d()