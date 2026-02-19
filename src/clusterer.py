import os
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from scipy.spatial.distance import cdist
from src.utils.config_loader import load_config

# 1. Load Configuration
config = load_config()
ANCHORS_CONFIG = config["anchors_config"]
GLOBAL_SUB_TOPICS = [sub for sub_list in ANCHORS_CONFIG.values() for sub in sub_list]

def steer_global_topics(embeddings_5d, anchor_5d, hdb_labels):
    """Maps HDBSCAN clusters AND Noise points (-1) to predefined Business Anchors."""
    final_labels = np.copy(hdb_labels)
    unique_labels = [l for l in np.unique(hdb_labels) if l != -1]
    
    if len(unique_labels) > 0:
        centroids = np.array([embeddings_5d[hdb_labels == l].mean(axis=0) for l in unique_labels])
        dists = cdist(centroids, anchor_5d, metric='cosine')
        mapping = {label: np.argmin(dists[i]) for i, label in enumerate(unique_labels)}
        
        for i in range(len(final_labels)):
            if final_labels[i] != -1:
                final_labels[i] = mapping[final_labels[i]]
                
    noise_indices = np.where(hdb_labels == -1)[0]
    if len(noise_indices) > 0:
        noise_embeddings = embeddings_5d[noise_indices]
        noise_dists = cdist(noise_embeddings, anchor_5d, metric='cosine')
        final_labels[noise_indices] = np.argmin(noise_dists, axis=1)
        
    return final_labels

def run_clustering_pipeline(pdf: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """FIT MODE: Executes the pipeline and SAVES models for production."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True) 
    
    model = SentenceTransformer('intfloat/multilingual-e5-base')
    sentences = ["query: " + str(s) for s in pdf[text_col]]
    embeddings = model.encode(sentences, show_progress_bar=True)

    reducer = umap.UMAP(n_neighbors=15, n_components=5, metric='cosine', random_state=42)
    embeddings_5d = reducer.fit_transform(embeddings)
    joblib.dump(reducer, os.path.join(models_dir, "umap_reducer.joblib"))

    clusterer = hdbscan.HDBSCAN(min_cluster_size=30, min_samples=1, cluster_selection_epsilon=0.1, prediction_data=True)
    hdb_labels = clusterer.fit_predict(embeddings_5d)
    joblib.dump(clusterer, os.path.join(models_dir, "hdbscan_clusterer.joblib"))

    anchor_text = ["query: " + s for s in GLOBAL_SUB_TOPICS]
    anchor_5d = reducer.transform(model.encode(anchor_text))

    pdf['sub_topic_id'] = steer_global_topics(embeddings_5d, anchor_5d, hdb_labels)
    
    topic_map = {i: name for i, name in enumerate(GLOBAL_SUB_TOPICS)}
    topic_map[-1] = "Outlier/Noise"
    pdf['sub_topic'] = pdf['sub_topic_id'].map(topic_map)
    
    return pdf

def predict_clustering_pipeline(pdf: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """INFERENCE MODE: Loads saved models to instantly predict sub-topics."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    
    try:
        reducer = joblib.load(os.path.join(models_dir, "umap_reducer.joblib"))
        clusterer = joblib.load(os.path.join(models_dir, "hdbscan_clusterer.joblib"))
    except FileNotFoundError:
        raise FileNotFoundError("🚨 Models not found! Run python fit_pipeline.py first.")

    model = SentenceTransformer('intfloat/multilingual-e5-base')
    sentences = ["query: " + str(s) for s in pdf[text_col]]
    embeddings = model.encode(sentences, show_progress_bar=False)

    embeddings_5d = reducer.transform(embeddings)
    hdb_labels, probabilities = hdbscan.approximate_predict(clusterer, embeddings_5d)

    anchor_text = ["query: " + s for s in GLOBAL_SUB_TOPICS]
    anchor_5d = reducer.transform(model.encode(anchor_text))

    pdf['sub_topic_id'] = steer_global_topics(embeddings_5d, anchor_5d, hdb_labels)
    
    topic_map = {i: name for i, name in enumerate(GLOBAL_SUB_TOPICS)}
    topic_map[-1] = "Outlier/Noise"
    pdf['sub_topic'] = pdf['sub_topic_id'].map(topic_map)
    
    return pdf