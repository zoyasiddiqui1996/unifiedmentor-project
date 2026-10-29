"""
Clustering and Learner Segmentation Module for EduPro.
Performs scaling, Elbow analysis, Silhouette comparison (K-Means vs Hierarchical),
and segment profiling.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA

CLUSTERING_FEATURES = [
    'total_enrolments',
    'total_spend',
    'avg_spend',
    'paid_enrolment_ratio',
    'categories_explored_count',
    'learning_diversity_score',
    'learning_depth_index',
    'avg_course_duration'
]

def evaluate_clusters(df_features, k_range=range(2, 9)):
    """
    Evaluates K-Means and Agglomerative Hierarchical clustering across k_range.
    Returns:
        dict containing metrics, scaled_features, and models.
    """
    X = df_features[CLUSTERING_FEATURES].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans_results = []
    hierarchical_results = []
    
    for k in k_range:
        # K-Means
        km = KMeans(n_clusters=k, random_state=42, n_init=15)
        km_labels = km.fit_predict(X_scaled)
        km_sil = silhouette_score(X_scaled, km_labels)
        km_db = davies_bouldin_score(X_scaled, km_labels)
        km_ch = calinski_harabasz_score(X_scaled, km_labels)
        kmeans_results.append({
            'k': k,
            'inertia': km.inertia_,
            'silhouette': km_sil,
            'davies_bouldin': km_db,
            'calinski_harabasz': km_ch,
            'model': km,
            'labels': km_labels
        })
        
        # Hierarchical (Agglomerative with Ward linkage)
        hc = AgglomerativeClustering(n_clusters=k, linkage='ward')
        hc_labels = hc.fit_predict(X_scaled)
        hc_sil = silhouette_score(X_scaled, hc_labels)
        hc_db = davies_bouldin_score(X_scaled, hc_labels)
        hc_ch = calinski_harabasz_score(X_scaled, hc_labels)
        hierarchical_results.append({
            'k': k,
            'silhouette': hc_sil,
            'davies_bouldin': hc_db,
            'calinski_harabasz': hc_ch,
            'model': hc,
            'labels': hc_labels
        })
        
    return {
        'scaler': scaler,
        'X_scaled': X_scaled,
        'features_used': CLUSTERING_FEATURES,
        'kmeans': pd.DataFrame(kmeans_results),
        'hierarchical': pd.DataFrame(hierarchical_results)
    }

def fit_final_model(df_features, chosen_k=4, model_type='kmeans'):
    """
    Fits final segmentation model and assigns clusters and interpretable labels.
    """
    X = df_features[CLUSTERING_FEATURES].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if model_type == 'kmeans':
        model = KMeans(n_clusters=chosen_k, random_state=42, n_init=20)
        labels = model.fit_predict(X_scaled)
    else:
        model = AgglomerativeClustering(n_clusters=chosen_k, linkage='ward')
        labels = model.fit_predict(X_scaled)
        
    df_segmented = df_features.copy()
    df_segmented['Cluster'] = labels
    
    # 2D PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_scaled)
    df_segmented['PCA1'] = pca_coords[:, 0]
    df_segmented['PCA2'] = pca_coords[:, 1]
    
    return df_segmented, model, scaler, pca

def build_cluster_profiles(df_segmented):
    """
    Computes statistical profiles (mean & median) for each segment.
    """
    numeric_cols = CLUSTERING_FEATURES + ['Age']
    profile_mean = df_segmented.groupby('Cluster')[numeric_cols].mean()
    profile_median = df_segmented.groupby('Cluster')[numeric_cols].median()
    cluster_counts = df_segmented['Cluster'].value_counts().sort_index()
    
    return {
        'counts': cluster_counts,
        'percentages': (cluster_counts / len(df_segmented) * 100).round(2),
        'mean': profile_mean.round(2),
        'median': profile_median.round(2)
    }
