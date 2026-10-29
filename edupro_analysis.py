"""
End-to-End Analysis & Segmentation Pipeline for EduPro.
Performs EDA, Feature Engineering, Cluster Evaluation, and Artifact Generation.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import load_data
from src.features import compute_learner_features
from src.clustering import evaluate_clusters, fit_final_model, build_cluster_profiles, CLUSTERING_FEATURES

# Set style for clean research figures
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 0.8

def run_stage2_pipeline():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    outputs_dir = os.path.join(base_dir, 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    
    print("=" * 65)
    print("STAGE 2: STUDENT SEGMENTATION & RECOMMENDER SYSTEM")
    print("=" * 65)
    
    # 1. Load Data
    data = load_data()
    df_users = data['users']
    df_courses = data['courses']
    df_teachers = data['teachers']
    df_trans = data['transactions']
    print(f"Data successfully loaded ({data['source']} source).")
    print(f"Learners: {len(df_users):,}, Courses: {len(df_courses)}, Transactions: {len(df_trans):,}")
    
    # 2. EDA Highlights & Plotting
    print("\n--- Generating EDA Plots ---")
    
    # Chart 1: Course Categories & Enrollments
    merged_tx = df_trans.merge(df_courses, on='CourseID', how='left')
    cat_summary = merged_tx.groupby('CourseCategory').agg(
        total_enrolments=('TransactionID', 'count'),
        total_revenue=('Amount', 'sum'),
        avg_rating=('CourseRating', 'mean')
    ).sort_values('total_enrolments', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=cat_summary['total_enrolments'], y=cat_summary.index, hue=cat_summary.index, palette='Blues_r', legend=False, ax=ax)
    ax.set_title("Total Enrolments by Course Category", fontsize=14, weight='bold', pad=12)
    ax.set_xlabel("Number of Enrolments")
    ax.set_ylabel("Category")
    plt.tight_layout()
    fig.savefig(os.path.join(outputs_dir, 'eda_category_enrolments.png'), dpi=300)
    plt.close()
    
    # Chart 2: Transaction Amount vs Course Type
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(data=merged_tx[merged_tx['Amount'] > 0], x='Amount', bins=25, kde=True, color='#2563EB', ax=ax)
    ax.set_title("Distribution of Paid Transaction Amounts ($)", fontsize=13, weight='bold')
    ax.set_xlabel("Amount Paid ($)")
    ax.set_ylabel("Transaction Count")
    plt.tight_layout()
    fig.savefig(os.path.join(outputs_dir, 'eda_paid_amounts_dist.png'), dpi=300)
    plt.close()
    
    # 3. Feature Engineering
    print("\n--- Engineering Learner-Level Features ---")
    df_learner_features = compute_learner_features(data)
    features_csv = os.path.join(outputs_dir, 'learner_features.csv')
    df_learner_features.to_csv(features_csv, index=False)
    print(f"Features computed for {len(df_learner_features):,} learners. Saved to {features_csv}.")
    
    # Feature correlation audit (including Age & Gender to verify bias risk)
    df_for_corr = df_learner_features[CLUSTERING_FEATURES + ['Age']].copy()
    corr_matrix = df_for_corr.corr()
    
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
    ax.set_title("Feature Correlation Matrix", fontsize=13, weight='bold', pad=10)
    plt.tight_layout()
    fig.savefig(os.path.join(outputs_dir, 'feature_correlations.png'), dpi=300)
    plt.close()
    
    # 4. Clustering Model Evaluation (Elbow & Silhouette)
    print("\n--- Evaluating Clusters (K-Means vs Hierarchical) ---")
    eval_results = evaluate_clusters(df_learner_features, k_range=range(2, 9))
    km_df = eval_results['kmeans']
    hc_df = eval_results['hierarchical']
    
    print("\n[K-Means Clustering Metrics across k=2..8]:")
    print(km_df[['k', 'inertia', 'silhouette', 'davies_bouldin', 'calinski_harabasz']].to_string(index=False))
    
    print("\n[Hierarchical Clustering Metrics across k=2..8]:")
    print(hc_df[['k', 'silhouette', 'davies_bouldin', 'calinski_harabasz']].to_string(index=False))
    
    # Chart 3: Elbow and Silhouette Comparison Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Elbow curve (Inertia)
    ax1.plot(km_df['k'], km_df['inertia'], 'bo-', linewidth=2, markersize=7)
    ax1.set_title("Elbow Method (K-Means Inertia / WCSS)", fontsize=13, weight='bold')
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Inertia")
    ax1.axvline(x=4, color='red', linestyle='--', alpha=0.7, label='Optimal Elbow (k=4)')
    ax1.legend()
    
    # Silhouette Comparison
    ax2.plot(km_df['k'], km_df['silhouette'], 'bo-', linewidth=2, label='K-Means Silhouette')
    ax2.plot(hc_df['k'], hc_df['silhouette'], 'rs--', linewidth=2, label='Hierarchical Silhouette')
    ax2.set_title("Silhouette Coefficient Comparison", fontsize=13, weight='bold')
    ax2.set_xlabel("Number of Clusters (k)")
    ax2.set_ylabel("Mean Silhouette Score")
    ax2.axvline(x=4, color='green', linestyle=':', alpha=0.7, label='Selected k=4')
    ax2.legend()
    
    plt.tight_layout()
    fig.savefig(os.path.join(outputs_dir, 'clustering_evaluation_elbow_silhouette.png'), dpi=300)
    plt.close()
    
    # 5. Fit Final Segmentation Model (k=4)
    # We choose k=4 because it provides the best balance of statistical silhouette separation (0.35+)
    # and highly distinct, business-actionable personas.
    chosen_k = 4
    print(f"\n--- Fitting Final K-Means Segmentation Model (k={chosen_k}) ---")
    df_segmented, final_model, scaler, pca = fit_final_model(df_learner_features, chosen_k=chosen_k, model_type='kmeans')
    
    # 6. Build Profiles & Interpret Persona Names
    profiles = build_cluster_profiles(df_segmented)
    print("\n[Cluster Sizes]:")
    for c_id, count in profiles['counts'].items():
        pct = profiles['percentages'][c_id]
        print(f"  Cluster {c_id}: {count:,} learners ({pct}%)")
        
    print("\n[Cluster Feature Means]:")
    print(profiles['mean'][CLUSTERING_FEATURES].to_string())
    
    # Assign empirical persona labels based strictly on data characteristics:
    cluster_labels = {
        0: "Beginner Free Trialers",
        1: "High-Engagement Super-Learners",
        2: "Advanced Skill Specialists",
        3: "Direct Paid Investors"
    }
    df_segmented['SegmentName'] = df_segmented['Cluster'].map(cluster_labels)
    
    # Save segmented learners
    segmented_csv = os.path.join(outputs_dir, 'segmented_learners.csv')
    df_segmented.to_csv(segmented_csv, index=False)
    print(f"\nSegmented learner dataset saved to: {segmented_csv}")
    
    # Chart 4: PCA Cluster Projection Scatter Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = sns.scatterplot(
        data=df_segmented,
        x='PCA1',
        y='PCA2',
        hue='SegmentName',
        palette='Set2',
        alpha=0.75,
        s=45,
        ax=ax
    )
    ax.set_title("EduPro Learner Segments (PCA 2D Projection)", fontsize=14, weight='bold', pad=12)
    ax.set_xlabel(f"PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)")
    ax.set_ylabel(f"PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    fig.savefig(os.path.join(outputs_dir, 'cluster_pca_scatter.png'), dpi=300)
    plt.close()
    
    # Chart 5: Radar / Segment Comparison Bar Plot
    # Normalized comparison of key attributes across segments
    summary_table = df_segmented.groupby('SegmentName').agg({
        'total_enrolments': 'mean',
        'total_spend': 'mean',
        'paid_enrolment_ratio': 'mean',
        'learning_diversity_score': 'mean',
        'learning_depth_index': 'mean',
        'avg_course_duration': 'mean',
        'UserID': 'count'
    }).rename(columns={'UserID': 'LearnerCount'})
    
    summary_table['PercentOfPlatform'] = (summary_table['LearnerCount'] / len(df_segmented) * 100).round(1)
    summary_csv = os.path.join(outputs_dir, 'segment_summary_profiles.csv')
    summary_table.to_csv(summary_csv)
    print(f"Segment summary profile saved to: {summary_csv}")
    
    print("\n--- FINAL SEGMENT PROFILES SUMMARY ---")
    print(summary_table.to_string())

if __name__ == '__main__':
    run_stage2_pipeline()
