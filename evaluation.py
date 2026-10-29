"""
Offline Evaluation Module for EduPro Recommendation Engine.
Implements chronological leave-one-out train/test split, calculates ranking metrics
(HitRate@5, Precision@5, MRR@5, NDCG@5) and compares against a Popularity Baseline.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.features import compute_learner_features
from src.clustering import fit_final_model
from src.recommender import EduProRecommender

def evaluate_recommender(data_dict, chosen_k=4, weights=None):
    """
    Performs chronological leave-one-out evaluation.
    Returns:
        dict containing metrics for Hybrid Model and Popularity Baseline.
    """
    df_trans = data_dict['transactions'].sort_values('TransactionDate').copy()
    
    # Filter users with >= 2 transactions
    user_tx_counts = df_trans.groupby('UserID').size()
    eval_users = user_tx_counts[user_tx_counts >= 2].index.tolist()
    
    train_records = []
    test_dict = {} # UserID -> held_out CourseID
    
    for uid, group in df_trans.groupby('UserID', sort=False):
        if uid in eval_users:
            # Last transaction is test
            train_part = group.iloc[:-1]
            test_row = group.iloc[-1]
            train_records.append(train_part)
            test_dict[uid] = test_row['CourseID']
        else:
            # Single transaction users stay in train (or cold-start context)
            train_records.append(group)
            
    df_train_trans = pd.concat(train_records, ignore_index=True)
    
    train_data_dict = {
        'users': data_dict['users'],
        'teachers': data_dict['teachers'],
        'courses': data_dict['courses'],
        'transactions': df_train_trans
    }
    
    # Train features and clustering on TRAIN DATA ONLY
    train_features = compute_learner_features(train_data_dict)
    df_segmented_train, _, _, _ = fit_final_model(train_features, chosen_k=chosen_k, model_type='kmeans')
    
    cluster_labels = {
        0: "Beginner Free Trialers",
        1: "High-Engagement Super-Learners",
        2: "Advanced Skill Specialists",
        3: "Direct Paid Investors"
    }
    df_segmented_train['SegmentName'] = df_segmented_train['Cluster'].map(cluster_labels)
    
    # Initialize Recommender on train set
    recommender = EduProRecommender(train_data_dict, df_segmented_train, weights=weights)
    
    # Global popularity ranking on train set (Baseline)
    global_popular_courses = df_train_trans['CourseID'].value_counts().index.tolist()
    train_history = df_train_trans.groupby('UserID')['CourseID'].apply(set).to_dict()
    
    # Metrics accumulator
    results = {
        'hybrid': {'hits': 0, 'precision': [], 'mrr': [], 'ndcg': []},
        'popularity': {'hits': 0, 'precision': [], 'mrr': [], 'ndcg': []}
    }
    
    n_test_users = len(test_dict)
    
    for uid, true_cid in test_dict.items():
        user_seen = train_history.get(uid, set())
        
        # 1. Hybrid Model Top 5
        recs_df = recommender.recommend(uid, top_n=5)
        hybrid_top5 = recs_df['CourseID'].tolist() if not recs_df.empty else []
        
        # 2. Baseline Model Top 5 (most popular courses unseen by user)
        pop_top5 = [cid for cid in global_popular_courses if cid not in user_seen][:5]
        
        # Evaluate Hybrid
        if true_cid in hybrid_top5:
            rank = hybrid_top5.index(true_cid) + 1
            results['hybrid']['hits'] += 1
            results['hybrid']['precision'].append(1.0 / 5.0)
            results['hybrid']['mrr'].append(1.0 / rank)
            results['hybrid']['ndcg'].append(1.0 / np.log2(rank + 1))
        else:
            results['hybrid']['precision'].append(0.0)
            results['hybrid']['mrr'].append(0.0)
            results['hybrid']['ndcg'].append(0.0)
            
        # Evaluate Baseline
        if true_cid in pop_top5:
            rank = pop_top5.index(true_cid) + 1
            results['popularity']['hits'] += 1
            results['popularity']['precision'].append(1.0 / 5.0)
            results['popularity']['mrr'].append(1.0 / rank)
            results['popularity']['ndcg'].append(1.0 / np.log2(rank + 1))
        else:
            results['popularity']['precision'].append(0.0)
            results['popularity']['mrr'].append(0.0)
            results['popularity']['ndcg'].append(0.0)
            
    summary = {
        'Metric': ['Hit Rate@5', 'Precision@5', 'MRR@5', 'NDCG@5', 'Evaluated Users'],
        'Hybrid Model': [
            round(results['hybrid']['hits'] / n_test_users, 4),
            round(np.mean(results['hybrid']['precision']), 4),
            round(np.mean(results['hybrid']['mrr']), 4),
            round(np.mean(results['hybrid']['ndcg']), 4),
            n_test_users
        ],
        'Popularity Baseline': [
            round(results['popularity']['hits'] / n_test_users, 4),
            round(np.mean(results['popularity']['precision']), 4),
            round(np.mean(results['popularity']['mrr']), 4),
            round(np.mean(results['popularity']['ndcg']), 4),
            n_test_users
        ]
    }
    
    df_eval = pd.DataFrame(summary)
    # Relative lift calculation
    lifts = []
    for h, p in zip(df_eval['Hybrid Model'][:4], df_eval['Popularity Baseline'][:4]):
        lift = ((h - p) / p * 100) if p > 0 else 0.0
        lifts.append(f"+{lift:.1f}%" if lift >= 0 else f"{lift:.1f}%")
    lifts.append("N/A")
    df_eval['Relative Lift'] = lifts
    
    return df_eval

def run_evaluation_pipeline():
    from src.data_loader import load_data
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outputs_dir = os.path.join(base_dir, 'outputs')
    
    print("=" * 65)
    print("STAGE 3: OFFLINE EVALUATION OF RECOMMENDATION ENGINE")
    print("=" * 65)
    
    data = load_data()
    df_eval = evaluate_recommender(data)
    
    print("\n--- OFFLINE RECOMMENDATION EVALUATION RESULTS ---")
    print(df_eval.to_string(index=False))
    
    eval_csv = os.path.join(outputs_dir, 'model_evaluation_metrics.csv')
    df_eval.to_csv(eval_csv, index=False)
    print(f"\nEvaluation metrics saved to: {eval_csv}")
    
    # Plotting comparison
    metrics_subset = df_eval[df_eval['Metric'] != 'Evaluated Users'].copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(metrics_subset))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, metrics_subset['Hybrid Model'], width, label='EduPro Hybrid Engine', color='#2563EB')
    rects2 = ax.bar(x + width/2, metrics_subset['Popularity Baseline'], width, label='Popularity Baseline', color='#94A3B8')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Recommendation Engine Offline Performance vs Popularity Baseline', fontsize=13, weight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_subset['Metric'], fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, max(metrics_subset['Hybrid Model'].max(), metrics_subset['Popularity Baseline'].max()) * 1.25)
    
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10, weight='bold')
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
                    
    plt.tight_layout()
    chart_path = os.path.join(outputs_dir, 'recommender_evaluation.png')
    fig.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Comparison chart saved to: {chart_path}")

if __name__ == '__main__':
    run_evaluation_pipeline()
