"""
Feature engineering module for EduPro learner segmentation.
Computes mathematically sound behavioral features per UserID.
"""

import pandas as pd
import numpy as np

def compute_learner_features(data_dict):
    """
    Computes learner-level behavioral features from transactions and course metadata.
    Args:
        data_dict: dict with 'users', 'courses', 'transactions', etc.
    Returns:
        pd.DataFrame indexed by UserID with all engineered features.
    """
    df_users = data_dict['users'].copy()
    df_courses = data_dict['courses'].copy()
    df_trans = data_dict['transactions'].copy()
    
    # Merge transactions with course metadata
    merged = df_trans.merge(
        df_courses[['CourseID', 'CourseCategory', 'CourseType', 'CourseLevel', 'CoursePrice', 'CourseDuration', 'CourseRating']],
        on='CourseID',
        how='left'
    )
    
    # Map CourseLevel to numeric depth score: Beginner=1, Intermediate=2, Advanced=3
    level_weights = {'Beginner': 1, 'Intermediate': 2, 'Advanced': 3}
    merged['level_weight'] = merged['CourseLevel'].map(level_weights).fillna(1)
    merged['is_paid'] = (merged['CourseType'] == 'Paid').astype(int)
    
    # Total unique categories on platform
    total_platform_categories = df_courses['CourseCategory'].nunique()
    
    # Group by UserID
    records = []
    
    for uid, group in merged.groupby('UserID'):
        n_enrolments = len(group)
        tot_spend = group['Amount'].sum()
        avg_spend = tot_spend / n_enrolments if n_enrolments > 0 else 0.0
        paid_ratio = group['is_paid'].mean()
        
        # Category exploration
        cat_counts = group['CourseCategory'].value_counts()
        n_categories = len(cat_counts)
        pref_category = cat_counts.index[0]
        
        # Diversity score: Shannon entropy normalized by ln(total_platform_categories)
        probs = cat_counts / n_enrolments
        entropy = -np.sum(probs * np.log(probs + 1e-12))
        max_possible_entropy = np.log(total_platform_categories)
        diversity_score = float(np.clip(entropy / max_possible_entropy, 0.0, 1.0))
        
        # Learning depth index: weighted level mean
        depth_index = float(group['level_weight'].mean())
        pref_level = group['CourseLevel'].mode()[0]
        
        # Average catalog course rating & duration chosen by learner
        avg_course_rating = float(group['CourseRating'].mean())
        avg_duration = float(group['CourseDuration'].mean())
        
        # Enrolment timespan in days
        min_date = group['TransactionDate'].min()
        max_date = group['TransactionDate'].max()
        active_days = max(1, (max_date - min_date).days + 1)
        # Monthly frequency: enrollments per 30-day period
        enrolment_frequency = (n_enrolments / active_days) * 30.0
        
        pref_payment = group['PaymentMethod'].mode()[0]
        
        records.append({
            'UserID': uid,
            'total_enrolments': n_enrolments,
            'total_spend': round(tot_spend, 2),
            'avg_spend': round(avg_spend, 2),
            'paid_enrolment_ratio': round(paid_ratio, 4),
            'categories_explored_count': n_categories,
            'preferred_category': pref_category,
            'preferred_level': pref_level,
            'learning_diversity_score': round(diversity_score, 4),
            'learning_depth_index': round(depth_index, 4),
            'avg_catalog_course_rating': round(avg_course_rating, 2),
            'avg_course_duration': round(avg_duration, 2),
            'active_days_span': active_days,
            'enrolment_frequency_monthly': round(enrolment_frequency, 2),
            'preferred_payment_method': pref_payment
        })
        
    df_features = pd.DataFrame(records)
    
    # Merge demographics (Age, Gender) for descriptive profile reporting (NOT for recommendation bias)
    df_final = df_features.merge(df_users[['UserID', 'Age', 'Gender']], on='UserID', how='left')
    return df_final
