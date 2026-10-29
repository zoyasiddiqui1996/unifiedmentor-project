"""
Hybrid Recommendation Engine for EduPro.
Combines Segment Popularity, Content Match, Similar-Learner Collaborative Filtering, and Course Ratings.
Provides plain-English explanations and enforces exclusion of previously enrolled courses.
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

class EduProRecommender:
    def __init__(self, data_dict, df_segmented, weights=None):
        """
        Initializes the recommender with dataset and cluster assignments.
        weights: dict with keys 'segment_popularity', 'content_match', 'similar_learners', 'course_rating'.
                 Default: 0.35, 0.25, 0.20, 0.20.
        """
        self.users = data_dict['users'].copy()
        self.courses = data_dict['courses'].copy()
        self.transactions = data_dict['transactions'].copy()
        self.segmented = df_segmented.copy()
        
        # Merge course details into transactions
        self.tx_merged = self.transactions.merge(
            self.courses[['CourseID', 'CourseCategory', 'CourseLevel', 'CourseType', 'CourseRating', 'CoursePrice']],
            on='CourseID',
            how='left'
        )
        
        # User history map
        self.user_history = self.transactions.groupby('UserID')['CourseID'].apply(set).to_dict()
        
        # Recommendation weights
        if weights is None:
            self.weights = {
                'segment_popularity': 0.35,
                'content_match': 0.25,
                'similar_learners': 0.20,
                'course_rating': 0.20
            }
        else:
            self.weights = weights
            
        # Precompute segment popularity scores: dict[cluster_id, Series[CourseID -> normalized score]]
        self._precompute_segment_popularity()
        
        # Precompute nearest neighbors model on scaled behavioral features
        self._build_knn_model()
        
        # Precompute course quality scores
        self._precompute_course_quality()

    def _precompute_segment_popularity(self):
        """Precomputes course enrollment frequency per segment, normalized to [0, 1]."""
        self.segment_pop = {}
        merged_user_cluster = self.tx_merged.merge(self.segmented[['UserID', 'Cluster']], on='UserID', how='left')
        
        all_courses = self.courses['CourseID'].unique()
        
        for c_id in self.segmented['Cluster'].unique():
            cluster_tx = merged_user_cluster[merged_user_cluster['Cluster'] == c_id]
            counts = cluster_tx['CourseID'].value_counts()
            # Reindex to all courses
            full_counts = pd.Series(0.0, index=all_courses)
            full_counts.update(counts.astype(float))
            max_c = full_counts.max()
            self.segment_pop[c_id] = (full_counts / max_c) if max_c > 0 else full_counts

    def _precompute_course_quality(self):
        """Precomputes normalized course rating score in [0, 1]."""
        min_r = self.courses['CourseRating'].min()
        max_r = self.courses['CourseRating'].max()
        if max_r > min_r:
            self.course_quality = ((self.courses.set_index('CourseID')['CourseRating'] - min_r) / (max_r - min_r)).to_dict()
        else:
            self.course_quality = {cid: 1.0 for cid in self.courses['CourseID']}

    def _build_knn_model(self):
        """Fits k-NN model on behavioral features for collaborative filtering."""
        from src.clustering import CLUSTERING_FEATURES
        from sklearn.preprocessing import StandardScaler
        
        self.cf_user_ids = self.segmented['UserID'].values
        X = self.segmented[CLUSTERING_FEATURES].values
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.knn = NearestNeighbors(n_neighbors=min(25, len(self.segmented)), metric='cosine')
        self.knn.fit(X_scaled)
        self.user_to_idx = {uid: idx for idx, uid in enumerate(self.cf_user_ids)}
        self.X_scaled = X_scaled

    def _get_content_match_score(self, learner_row, course_row):
        """Calculates content-based match score [0, 1] based on category and level alignment."""
        score = 0.0
        # Category match (60% of content score)
        if course_row['CourseCategory'] == learner_row['preferred_category']:
            score += 0.60
            
        # Level progression match (40% of content score)
        # If user preferred Beginner, Intermediate is natural progression (+0.4)
        pref_lvl = learner_row['preferred_level']
        course_lvl = course_row['CourseLevel']
        
        if pref_lvl == course_lvl:
            score += 0.40
        elif pref_lvl == 'Beginner' and course_lvl == 'Intermediate':
            score += 0.35
        elif pref_lvl == 'Intermediate' and course_lvl == 'Advanced':
            score += 0.35
        elif pref_lvl == 'Advanced' and course_lvl == 'Intermediate':
            score += 0.25
        else:
            score += 0.10
            
        return score

    def _get_similar_learner_score(self, user_id):
        """Returns course frequency distribution among nearest neighbor learners."""
        if user_id not in self.user_to_idx:
            return pd.Series(0.0, index=self.courses['CourseID'].unique())
            
        idx = self.user_to_idx[user_id]
        distances, indices = self.knn.kneighbors([self.X_scaled[idx]])
        neighbor_uids = [self.cf_user_ids[i] for i in indices[0][1:]] # exclude self
        
        neighbor_courses = []
        for n_uid in neighbor_uids:
            neighbor_courses.extend(list(self.user_history.get(n_uid, [])))
            
        all_courses = self.courses['CourseID'].unique()
        counts = pd.Series(0.0, index=all_courses)
        if neighbor_courses:
            vc = pd.Series(neighbor_courses).value_counts().astype(float)
            counts.update(vc)
            max_val = counts.max()
            if max_val > 0:
                counts = counts / max_val
        return counts

    def recommend(self, user_id, top_n=5, category_filter=None, level_filter=None):
        """
        Generates top_n course recommendations with plain-English rationales.
        Enforces strict exclusion of already enrolled courses.
        """
        user_rows = self.segmented[self.segmented['UserID'] == user_id]
        if user_rows.empty:
            # Cold-start fallback: Top platform courses
            return self._cold_start_fallback(top_n, category_filter, level_filter)
            
        learner = user_rows.iloc[0]
        cluster_id = learner['Cluster']
        segment_name = learner['SegmentName']
        enrolled_courses = self.user_history.get(user_id, set())
        
        # Precomputed scores
        seg_scores = self.segment_pop.get(cluster_id, pd.Series(0.0, index=self.courses['CourseID']))
        sim_scores = self._get_similar_learner_score(user_id)
        
        scored_courses = []
        
        for _, course in self.courses.iterrows():
            cid = course['CourseID']
            # STRICT EXCLUSION: Skip already enrolled courses
            if cid in enrolled_courses:
                continue
                
            # Optional Category Filter
            if category_filter and course['CourseCategory'] != category_filter:
                continue
                
            # Optional Level Filter
            if level_filter and course['CourseLevel'] != level_filter:
                continue
                
            # 1. Segment Popularity
            s_pop = float(seg_scores.get(cid, 0.0))
            # 2. Content Match
            c_match = self._get_content_match_score(learner, course)
            # 3. Similar Learners
            s_sim = float(sim_scores.get(cid, 0.0))
            # 4. Course Rating Quality
            c_qual = float(self.course_quality.get(cid, 0.5))
            
            # Weighted hybrid score
            final_score = (
                self.weights['segment_popularity'] * s_pop +
                self.weights['content_match'] * c_match +
                self.weights['similar_learners'] * s_sim +
                self.weights['course_rating'] * c_qual
            )
            
            # Plain-English explanation generator
            reasons = []
            if course['CourseCategory'] == learner['preferred_category']:
                reasons.append(f"Matches your primary interest in {course['CourseCategory']}")
            if s_pop >= 0.70:
                reasons.append(f"Trending among {segment_name}")
            if s_sim >= 0.50:
                reasons.append("Frequently chosen by learners with similar study habits")
            if course['CourseRating'] >= 4.5:
                reasons.append(f"Top-rated catalog course ({course['CourseRating']:.2f}★)")
            if not reasons:
                reasons.append(f"High-quality {course['CourseLevel']} course in {course['CourseCategory']}")
                
            scored_courses.append({
                'CourseID': cid,
                'CourseName': course['CourseName'],
                'CourseCategory': course['CourseCategory'],
                'CourseLevel': course['CourseLevel'],
                'CourseType': course['CourseType'],
                'CoursePrice': course['CoursePrice'],
                'CourseRating': course['CourseRating'],
                'MatchScore': round(final_score * 100, 1),
                'RecommendationReason': " | ".join(reasons)
            })
            
        df_recs = pd.DataFrame(scored_courses)
        if df_recs.empty:
            return pd.DataFrame()
            
        return df_recs.sort_values('MatchScore', ascending=False).head(top_n).reset_index(drop=True)

    def _cold_start_fallback(self, top_n=5, category_filter=None, level_filter=None):
        """Fallback for users without any history: returns top-rated courses."""
        filtered = self.courses.copy()
        if category_filter:
            filtered = filtered[filtered['CourseCategory'] == category_filter]
        if level_filter:
            filtered = filtered[filtered['CourseLevel'] == level_filter]
            
        filtered['MatchScore'] = (filtered['CourseRating'] / 5.0 * 100).round(1)
        filtered['RecommendationReason'] = "Platform recommendation based on high catalog rating."
        cols = ['CourseID', 'CourseName', 'CourseCategory', 'CourseLevel', 'CourseType', 'CoursePrice', 'CourseRating', 'MatchScore', 'RecommendationReason']
        return filtered[cols].sort_values('MatchScore', ascending=False).head(top_n).reset_index(drop=True)
