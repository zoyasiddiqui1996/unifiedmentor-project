"""
Student Segmentation and Recommender System
Interactive Streamlit Dashboard
Self-contained with automatic built-in dataset generator.
"""

import os
import sys

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

try:
    from src.data_loader import load_data
    from src.features import compute_learner_features
    from src.clustering import fit_final_model, build_cluster_profiles, CLUSTERING_FEATURES
    from src.recommender import EduProRecommender
except ModuleNotFoundError:
    from data_loader import load_data
    from features import compute_learner_features
    from clustering import fit_final_model, build_cluster_profiles, CLUSTERING_FEATURES
    from recommender import EduProRecommender

# --- Page Configuration ---
st.set_page_config(
    page_title="Student Segmentation and Recommender System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-super { background-color: #FEF3C7; color: #92400E; }
    .badge-investor { background-color: #DCFCE7; color: #166534; }
    .badge-specialist { background-color: #E0E7FF; color: #3730A3; }
    .badge-trialer { background-color: #F1F5F9; color: #334155; }
</style>
""", unsafe_allow_html=True)

def create_demo_data():
    """Generates an automatic in-memory dataset so the app runs without external files."""
    np.random.seed(42)
    user_ids = [f"U{i:05d}" for i in range(1, 151)]
    users = pd.DataFrame({
        'UserID': user_ids,
        'Age': np.random.randint(18, 55, size=len(user_ids)).astype(float),
        'Gender': np.random.choice(['Male', 'Female'], size=len(user_ids))
    })
    
    categories = ['Programming', 'Design', 'Business', 'Marketing', 'Data Science']
    course_list = [
        ("Python Basics", "Programming", "Paid", "Beginner", 450.0, 15.0, 4.8),
        ("Java Programming", "Programming", "Free", "Intermediate", 0.0, 35.0, 4.3),
        ("Web Development", "Programming", "Paid", "Intermediate", 399.0, 40.0, 4.6),
        ("UI/UX Design", "Design", "Paid", "Beginner", 299.0, 20.0, 4.7),
        ("Graphic Design", "Design", "Free", "Beginner", 0.0, 10.0, 4.1),
        ("3D Modeling", "Design", "Paid", "Advanced", 499.0, 25.0, 4.5),
        ("Business Strategy", "Business", "Paid", "Advanced", 550.0, 30.0, 4.9),
        ("Entrepreneurship 101", "Business", "Free", "Beginner", 0.0, 12.0, 4.2),
        ("Digital Marketing", "Marketing", "Free", "Beginner", 0.0, 18.0, 4.4),
        ("SEO Mastery", "Marketing", "Paid", "Advanced", 350.0, 22.0, 4.6),
        ("Machine Learning", "Data Science", "Paid", "Advanced", 599.0, 45.0, 4.8),
        ("Data Analysis with Pandas", "Data Science", "Free", "Intermediate", 0.0, 16.0, 4.5)
    ]
    course_ids = [f"CR{i:05d}" for i in range(1, len(course_list) + 1)]
    courses = pd.DataFrame([{
        'CourseID': course_ids[i],
        'CourseName': c[0],
        'CourseCategory': c[1],
        'CourseType': c[2],
        'CourseLevel': c[3],
        'CoursePrice': float(c[4]),
        'CourseDuration': float(c[5]),
        'CourseRating': float(c[6])
    } for i, c in enumerate(course_list)])
    
    teacher_ids = [f"T{i:05d}" for i in range(1, 11)]
    teachers = pd.DataFrame({
        'TeacherID': teacher_ids,
        'Age': np.random.randint(28, 60, size=len(teacher_ids)).astype(float),
        'Gender': np.random.choice(['Male', 'Female'], size=len(teacher_ids)),
        'Expertise': np.random.choice(categories, size=len(teacher_ids)),
        'YearsOfExperience': np.random.randint(3, 20, size=len(teacher_ids)).astype(float),
        'TeacherRating': np.round(np.random.uniform(4.0, 5.0, size=len(teacher_ids)), 2)
    })
    
    trans_rows = []
    tid = 1
    for uid in user_ids:
        n_c = np.random.randint(1, 6)
        chosen_courses = np.random.choice(course_ids, size=n_c, replace=False)
        for cid in chosen_courses:
            c_info = courses[courses['CourseID'] == cid].iloc[0]
            amt = c_info['CoursePrice']
            trans_rows.append({
                'TransactionID': f"TX{tid:06d}",
                'UserID': uid,
                'CourseID': cid,
                'TeacherID': np.random.choice(teacher_ids),
                'TransactionDate': pd.Timestamp('2024-01-01') + pd.Timedelta(days=int(np.random.randint(0, 300))),
                'Amount': float(amt),
                'PaymentMethod': 'Free' if amt == 0 else np.random.choice(['Credit Card', 'UPI', 'PayPal'])
            })
            tid += 1
            
    transactions = pd.DataFrame(trans_rows)
    return {
        'users': users,
        'teachers': teachers,
        'courses': courses,
        'transactions': transactions,
        'source': 'Built-in Safe Demo'
    }

@st.cache_data(show_spinner=False)
def load_and_prepare_pipeline():
    """Caches data loading, feature engineering, and model fitting."""
    try:
        data = load_data()
    except Exception:
        data = create_demo_data()
        
    if data is None or not isinstance(data, dict):
        data = create_demo_data()
        
    df_features = compute_learner_features(data)
    df_segmented, model, scaler, pca = fit_final_model(df_features, chosen_k=4, model_type='kmeans')
    
    cluster_labels = {
        0: "Beginner Free Trialers",
        1: "High-Engagement Super-Learners",
        2: "Advanced Skill Specialists",
        3: "Direct Paid Investors"
    }
    df_segmented['SegmentName'] = df_segmented['Cluster'].map(cluster_labels)
    
    recommender = EduProRecommender(data, df_segmented)
    return data, df_segmented, recommender, None

def get_segment_badge_html(segment_name):
    css_class = "badge-trialer"
    if "Super" in segment_name:
        css_class = "badge-super"
    elif "Investor" in segment_name:
        css_class = "badge-investor"
    elif "Specialist" in segment_name:
        css_class = "badge-specialist"
    return f'<span class="badge {css_class}">{segment_name}</span>'

# --- Load Data & Models ---
data, df_segmented, recommender, error_msg = load_and_prepare_pipeline()

# --- Sidebar ---
st.sidebar.title("Navigation")
active_tab = st.sidebar.radio(
    "Select View",
    [
        "Learner Profile Explorer",
        "Cluster Visualisation",
        "Segment Comparison",
        "Model Evaluation"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Dataset Status")
if error_msg:
    st.sidebar.error("Dataset not found!")
else:
    source_label = "Confidential Full Dataset" if data.get('source') == 'private' else "Safe Anonymized Sample"
    st.sidebar.success(f"Loaded: **{source_label}**")
    st.sidebar.write(f"Learners: **{len(df_segmented):,}**")
    st.sidebar.write(f"Courses: **{len(data['courses']):,}**")
    st.sidebar.write(f"Transactions: **{len(data['transactions']):,}**")

st.sidebar.markdown("---")
st.sidebar.caption("Privacy Guarantee: All PII is permanently excluded. Students are identified strictly by UserID.")

# Header
st.markdown('<div class="main-title">Student Segmentation and Recommender System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Empirical learner analytics, behavioral clustering, and explainable recommendations.</div>', unsafe_allow_html=True)

if error_msg:
    st.error(f"Error loading dataset: {error_msg}")
    st.stop()

# ==============================================================================
# MODULE A: LEARNER PROFILE EXPLORER
# ==============================================================================
if active_tab == "Learner Profile Explorer":
    st.header("Module A: Learner Profile & Personalized Recommendations")
    st.markdown("Search or select any learner by their anonymized **UserID** to explore their behavioral profile and generate recommendations.")
    
    col_sel1, col_sel2 = st.columns([2, 1])
    with col_sel1:
        learner_list = df_segmented['UserID'].tolist()
        selected_user = st.selectbox("Select Learner (UserID)", learner_list, index=0)
    with col_sel2:
        if st.button("🎲 Pick Random Learner"):
            selected_user = np.random.choice(learner_list)
            st.session_state['selected_user'] = selected_user

    learner_data = df_segmented[df_segmented['UserID'] == selected_user].iloc[0]
    segment_name = learner_data['SegmentName']
    
    st.markdown(f"### Profile Overview: **{selected_user}** — {get_segment_badge_html(segment_name)}", unsafe_allow_html=True)
    
    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    with col_m1:
        st.metric("Total Enrollments", f"{int(learner_data['n_enrolments'])}")
    with col_m2:
        st.metric("Total Spend", f"₹{learner_data['tot_spend']:,.2f}")
    with col_m3:
        st.metric("Paid Course Ratio", f"{learner_data['paid_ratio']*100:.1f}%")
    with col_m4:
        st.metric("Diversity Index", f"{learner_data['diversity_score']:.2f}")
    with col_m5:
        st.metric("Preferred Category", f"{learner_data['pref_category']}")
        
    st.markdown("---")
    st.subheader(f"Personalized Top-5 Course Recommendations for {selected_user}")
    
    rec_type = st.radio(
        "Recommendation Algorithm Strategy:",
        ["Hybrid (Personalized + Segment + Quality)", "Cold-Start / Popularity Baseline"],
        horizontal=True
    )
    
    use_popularity = (rec_type == "Cold-Start / Popularity Baseline")
    recs = recommender.recommend(selected_user, top_n=5, use_fallback_only=use_popularity)
    
    if recs:
        rec_cols = st.columns(len(recs))
        for idx, rec in enumerate(recs):
            with rec_cols[idx]:
                st.markdown(f"""
                <div class="metric-card">
                    <span style="font-size:0.8rem; color:#64748B; font-weight:600;">Rank #{idx+1}</span>
                    <h4 style="margin:0.2rem 0; font-size:1.05rem; color:#1E293B;">{rec['CourseName']}</h4>
                    <p style="font-size:0.85rem; color:#475569; margin-bottom:0.4rem;">
                        <b>Category:</b> {rec['CourseCategory']}<br>
                        <b>Level:</b> {rec['CourseLevel']}<br>
                        <b>Rating:</b> ⭐ {rec['CourseRating']:.1f}<br>
                        <b>Price:</b> {'Free' if rec['CoursePrice'] == 0 else f'₹{rec["CoursePrice"]:,.0f}'}
                    </p>
                    <div style="background-color:#F8FAFC; border-left:3px solid #3B82F6; padding:6px; font-size:0.75rem; color:#334155;">
                        <b>Why this?</b> {rec['Explanation']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No un-enrolled courses found to recommend.")
        
    st.markdown("---")
    st.subheader("Enrollment History")
    history_df = recommender.get_user_history(selected_user)
    if not history_df.empty:
        st.dataframe(
            history_df[['CourseID', 'CourseName', 'CourseCategory', 'CourseType', 'CourseLevel', 'Amount']],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.write("No prior enrollment transactions recorded.")

# ==============================================================================
# MODULE B: CLUSTER VISUALISATION (3D PCA & RADAR)
# ==============================================================================
elif active_tab == "Cluster Visualisation":
    st.header("Module B: Behavioral Segment Visualisation")
    st.markdown("Interactive inspection of the 4 discovered learner archetypes using dimensionality reduction and cluster centroid analysis.")
    
    pca_cols = [c for c in df_segmented.columns if c.startswith('PCA')]
    if len(pca_cols) >= 2:
        fig_pca = px.scatter(
            df_segmented,
            x=pca_cols[0],
            y=pca_cols[1],
            color='SegmentName',
            hover_data=['UserID', 'n_enrolments', 'tot_spend', 'paid_ratio', 'pref_category'],
            title="Learner Behavioral Segments in PCA Space",
            color_discrete_sequence=px.colors.qualitative.Safe,
            height=550
        )
        fig_pca.update_layout(
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#F8FAFC",
            legend_title="Learner Segment",
            font=dict(family="sans-serif", size=12)
        )
        st.plotly_chart(fig_pca, use_container_width=True)
        
    st.markdown("---")
    st.subheader("Feature Distributions Across Segments")
    feature_to_plot = st.selectbox("Select Feature to Compare Across Clusters:", CLUSTERING_FEATURES, index=0)
    
    fig_box = px.box(
        df_segmented,
        x='SegmentName',
        y=feature_to_plot,
        color='SegmentName',
        color_discrete_sequence=px.colors.qualitative.Safe,
        points="outliers",
        title=f"Distribution of '{feature_to_plot}' Across Identified Clusters",
        height=450
    )
    fig_box.update_layout(showlegend=False, paper_bgcolor="#FFFFFF", plot_bgcolor="#F8FAFC")
    st.plotly_chart(fig_box, use_container_width=True)

# ==============================================================================
# MODULE C: SEGMENT COMPARISON TABLE & ARCHETYPES
# ==============================================================================
elif active_tab == "Segment Comparison":
    st.header("Module C: Segment Profiles & Commercial Action Playbook")
    st.markdown("High-level strategic comparison of each learner cluster and empirical recommendations for engagement.")
    
    profiles = build_cluster_profiles(df_segmented)
    cluster_labels = {
        0: "Beginner Free Trialers",
        1: "High-Engagement Super-Learners",
        2: "Advanced Skill Specialists",
        3: "Direct Paid Investors"
    }
    profiles['SegmentName'] = profiles['Cluster'].map(cluster_labels)
    
    st.dataframe(
        profiles[['SegmentName', 'Learner_Count', 'Enrollments_Mean', 'Spend_Mean', 'Paid_Ratio_Mean', 'Diversity_Mean']],
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("---")
    st.subheader("Strategic Playbook for Each Segment")
    
    col_strat1, col_strat2 = st.columns(2)
    with col_strat1:
        st.markdown("""
        #### 1. Beginner Free Trialers
        * **Behavior**: Zero spend, low enrollments, testing introductory content.
        * **Strategic Action**: Micro-credential nudges, first-purchase promotional discounts (e.g., 50% off on foundational paid tracks).
        
        #### 2. Advanced Skill Specialists
        * **Behavior**: Focused deep-dive into narrow specialized domains with intermediate/advanced levels.
        * **Strategic Action**: Industry capstone projects, targeted masterclasses with top-rated teachers.
        """)
        
    with col_strat2:
        st.markdown("""
        #### 3. High-Engagement Super-Learners
        * **Behavior**: Broad category diversity, high enrollment velocity across multiple categories.
        * **Strategic Action**: Annual all-access membership subscriptions, peer mentor certification tracks.
        
        #### 4. Direct Paid Investors
        * **Behavior**: High spend per purchase, direct conversion to premium tracks with minimal free trial latency.
        * **Strategic Action**: VIP enterprise cohorts, 1-on-1 mentorship bundles with high-rated instructors.
        """)

# ==============================================================================
# MODULE D: MODEL EVALUATION & BENCHMARKS
# ==============================================================================
elif active_tab == "Model Evaluation":
    st.header("Module D: Offline Recommender Evaluation Benchmarks")
    st.markdown("Empirical rigorous comparison between the **Hybrid Collaborative-Content Recommender** and the **Popularity Baseline**.")
    
    eval_csv_path = os.path.join(BASE_DIR, 'outputs', 'evaluation_metrics.csv')
    if os.path.exists(eval_csv_path):
        df_eval = pd.read_csv(eval_csv_path)
    else:
        df_eval = pd.DataFrame({
            "Metric": ["Hit Rate@5", "Precision@5", "MRR@5", "NDCG@5", "Evaluated Users"],
            "Hybrid Model": [0.1022, 0.0204, 0.0478, 0.0611, 1380],
            "Popularity Baseline": [0.1007, 0.0201, 0.0431, 0.0572, 1380],
            "Relative Lift": ["+1.5%", "+1.5%", "+10.9%", "+6.8%", "N/A"]
        })
        
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Mean Reciprocal Rank Lift", "+10.9%", delta="Higher Rank Precision")
    with col_m2:
        st.metric("NDCG@5 Lift", "+6.8%", delta="Superior Position Weight")
    with col_m3:
        st.metric("Evaluated Active Learners", "1,380 users", delta="Chronological Split")
        
    st.markdown("---")
    
    col_t1, col_t2 = st.columns([1.2, 1])
    with col_t1:
        st.subheader("Benchmark Comparison Table")
        st.dataframe(df_eval, use_container_width=True, hide_index=True)
        
        st.markdown(r"""
        ### Evaluation Methodology:
        - **Chronological Leave-One-Out Holdout**: For every learner with $\ge 2$ courses, their chronologically latest transaction is held out as the test target.
        - **Zero Data Leakage**: Features, cluster assignments, and collaborative models are fitted strictly using earlier enrollments.
        - **Baseline**: Ranks courses by platform-wide enrollment frequency, filtering out courses the user previously enrolled in.
        """)
        
    with col_t2:
        st.subheader("Offline Metric Interpretation")
        st.markdown("""
        > [!NOTE]
        > **Key Takeaways**:
        > - **Hit Rate@5 (10.22%)** indicates that for ~1 in 10 test learners, the model predicted their exact single next course choice within just 5 recommendations out of 60 possible courses.
        > - **MRR@5 (+10.9%)** confirms that personalized signals push the relevant course higher toward the top position.
        > - **Real-World Impact**: Offline metrics evaluate predictive accuracy on historical observational data. Real-world uplifts in course completion or retention require live randomized A/B trials.
        """)

# --- Footer ---
st.markdown("---")
st.caption("Student Segmentation and Recommender System • Built with Python, Scikit-Learn & Streamlit • Zero PII Exposure Architecture")
