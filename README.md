# Student Segmentation and Recommender System

An end-to-end data science and machine learning application to analyze learner enrollment behavior, segment users into meaningful behavioral clusters, and provide explainable, multi-factor course recommendations.

---

## Project Structure

```text
├── .streamlit/
│   └── config.toml             # Streamlit visual theme and server settings
├── data/
│   ├── private/                # (Git-ignored) Full confidential dataset
│   │   ├── EduPro_Dataset.xlsx # Original 4-sheet workbook
│   │   ├── users.csv
│   │   ├── teachers.csv
│   │   ├── courses.csv
│   │   └── transactions.csv
│   └── sample/                 # Safe anonymized sample dataset (zero PII)
├── docs/                       # Research papers, executive summaries, methodology notes
├── outputs/                    # Exported figures, cluster models, evaluation metrics
├── src/                        # Modular source code
│   ├── extract_dataset.py      # Dataset extraction utility
│   ├── stage1_inspect.py       # Data quality audit script
│   ├── data_loader.py          # Data ingestion and privacy sanitization
│   ├── features.py             # Learner feature engineering
│   ├── clustering.py           # Segmentation models (K-Means, Hierarchical)
│   ├── recommender.py          # Hybrid recommendation engine
│   └── evaluation.py           # Offline ranking and hit-rate evaluation
├── tests/                      # Automated unit and integration tests (pytest)
├── app.py                      # Interactive Streamlit dashboard
├── edupro_analysis.py          # Standalone reproducible analysis pipeline
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git exclusions for privacy & caches
└── README.md                   # Project documentation
