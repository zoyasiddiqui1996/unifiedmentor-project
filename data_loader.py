"""
Data loader and preprocessor for EduPro dataset.
Ensures strict privacy: drops all PII (names, emails) upon ingestion.
"""

import os
import pandas as pd
import numpy as np

def get_data_paths():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    private_path = os.path.join(base_dir, 'data', 'private', 'EduPro_Dataset.xlsx')
    sample_dir = os.path.join(base_dir, 'data', 'sample')
    return private_path, sample_dir

def load_data():
    """
    Loads dataset from private storage if available, otherwise falls back to sample data.
    Strictly removes PII columns (UserName, Email, TeacherName).
    Returns:
        dict containing 'users', 'teachers', 'courses', 'transactions' DataFrames.
    """
    private_path, sample_dir = get_data_paths()
    
    if os.path.exists(private_path):
        excel = pd.ExcelFile(private_path)
        df_users = pd.read_excel(excel, sheet_name='Users')
        df_teachers = pd.read_excel(excel, sheet_name='Teachers')
        df_courses = pd.read_excel(excel, sheet_name='Courses')
        df_trans = pd.read_excel(excel, sheet_name='Transactions')
        data_source = 'private'
    elif os.path.exists(os.path.join(sample_dir, 'users_sample.csv')):
        df_users = pd.read_csv(os.path.join(sample_dir, 'users_sample.csv'))
        df_teachers = pd.read_csv(os.path.join(sample_dir, 'teachers_sample.csv'))
        df_courses = pd.read_csv(os.path.join(sample_dir, 'courses_sample.csv'))
        df_trans = pd.read_csv(os.path.join(sample_dir, 'transactions_sample.csv'))
        data_source = 'sample'
    else:
        raise FileNotFoundError("Neither private nor sample dataset found. Please place EduPro_Dataset.xlsx in data/private/.")
        
    # --- Strict Privacy Protection: Drop PII immediately ---
    pii_cols_users = [c for c in ['UserName', 'Email'] if c in df_users.columns]
    if pii_cols_users:
        df_users = df_users.drop(columns=pii_cols_users)
        
    pii_cols_teachers = [c for c in ['TeacherName'] if c in df_teachers.columns]
    if pii_cols_teachers:
        df_teachers = df_teachers.drop(columns=pii_cols_teachers)
        
    # Standardize data types
    df_users['UserID'] = df_users['UserID'].astype(str).str.strip()
    df_users['Age'] = pd.to_numeric(df_users['Age'], errors='coerce')
    df_users['Gender'] = df_users['Gender'].astype(str).str.strip()
    
    df_courses['CourseID'] = df_courses['CourseID'].astype(str).str.strip()
    df_courses['CourseName'] = df_courses['CourseName'].astype(str).str.strip()
    df_courses['CourseCategory'] = df_courses['CourseCategory'].astype(str).str.strip()
    df_courses['CourseType'] = df_courses['CourseType'].astype(str).str.strip()
    df_courses['CourseLevel'] = df_courses['CourseLevel'].astype(str).str.strip()
    df_courses['CoursePrice'] = pd.to_numeric(df_courses['CoursePrice'], errors='coerce').fillna(0.0)
    df_courses['CourseDuration'] = pd.to_numeric(df_courses['CourseDuration'], errors='coerce')
    df_courses['CourseRating'] = pd.to_numeric(df_courses['CourseRating'], errors='coerce')
    
    df_teachers['TeacherID'] = df_teachers['TeacherID'].astype(str).str.strip()
    df_teachers['Age'] = pd.to_numeric(df_teachers['Age'], errors='coerce')
    df_teachers['Gender'] = df_teachers['Gender'].astype(str).str.strip()
    df_teachers['Expertise'] = df_teachers['Expertise'].astype(str).str.strip()
    df_teachers['YearsOfExperience'] = pd.to_numeric(df_teachers['YearsOfExperience'], errors='coerce')
    df_teachers['TeacherRating'] = pd.to_numeric(df_teachers['TeacherRating'], errors='coerce')
    
    df_trans['TransactionID'] = df_trans['TransactionID'].astype(str).str.strip()
    df_trans['UserID'] = df_trans['UserID'].astype(str).str.strip()
    df_trans['CourseID'] = df_trans['CourseID'].astype(str).str.strip()
    df_trans['TransactionDate'] = pd.to_datetime(df_trans['TransactionDate'], errors='coerce')
    df_trans['Amount'] = pd.to_numeric(df_trans['Amount'], errors='coerce').fillna(0.0)
    df_trans['PaymentMethod'] = df_trans['PaymentMethod'].astype(str).str.strip()
    df_trans['TeacherID'] = df_trans['TeacherID'].astype(str).str.strip()
    
    return {
        'users': df_users,
        'teachers': df_teachers,
        'courses': df_courses,
        'transactions': df_trans,
        'source': data_source
    }
