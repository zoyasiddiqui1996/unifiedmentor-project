import os
import pandas as pd
import numpy as np

def run_inspection():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data', 'private')
    excel_path = os.path.join(data_dir, 'EduPro_Dataset.xlsx')
    
    print("=" * 60)
    print("STAGE 1: COMPREHENSIVE DATA AUDIT & INSPECTION REPORT")
    print("=" * 60)
    print(f"Dataset path: {excel_path}\n")
    
    excel = pd.ExcelFile(excel_path)
    sheets = excel.sheet_names
    print(f"Workbook Sheets Found ({len(sheets)}): {sheets}\n")
    
    dfs = {s: pd.read_excel(excel_path, sheet_name=s) for s in sheets}
    
    # 1. Structure, Counts, Nulls, Duplicates
    print("-" * 50)
    print("1. TABLE STRUCTURE & DATA QUALITY SUMMARY")
    print("-" * 50)
    for name, df in dfs.items():
        print(f"\n[Table: {name}]")
        print(f"  - Rows: {len(df):,}")
        print(f"  - Columns ({len(df.columns)}): {list(df.columns)}")
        print("  - Data Types:")
        for col, dtype in df.dtypes.items():
            print(f"      {col}: {dtype}")
        
        # Missing values
        null_counts = df.isnull().sum()
        total_nulls = null_counts.sum()
        print(f"  - Total Missing Values: {total_nulls}")
        if total_nulls > 0:
            for col, n in null_counts[null_counts > 0].items():
                print(f"      {col}: {n} missing ({n/len(df)*100:.2f}%)")
        
        # Duplicates
        pk = df.columns[0]
        dupe_pk = df[pk].duplicated().sum()
        full_dupes = df.duplicated().sum()
        print(f"  - Duplicate Primary Keys ({pk}): {dupe_pk}")
        print(f"  - Completely Duplicate Rows: {full_dupes}")

    # 2. Referential Integrity (Foreign Keys)
    print("\n" + "-" * 50)
    print("2. REFERENTIAL INTEGRITY (FOREIGN KEYS)")
    print("-" * 50)
    df_users = dfs['Users']
    df_courses = dfs['Courses']
    df_teachers = dfs['Teachers']
    df_trans = dfs['Transactions']
    
    user_ids = set(df_users['UserID'])
    course_ids = set(df_courses['CourseID'])
    teacher_ids = set(df_teachers['TeacherID'])
    
    orphan_users = set(df_trans['UserID']) - user_ids
    orphan_courses = set(df_trans['CourseID']) - course_ids
    orphan_teachers = set(df_trans['TeacherID']) - teacher_ids
    
    print(f"  - Transactions referencing missing Users: {len(orphan_users)}")
    print(f"  - Transactions referencing missing Courses: {len(orphan_courses)}")
    print(f"  - Transactions referencing missing Teachers: {len(orphan_teachers)}")
    
    users_with_trans = len(set(df_trans['UserID']))
    print(f"  - Unique learners with at least one transaction: {users_with_trans} / {len(df_users)} ({users_with_trans/len(df_users)*100:.2f}%)")
    courses_with_trans = len(set(df_trans['CourseID']))
    print(f"  - Unique courses with at least one transaction: {courses_with_trans} / {len(df_courses)} ({courses_with_trans/len(df_courses)*100:.2f}%)")
    teachers_with_trans = len(set(df_trans['TeacherID']))
    print(f"  - Unique teachers assigned to transactions: {teachers_with_trans} / {len(df_teachers)} ({teachers_with_trans/len(df_teachers)*100:.2f}%)")

    # 3. Transaction Dates & Integrity
    print("\n" + "-" * 50)
    print("3. TRANSACTION DATES & TEMPORAL INTEGRITY")
    print("-" * 50)
    df_trans['TransactionDate'] = pd.to_datetime(df_trans['TransactionDate'], errors='coerce')
    invalid_dates = df_trans['TransactionDate'].isnull().sum()
    print(f"  - Invalid / Unparseable Dates: {invalid_dates}")
    min_date = df_trans['TransactionDate'].min()
    max_date = df_trans['TransactionDate'].max()
    print(f"  - Date Range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
    print(f"  - Total Calendar Days Covered: {(max_date - min_date).days + 1}")

    # 4. Financial & Pricing Integrity (Zero Amount Investigation)
    print("\n" + "-" * 50)
    print("4. FINANCIAL INTEGRITY & ZERO-AMOUNT INVESTIGATION")
    print("-" * 50)
    merged_trans = df_trans.merge(df_courses[['CourseID', 'CourseType', 'CoursePrice']], on='CourseID', how='left')
    free_courses = set(df_courses[df_courses['CourseType'] == 'Free']['CourseID'])
    paid_courses = set(df_courses[df_courses['CourseType'] == 'Paid']['CourseID'])
    
    free_enrollments = merged_trans[merged_trans['CourseType'] == 'Free']
    paid_enrollments = merged_trans[merged_trans['CourseType'] == 'Paid']
    
    print(f"  - Total Transactions: {len(merged_trans):,}")
    print(f"  - Free Course Enrollments: {len(free_enrollments):,}")
    print(f"      Amount min/max for Free courses: {free_enrollments['Amount'].min():.2f} / {free_enrollments['Amount'].max():.2f}")
    print(f"  - Paid Course Transactions: {len(paid_enrollments):,}")
    print(f"      Paid course catalog prices: min=${df_courses[df_courses['CourseType']=='Paid']['CoursePrice'].min():.2f}, max=${df_courses[df_courses['CourseType']=='Paid']['CoursePrice'].max():.2f}")
    
    # Check if transaction amount matches catalog price
    price_diff = paid_enrollments['Amount'] - paid_enrollments['CoursePrice']
    mismatches = (price_diff.abs() > 0.01).sum()
    print(f"  - Paid Transactions with Amount mismatch vs CoursePrice: {mismatches}")
    paid_zero = (paid_enrollments['Amount'] == 0.0).sum()
    print(f"  - Paid Course Enrollments with $0.00 amount: {paid_zero}")
    
    total_rev = df_trans['Amount'].sum()
    print(f"  - Total Platform Revenue in Dataset: ${total_rev:,.2f}")
    print("  - Payment Methods Breakdown:")
    print(df_trans['PaymentMethod'].value_counts().to_string())

    # 5. Course Catalog & Rating Analysis
    print("\n" + "-" * 50)
    print("5. COURSE CATALOG & RATING FINDINGS")
    print("-" * 50)
    print("  - Course Categories (Count of Courses):")
    print(df_courses['CourseCategory'].value_counts().to_string())
    print("\n  - Course Levels:")
    print(df_courses['CourseLevel'].value_counts().to_string())
    print("\n  - Course Type:")
    print(df_courses['CourseType'].value_counts().to_string())
    print("\n  - Course Rating Summary (Platform Catalog Rating):")
    print(df_courses['CourseRating'].describe().to_string())
    print("\n  [CRITICAL NOTE ON RATINGS]:")
    print("  The dataset contains 'CourseRating' in the Courses table (catalog rating) and 'TeacherRating' in Teachers.")
    print("  There is NO individual learner rating column in Transactions or Users.")
    print("  Therefore, per Data Integrity Rules, we must NOT attribute catalog ratings as individual learner satisfaction ratings.")

    # 6. Learner Demographics & Activity Distribution
    print("\n" + "-" * 50)
    print("6. LEARNER ACTIVITY & DEMOGRAPHICS AUDIT")
    print("-" * 50)
    print(f"  - Learner Age: min={df_users['Age'].min():.0f}, median={df_users['Age'].median():.0f}, max={df_users['Age'].max():.0f}")
    print("  - Learner Gender distribution:")
    print(df_users['Gender'].value_counts().to_string())
    
    trans_per_user = df_trans.groupby('UserID').size()
    print("\n  - Transactions per Learner:")
    print(f"      Min: {trans_per_user.min()}, Median: {trans_per_user.median():.0f}, Max: {trans_per_user.max()}")
    print(f"      25th percentile: {trans_per_user.quantile(0.25):.0f}, 75th percentile: {trans_per_user.quantile(0.75):.0f}")
    print(f"      Learners with exactly 1 transaction: {(trans_per_user == 1).sum():,} ({(trans_per_user == 1).sum()/len(df_users)*100:.2f}%)")
    print(f"      Learners with 2-4 transactions: {((trans_per_user >= 2) & (trans_per_user <= 4)).sum():,}")
    print(f"      Learners with 5+ transactions: {(trans_per_user >= 5).sum():,}")

    print("\n" + "=" * 60)
    print("STAGE 1 INSPECTION COMPLETE.")
    print("=" * 60)

if __name__ == '__main__':
    run_inspection()
