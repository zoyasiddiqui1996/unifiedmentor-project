import os
import re
import pypdf
import pandas as pd

def extract_all():
    pdf_path = r'C:\Users\zoya\.gemini\antigravity\brain\4ca06963-608d-4dd4-b4a5-621a9b07d9e7\.user_uploaded\media_1793164484815_ba204532.pdf'
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'private')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Reading PDF from: {pdf_path}")
    reader = pypdf.PdfReader(pdf_path)
    total_pages = len(reader.pages)
    print(f"Total pages: {total_pages}")
    
    users = []
    teachers = []
    courses = []
    transactions = []
    
    # 1. Users (Pages 1 to 66 -> indices 0 to 65)
    print("Extracting Users...")
    for idx in range(0, 66):
        text = reader.pages[idx].extract_text()
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        # skip header
        i = 0
        while i < len(lines):
            if lines[i].startswith('U') and re.match(r'^U\d{5}$', lines[i]):
                # Found UserID
                uid = lines[i]
                uname = lines[i+1]
                age = float(lines[i+2])
                gender = lines[i+3]
                email = lines[i+4]
                users.append({
                    'UserID': uid,
                    'UserName': uname,
                    'Age': age,
                    'Gender': gender,
                    'Email': email
                })
                i += 5
            else:
                i += 1
                
    # 2. Teachers (Pages 67 to 68 -> indices 66 to 67)
    print("Extracting Teachers...")
    for idx in range(66, 68):
        text = reader.pages[idx].extract_text()
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        i = 0
        while i < len(lines):
            if lines[i].startswith('TC') and re.match(r'^TC\d{5}$', lines[i]):
                tid = lines[i]
                tname = lines[i+1]
                age = float(lines[i+2])
                gender = lines[i+3]
                expertise = lines[i+4]
                yoe = float(lines[i+5])
                rating = float(lines[i+6])
                teachers.append({
                    'TeacherID': tid,
                    'TeacherName': tname,
                    'Age': age,
                    'Gender': gender,
                    'Expertise': expertise,
                    'YearsOfExperience': yoe,
                    'TeacherRating': rating
                })
                i += 7
            else:
                i += 1

    # 3. Courses (Pages 69 to 70 -> indices 68 to 69)
    print("Extracting Courses...")
    for idx in range(68, 70):
        text = reader.pages[idx].extract_text()
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        i = 0
        while i < len(lines):
            if lines[i].startswith('CR') and re.match(r'^CR\d{5}$', lines[i]):
                cid = lines[i]
                cname = lines[i+1]
                category = lines[i+2]
                ctype = lines[i+3]
                level = lines[i+4]
                price = float(lines[i+5])
                duration = float(lines[i+6])
                rating = float(lines[i+7])
                courses.append({
                    'CourseID': cid,
                    'CourseName': cname,
                    'CourseCategory': category,
                    'CourseType': ctype,
                    'CourseLevel': level,
                    'CoursePrice': price,
                    'CourseDuration': duration,
                    'CourseRating': rating
                })
                i += 8
            else:
                i += 1

    # 4. Transactions (Pages 71 to 288 -> indices 70 to 287)
    print("Extracting Transactions...")
    for idx in range(70, total_pages):
        text = reader.pages[idx].extract_text()
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        i = 0
        while i < len(lines):
            if lines[i].startswith('TT') and re.match(r'^TT\d{5}$', lines[i]):
                tid = lines[i]
                uid = lines[i+1]
                cid = lines[i+2]
                tdate = lines[i+3]
                amt = float(lines[i+4])
                pay_method = lines[i+5]
                teacher_id = lines[i+6]
                transactions.append({
                    'TransactionID': tid,
                    'UserID': uid,
                    'CourseID': cid,
                    'TransactionDate': tdate,
                    'Amount': amt,
                    'PaymentMethod': pay_method,
                    'TeacherID': teacher_id
                })
                i += 7
            else:
                i += 1

    df_users = pd.DataFrame(users)
    df_teachers = pd.DataFrame(teachers)
    df_courses = pd.DataFrame(courses)
    df_transactions = pd.DataFrame(transactions)
    
    print("\n--- Extraction Summary ---")
    print(f"Users extracted: {len(df_users)}")
    print(f"Teachers extracted: {len(df_teachers)}")
    print(f"Courses extracted: {len(df_courses)}")
    print(f"Transactions extracted: {len(df_transactions)}")
    
    # Save to Excel
    excel_path = os.path.join(output_dir, 'EduPro_Dataset.xlsx')
    print(f"Saving combined workbook to {excel_path}...")
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_users.to_excel(writer, sheet_name='Users', index=False)
        df_teachers.to_excel(writer, sheet_name='Teachers', index=False)
        df_courses.to_excel(writer, sheet_name='Courses', index=False)
        df_transactions.to_excel(writer, sheet_name='Transactions', index=False)
        
    # Also save separate CSVs
    df_users.to_csv(os.path.join(output_dir, 'users.csv'), index=False)
    df_teachers.to_csv(os.path.join(output_dir, 'teachers.csv'), index=False)
    df_courses.to_csv(os.path.join(output_dir, 'courses.csv'), index=False)
    df_transactions.to_csv(os.path.join(output_dir, 'transactions.csv'), index=False)
    print("All private files saved successfully.")

if __name__ == '__main__':
    extract_all()
