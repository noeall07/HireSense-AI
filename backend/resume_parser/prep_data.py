import pandas as pd
import re

print("Loading Kaggle dataset...")
kaggle_df = pd.read_csv('resume_parser/data/Resume.csv')

education_keywords = ['university', 'college', 'bachelor', 'master', 'degree', 'gpa', 'institute', 'school', 'diploma', 'cbse', 'b.tech', 'm.tech']
experience_keywords = ['worked', 'developed', 'managed', 'project', 'experience', 'intern', 'implemented', 'led', 'designed', 'responsible', 'created', 'maintained', 'engineered']
skill_keywords = ['skills', 'technologies', 'proficient', 'languages', 'frameworks', 'tools', 'stack', 'database', 'frontend', 'backend', 'programming']

pools = {"EDUCATION": [], "EXPERIENCE": [], "SKILLS": [], "EXTRA": []}

for _, row in kaggle_df.iterrows():
    lines = str(row['Resume_str']).split('\n')
    for line in lines:
        clean = line.strip()
        if len(clean) < 10 or len(clean.split()) < 3:
            continue
            
        line_lower = clean.lower()
        if any(w in line_lower for w in education_keywords):
            pools["EDUCATION"].append(clean)
        elif any(w in line_lower for w in skill_keywords):
            pools["SKILLS"].append(clean)
        elif any(w in line_lower for w in experience_keywords):
            pools["EXPERIENCE"].append(clean)
        else:
            pools["EXTRA"].append(clean)

# Target equal representation (up to 1,200 per class)
TARGET_PER_CLASS = min(len(pools["EDUCATION"]), len(pools["SKILLS"]), len(pools["EXPERIENCE"]), 1200)

balanced_rows = []
for label, items in pools.items():
    s = pd.Series(items).drop_duplicates().sample(n=TARGET_PER_CLASS, random_state=42)
    for text in s:
        balanced_rows.append({"text": text, "label": label})

final_df = pd.DataFrame(balanced_rows).sample(frac=1.0, random_state=42).reset_index(drop=True)
final_df.to_csv('resume_parser/data/resume_dataset.csv', index=False)
print(f"Dataset generated with {len(final_df)} evenly distributed samples ({TARGET_PER_CLASS} per class).")