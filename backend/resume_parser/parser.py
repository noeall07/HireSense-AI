import pymupdf
import re
import joblib
import spacy
import pandas as pd
import ollama

class ResumeParser:
    def __init__(self):
        self.section_classifier = joblib.load("resume_parser/models/section_classifier.joblib")
        self.ner_model = spacy.load("resume_parser/models/custom_ner")
        
        # Load skill dictionary and build canonical mapping
        self.canonical_map = {}
        try:
            skills_df = pd.read_csv('resume_parser/data/skills_db.csv')
            for _, row in skills_df.iterrows():
                self.canonical_map[str(row['skill']).strip().lower()] = str(row['canonical']).strip()
                
            # Add EntityRuler for exact keyword matching
            if "entity_ruler" not in self.ner_model.pipe_names:
                ruler = self.ner_model.add_pipe("entity_ruler", before="ner")
                patterns = []
                for skill in self.canonical_map.keys():
                    patterns.append({
                        "label": "SKILL", 
                        "pattern": [{"LOWER": token} for token in skill.split()]
                    })
                ruler.add_patterns(patterns)
        except FileNotFoundError:
            print("Warning: skills_db.csv not found. Relying solely on statistical NER.")

    def extract_text_blocks(self, pdf_path):
        doc = pymupdf.open(pdf_path)
        clean_blocks = []
        for page in doc:
            for b in page.get_text("blocks"):
                if b[6] == 0:
                    text = b[4].replace('\n', ' ').strip()
                    text = re.sub(r'^[❖•\-\|\*\s]+', '', text)
                    text = re.sub(r'[\s\|]+$', '', text)
                    if len(text.split()) > 2:
                        clean_blocks.append(text)
        return clean_blocks

    def parse(self, pdf_path):
        blocks = self.extract_text_blocks(pdf_path)
        
        structured_data = {
            "EDUCATION": [],
            "EXPERIENCE": [],
            "SKILLS": []
        }
        
        # 1. Section Classification via SVM
        for block in blocks:
            pred = self.section_classifier.predict([block])[0]
            if pred in structured_data:
                structured_data[pred].append(block)

        # 2. Hybrid NER Skill Extraction with Canonical Normalization
        detected_canonical_skills = set()
        for text in blocks:
            doc = self.ner_model(text)
            for ent in doc.ents:
                if ent.label_ == "SKILL":
                    clean_token = ent.text.strip().lower()
                    canonical_name = self.canonical_map.get(clean_token, ent.text.title())
                    detected_canonical_skills.add(canonical_name)

        # 3. Dynamic Technical Profiling
        skill_categories = {
            "Languages": ["C", "C++", "C#", "Python", "Java", "JavaScript", "SQL"],
            "Frontend": ["HTML", "CSS", "React", "Bootstrap", "Vite", "Three.js"],
            "Backend": ["Node.js", "Express", "Flask", "REST APIs", "JSON"],
            "Databases & Cloud": ["MongoDB", "PostgreSQL", "Google Cloud"],
            "Tools": ["Git", "GitHub", "Visual Studio Code", "Unity"]
        }

        categorized_skills = {}
        for category, keywords in skill_categories.items():
            matched = [s for s in detected_canonical_skills if s in keywords]
            if matched:
                categorized_skills[category] = matched
                
        #generate ques
        questions = self.generate_ques(list(detected_canonical_skills))

        return {
            "education": structured_data["EDUCATION"],
            "experience": structured_data["EXPERIENCE"],
            "extracted_skills": sorted(list(detected_canonical_skills)),
            "skill_breakdown": categorized_skills,
            "metrics": {
                "total_skills_found": len(detected_canonical_skills),
                "primary_domain": max(categorized_skills, key=lambda k: len(categorized_skills[k])) if categorized_skills else "None"
            },
            "interview_questions": questions
        }
        
    def generate_ques(self,skills):
        if not skills:
            return "No skills detected to generate questions."
        
        skills_str = ", ".join(skills[:5])
        
        prompt = f""" You are an expert interviewer. Generate 5 challenging interview questions 
        based on the following skills: {skills_str}. Generate only 3 technical interview questions 
        based strictly on these skills. Return ONLY the questions, numbered 1 to 3. Do not include
        answers or any other texts."""
        
        try:
            response = ollama.chat(model='mistral',messages=[{"role": "user", "content": prompt}])
            #split text block
            raw_out= response['message']['content']
            questions= [q.strip() for q in raw_out.strip().split('\n') if q.strip()]
            return questions[:3]
        
        except Exception as e:
            print(f"Error generating questions: {e}")
            return ["Error generating questions. Please try again later."]

if __name__ == "__main__":
    parser = ResumeParser()
    result = parser.parse("test_resume.pdf")
    import pprint
    pprint.pprint(result, width=100)