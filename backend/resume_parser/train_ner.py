import spacy
from spacy.training.example import Example
import pandas as pd
import random
import os
import re
import warnings

warnings.filterwarnings("ignore")

print("Loading data from CSV files...")

# 1. LOAD THE SKILLS DATABASE
try:
    skills_df = pd.read_csv('resume_parser/data/skills_db.csv')
    SKILLS_DB = skills_df['skill'].dropna().str.lower().tolist()
except FileNotFoundError:
    print("Error: Could not find resume_parser/data/skills_db.csv!")
    exit()

# 2. LOAD THE SENTENCES FROM KAGGLE DATA
dataset_df = pd.read_csv('resume_parser/data/resume_dataset.csv')
target_rows = dataset_df[dataset_df['label'].isin(['SKILLS', 'EXPERIENCE'])]
RAW_SENTENCES = target_rows['text'].dropna().astype(str).tolist()

TRAIN_DATA = []

print(f"Scanning {len(RAW_SENTENCES)} lines for skills using Regex...")
for sentence in RAW_SENTENCES:
    entities = []
    for skill in SKILLS_DB:
        # Search for the skill using regex word boundaries
        for match in re.finditer(r'\b' + re.escape(skill) + r'\b', sentence.lower()):
            entities.append((match.start(), match.end(), "SKILL"))
    
    if entities:
        # Sort entities and remove overlaps to prevent spaCy errors
        entities = sorted(entities, key=lambda x: x[0])
        valid_entities = []
        last_end = -1
        for start, end, label in entities:
            if start >= last_end:
                valid_entities.append((start, end, label))
                last_end = end
        
        if valid_entities:
            TRAIN_DATA.append((sentence, {"entities": valid_entities}))

print(f"Found {len(TRAIN_DATA)} fully annotated sentences to train on.")

# 3. TRAIN THE NER MODEL
print("Loading base English grammar model...")
nlp = spacy.load("en_core_web_sm")
ner = nlp.get_pipe("ner")

for _, annotations in TRAIN_DATA:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

print("Fine-tuning the NER model (this may take a minute)...")
pipe_exceptions = ["ner"]
unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

with nlp.disable_pipes(*unaffected_pipes):
    optimizer = nlp.create_optimizer()
    for itn in range(30):
        random.shuffle(TRAIN_DATA)
        losses = {}
        for text, annotations in TRAIN_DATA:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            nlp.update([example], drop=0.3, sgd=optimizer, losses=losses)
        
        if (itn + 1) % 10 == 0:
            print(f"Iteration {itn + 1} - Loss: {losses['ner']:.4f}")

os.makedirs("resume_parser/models/custom_ner", exist_ok=True)
nlp.to_disk("resume_parser/models/custom_ner")
print("\nSuccess! CSV-trained NER model saved.")