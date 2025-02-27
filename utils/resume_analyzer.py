import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

nlp = spacy.load('en_core_web_sm')

def analyze_resume(text):
    doc = nlp(text)
    
    # Extract skills (customize this list)
    skills = ['Python', 'Java', 'Machine Learning', 'SQL', 'Flask', 'Django']
    found_skills = [token.text for token in doc if token.text in skills]
    
    # Extract education (look for keywords)
    education = []
    for sent in doc.sents:
        if 'education' in sent.text.lower() or 'degree' in sent.text.lower():
            education.append(sent.text)
    
    # Extract experience (look for dates and job titles)
    experience = []
    for ent in doc.ents:
        if ent.label_ == 'DATE':
            experience.append(ent.text)
    
    return {
        'skills': list(set(found_skills)),
        'education': education,
        'experience': experience
    }

def calculate_relevance_score(resume_text, job_description):
    corpus = [resume_text, job_description]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    similarity = (tfidf_matrix * tfidf_matrix.T).A[0, 1]
    return similarity