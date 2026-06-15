from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# FastAPI অ্যাপ ইনিশিয়ালাইজ করা
app = FastAPI(title="JobOrbitBD ML API")

# ডাটা রিসিভ করার জন্য মডেল (ফ্লাটার থেকে এই ফরমেটে ডাটা আসবে)
class MatchRequest(BaseModel):
    student_skills: str
    job_requirements: str

# রুট এন্ডপয়েন্ট (চেক করার জন্য)
@app.get("/")
def read_root():
    return {"message": "Welcome to JobOrbitBD Machine Learning API 🚀"}

# --- আসল মেশিন লার্নিং লজিক (Skill Matching) ---
@app.post("/calculate-match")
def calculate_match(data: MatchRequest):
    # ১. স্টুডেন্টের স্কিল এবং জবের রিকোয়ারমেন্ট একসাথে রাখা
    documents = [data.student_skills, data.job_requirements]
    
    # ২. টেক্সট থেকে ভেক্টর (Number) তৈরি করা (TF-IDF মডেল)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    # ৩. Cosine Similarity দিয়ে দুইটার মধ্যে মিল (Match Score) বের করা
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    match_percentage = round(similarity[0][0] * 100, 2)
    
    # ৪. স্কিল গ্যাপ অ্যানালাইসিস (কোন স্কিলগুলো স্টুডেন্টের নেই তা বের করা)
    student_words = set(data.student_skills.lower().split(', '))
    job_words = set(data.job_requirements.lower().split(', '))
    missing_skills = list(job_words - student_words)

    # ফ্লাটার অ্যাপে রেজাল্ট পাঠানো
    return {
        "match_score": f"{match_percentage}%",
        "missing_skills": missing_skills,
        "message": f"Your profile is a {match_percentage}% match for this job!"
    }