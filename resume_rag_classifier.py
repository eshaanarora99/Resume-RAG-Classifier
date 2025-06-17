import os
import numpy as np
from openai import OpenAI
import PyPDF2

# === 1. Load OpenAI API key from file ===
key_path = r"C:\Users\eshaa\Downloads\openai_key.txt"
with open(key_path, "r") as key_file:
    api_key = key_file.read().strip()
os.environ["OPENAI_API_KEY"] = api_key
client = OpenAI(api_key=api_key)

# === 2. PDF parsing function ===
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

# === 3. Load and extract text from resumes ===
business_pdf_path = r"C:\Users\eshaa\OneDrive\Desktop\Resumes\Business Resume\Eshaan Arora Resume.pdf"
tech_pdf_path = r"C:\Users\eshaa\OneDrive\Desktop\Resumes\Tech Resume\Eshaan Arora Resume.pdf"

business_resume = extract_text_from_pdf(business_pdf_path)
technical_resume = extract_text_from_pdf(tech_pdf_path)

# === 4. Cosine similarity function ===
def cosine_similarity(vec1, vec2):
    vec1, vec2 = np.array(vec1), np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# === 5. Classification using embeddings ===
def classify_resume_fit(client, job_description: str) -> str:
    embedding_model = "text-embedding-3-small"

    job_embedding = client.embeddings.create(
        model=embedding_model, input=[job_description]
    ).data[0].embedding

    biz_embedding = client.embeddings.create(
        model=embedding_model, input=[business_resume]
    ).data[0].embedding

    tech_embedding = client.embeddings.create(
        model=embedding_model, input=[technical_resume]
    ).data[0].embedding

    sim_business = cosine_similarity(job_embedding, biz_embedding)
    sim_tech = cosine_similarity(job_embedding, tech_embedding)

    print(f"Business Resume Similarity: {sim_business:.4f}")
    print(f"Technical Resume Similarity: {sim_tech:.4f}")

    return "Tech" if sim_tech > sim_business else "Business"

# === 6. Main CLI ===
if __name__ == "__main__":
    print("Paste the job description below. Press Enter twice when done:")
    job_lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        job_lines.append(line)

    job_description = "\n".join(job_lines)
    result = classify_resume_fit(client, job_description)
    print(f"\nRecommended Resume: {result}")
