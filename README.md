# Resume-RAG-Classifier

This is a simple Python-based tool that assists job applicants in selecting the most appropriate version of their resume (i.e., Business or Technical) based on the content of a job description. By analyzing the semantic similarity between the job description and each resume type, the program recommends the better-fitting resume for submission.

Intended for use by Business/Data Analytics folks who might apply to a variety of roles that can skew either business or technical. Based on the job description, they will want to submit tailored resumes that highlight their strengths for that specific role and domain.

## Features

- Accepts a job description as input (plain text)
- Computes similarity scores against both business and technical resumes
- Recommends the more appropriate resume based on highest similarity
- Built using OpenAI's language model API for semantic comparison (OpenAI SDK v1+)

## Prerequisites

- Python 3.8+
- `openai` Python SDK (v1.0.0+)
- `PyPDF2` and `numpy`
- API key from OpenAI (with `gpt-4` or `gpt-3.5-turbo` access)

## Installation

1. Clone the repository or download the script:
```bash
git clone https://github.com/yourusername/resume-classifier.git
cd resume-classifier
```

2. Install dependencies:
```bash
pip install openai PyPDF2 numpy
```

3. Save your OpenAI API key in a file named openai_key.txt.

4. Update file paths inside the script to point to your own PDF resumes.

## Usage

Run the script and paste in the job description when prompted:

```
python resume_rag_classifier.py
```
