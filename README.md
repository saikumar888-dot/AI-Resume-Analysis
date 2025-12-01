# AI Resume Analyzer 📝

AI Resume Analyzer is an intelligent web-based tool that helps job seekers evaluate and improve their resumes. It compares your resume against a target job description using advanced AI, simulates ATS screening, and provides actionable feedback—all in a single Python file.

## Features

- **Resume Text Extraction**: Upload PDF resumes to extract text automatically.
- **Job Description Input**: Paste the job description for analysis.
- **ATS Similarity Score**: Sentence Transformers (BERT-based) calculate how well your resume matches the job criteria.
- **AI-Powered Evaluation**: Groq's Llama-based LLM rates your resume, highlights strengths/gaps with emojis, and provides personalized suggestions.
- **Downloadable Report**: Get a detailed, human-readable analysis to help optimize your resume.

## Installation

**Prerequisites:**  
- Python 3.x  
- Git

**Setup Steps:**

1. **Clone the repository**
    ```bash
    git clone https://github.com/Altoks-AI/AI-Resume-Analyzer.git
    cd AI-Resume-Analyzer
    ```

2. **Create and activate a virtual environment**
    ```bash
    python -m venv myenv
    # On Windows:
    myenv\Scripts\activate
    # On macOS/Linux:
    source myenv/bin/activate
    ```

3. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4. **Set your Groq API Key**  
   Create a `.env` file in the project root:
    ```
    GROQ_API_KEY=your_groq_api_key_here
    ```

5. **Run the Streamlit app**
    ```bash
    streamlit run main.py
    ```

The app will open at [http://localhost:8501](http://localhost:8501).

## Usage

- Upload your PDF resume.
- Paste the job description.
- View your similarity score and detailed feedback.
- Download the report for reference.
