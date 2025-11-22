import streamlit as st
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.lib.pagesizes import A4
from groq import Groq
import re
from dotenv import load_dotenv
import os
import io
import datetime

# PDF-related imports
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

import matplotlib.pyplot as plt

# ---------------- ENV + SESSION SETUP ----------------

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if "form_submitted" not in st.session_state:
    st.session_state.form_submitted = False
if "resume" not in st.session_state:
    st.session_state.resume = ""
if "job_desc" not in st.session_state:
    st.session_state.job_desc = ""

st.title("AI Resume Analyzer 📝")


# ---------------- CORE FUNCTIONS ----------------

def extract_pdf_text(uploaded_file):
    """Extract raw text from a PDF file."""
    try:
        return extract_text(uploaded_file)
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""


def calculate_similarity_bert(text1, text2):
    """Calculate semantic similarity using sentence-transformers."""
    ats_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    emb1 = ats_model.encode([text1])
    emb2 = ats_model.encode([text2])
    sim = cosine_similarity(emb1, emb2)[0][0]
    return sim


def get_report(resume, job_desc):
    """Call Groq LLM to generate a detailed resume vs JD analysis."""
    client = Groq(api_key=api_key)
    prompt = f"""
    You are an AI Resume Analyzer. Compare the candidate's resume against the job description.

    Requirements:
    - Identify key skills, experience, tools, and qualifications from the job description.
    - For each important point, score the resume's match out of 5. Format like: "✅ Skill XYZ – 4/5 – explanation..."
      - Use ✅ when the resume clearly matches.
      - Use ❌ when it clearly does not match.
      - Use ⚠ when it is unclear or partially met.
    - At the end, include a section with the heading:
      "Suggestions to improve your resume:"
      and list concrete, actionable suggestions.

    Resume:
    {resume}

    ---
    Job Description:
    {job_desc}
    """
    completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
    )
    return completion.choices[0].message.content


def extract_scores(text):
    """Extract all x/5 scores from the LLM report."""
    pattern = r'(\d+(?:\.\d+)?)/5'
    matches = re.findall(pattern, text)
    return [float(m) for m in matches]


# ---------------- PDF GENERATION ----------------

def create_score_chart(ats_score, avg_score):
    """
    Create a simple bar chart comparing ATS similarity and AI evaluation.
    Returns an in-memory PNG buffer.
    """
    fig, ax = plt.subplots(figsize=(4, 3))
    ats_percent = ats_score * 100
    ai_percent = avg_score * 20 * 100 / 100  # avg_score is 0–1 when used below

    labels = ["ATS Similarity (%)", "AI Evaluation (%)"]
    values = [ats_percent, avg_score * 100]  # avg_score is in 0–1 range

    ax.bar(labels, values)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percentage")
    ax.set_title("Resume Match Overview")

    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf


def generate_pdf_report(resume, job_desc, ats_score, avg_score, report):
    """
    Generate a professional PDF report with:
    - Title
    - Date
    - Scores table
    - Simple bar chart
    - Full AI feedback
    - Footer with reviewer
    Returns raw PDF bytes.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=40, leftMargin=40,
                            topMargin=40, bottomMargin=40)

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        name="TitleStyle",
        parent=styles["Title"],
        fontSize=20,
        leading=26,
        alignment=1,  # center
        textColor=colors.white,
        backColor=colors.HexColor("#1976D2"),
        spaceAfter=18
    )

    heading_style = ParagraphStyle(
        name="HeadingStyle",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=colors.HexColor("#1976D2"),
        spaceAfter=8
    )

    body_style = ParagraphStyle(
        name="BodyStyle",
        parent=styles["BodyText"],
        fontSize=11,
        leading=16
    )

    footer_style = ParagraphStyle(
        name="FooterStyle",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.gray,
        alignment=1,
        spaceBefore=20
    )

    story = []

    # Title
    story.append(Paragraph("AI Resume Analysis Report", title_style))
    story.append(Spacer(1, 12))

    # Date
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    story.append(Paragraph(f"Generated on: {now_str}", body_style))
    story.append(Spacer(1, 12))

    # Summary heading
    story.append(Paragraph("Summary", heading_style))

    # Scores table
    ats_percent = round(ats_score * 100, 2)
    ai_percent = round(avg_score * 100, 2)

    table_data = [
        ["Metric", "Value"],
        ["ATS Similarity Score", f"{ats_percent} %"],
        ["AI Evaluation Score", f"{ai_percent} % (≈ {round(avg_score*5,2)}/5)"],
    ]

    summary_table = Table(table_data, colWidths=[180, 260])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#BBDEFB")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#90CAF9")),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 16))

    # Chart
    story.append(Paragraph("Score Overview", heading_style))
    chart_buf = create_score_chart(ats_score, avg_score)
    chart_img = Image(chart_buf, width=300, height=220)
    story.append(chart_img)
    story.append(Spacer(1, 16))

    # Optional short summary section
    story.append(Paragraph("Candidate & Role Summary", heading_style))
    story.append(Paragraph(
        "This report analyzes how well the candidate's resume aligns with the provided job description "
        "in terms of skills, experience, and relevance. The AI system also provides suggestions to improve "
        "the resume for a better match.",
        body_style
    ))
    story.append(Spacer(1, 12))

    # AI Report section
    story.append(Paragraph("Detailed AI Feedback & Evaluation", heading_style))

    for line in report.split("\n"):
        line = line.strip()
        if not line:
            continue
        story.append(Paragraph(line, body_style))
        story.append(Spacer(1, 4))

    # Footer / signature
    story.append(Spacer(1, 18))
    story.append(Paragraph("Reviewed by: AI Resume Analyzer", footer_style))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


# ---------------- STREAMLIT WORKFLOW ----------------

if not st.session_state.form_submitted:
    with st.form("input_form"):
        resume_file = st.file_uploader("Upload your Resume (PDF)", type="pdf")
        st.session_state.job_desc = st.text_area(
            "Enter Job Description:",
            placeholder="Paste the job description here..."
        )

        if st.form_submit_button("Analyze"):
            if resume_file and st.session_state.job_desc:
                st.info("Extracting resume text...")
                st.session_state.resume = extract_pdf_text(resume_file)
                st.session_state.form_submitted = True
                st.rerun()
            else:
                st.warning("Please upload both Resume and Job Description.")

if st.session_state.form_submitted:
    st.info("Generating scores and AI analysis...")

    ats_score = calculate_similarity_bert(
        st.session_state.resume,
        st.session_state.job_desc
    )

    report = get_report(
        st.session_state.resume,
        st.session_state.job_desc
    )

    scores = extract_scores(report)
    avg_score = sum(scores) / (len(scores) * 5) if scores else 0.0

    col1, col2 = st.columns(2)
    with col1:
        st.write("ATS-based similarity score:")
        st.subheader(f"{round(ats_score * 100, 2)} %")
    with col2:
        st.write("AI Evaluation Score (approx):")
        st.subheader(f"{round(avg_score*5, 2)} / 5")

    st.subheader("AI Generated Detailed Report:")
    st.markdown(
        f"<div style='text-align: left; background-color: #111; "
        f"padding: 10px; border-radius: 10px;'>{report}</div>",
        unsafe_allow_html=True
    )

    # PDF download
    pdf_bytes = generate_pdf_report(
        resume=st.session_state.resume,
        job_desc=st.session_state.job_desc,
        ats_score=ats_score,
        avg_score=avg_score,
        report=report
    )

    st.download_button(
        label="📄 Download PDF Report",
        data=pdf_bytes,
        file_name="resume_analysis_report.pdf",
        mime="application/pdf"
    )
