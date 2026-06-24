import streamlit as st
import pickle
import pdfplumber
import re
import string
import pandas as pd
 
from sklearn.metrics.pairwise import cosine_similarity
 
# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
 
st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="wide"
)
 
# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
 
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
 
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
 
# ---------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------
 
def clean_text(text):
 
    if not isinstance(text, str):
        return ""
 
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )
    text = re.sub(r'\s+', ' ', text).strip()
 
    return text
 
 
def extract_skills(text):

    skills = [

        # Data Science / ML
        "python", "pandas", "numpy", "matplotlib", "seaborn",
        "scikit-learn", "machine learning", "deep learning",
        "tensorflow", "keras", "pytorch", "nlp",
        "data analysis", "data science", "statistics",
        "power bi", "tableau", "sql",

        # Web Development
        "html", "css", "javascript", "typescript",
        "react", "angular", "vue", "nextjs",
        "node", "nodejs", "express",
        "php", "laravel", "django", "flask",
        "mongodb", "mysql", "postgresql",

        # Full Stack
        "rest api", "graphql", "jwt", "firebase",

        # Cloud / DevOps
        "aws", "azure", "gcp", "docker",
        "kubernetes", "jenkins", "terraform",
        "linux", "git", "github",

        # Mobile Development
        "android", "ios", "flutter",
        "react native", "swift", "kotlin",

        # Cyber Security
        "penetration testing", "ethical hacking",
        "network security", "wireshark",
        "metasploit", "cyber security",

        # UI/UX Design
        "figma", "adobe xd", "photoshop",
        "illustrator", "ui design", "ux design",

        # Business / Management
        "project management", "agile", "scrum",
        "jira", "leadership", "communication",

        # Marketing
        "digital marketing", "seo", "sem",
        "google analytics", "content marketing",

        # Finance
        "financial analysis", "accounting",
        "budgeting", "forecasting", "excel",

        # HR
        "recruitment", "talent acquisition",
        "employee relations", "payroll",

        # Sales
        "salesforce", "crm", "lead generation",
        "business development", "negotiation",

        # Soft Skills
        "problem solving", "critical thinking",
        "teamwork", "time management",
        "presentation skills"
    ]

    text = text.lower()

    found_skills = []

    for skill in skills:
        if skill in text:
            found_skills.append(skill)

    return found_skills 
 
def extract_text_from_pdf(pdf_file):
 
    text = ""
 
    with pdfplumber.open(pdf_file) as pdf:
 
        for page in pdf.pages:
 
            page_text = page.extract_text()
 
            if page_text:
                text += page_text + " "
 
    return text
 
 
def predict_role(skills):

    skills = [s.lower() for s in skills]

    # Data Science
    if any(skill in skills for skill in [
        "machine learning", "data science", "pandas",
        "numpy", "tensorflow", "pytorch", "tableau"
    ]):
        return "Data Science / Machine Learning"

    # Full Stack
    elif "react" in skills and ("node" in skills or "nodejs" in skills):
        return "Full Stack Developer"

    # Frontend
    elif any(skill in skills for skill in [
        "react", "angular", "vue", "html", "css"
    ]):
        return "Frontend Developer"

    # Backend
    elif any(skill in skills for skill in [
        "node", "nodejs", "express", "django",
        "flask", "laravel"
    ]):
        return "Backend Developer"

    # DevOps
    elif any(skill in skills for skill in [
        "aws", "docker", "kubernetes", "jenkins"
    ]):
        return "DevOps Engineer"

    # Cyber Security
    elif any(skill in skills for skill in [
        "ethical hacking", "cyber security",
        "penetration testing"
    ]):
        return "Cyber Security Analyst"

    # UI/UX
    elif any(skill in skills for skill in [
        "figma", "adobe xd", "ui design", "ux design"
    ]):
        return "UI/UX Designer"

    # Marketing
    elif any(skill in skills for skill in [
        "seo", "digital marketing",
        "google analytics"
    ]):
        return "Digital Marketing Specialist"

    # Finance
    elif any(skill in skills for skill in [
        "accounting", "financial analysis",
        "budgeting"
    ]):
        return "Finance Analyst"

    # HR
    elif any(skill in skills for skill in [
        "recruitment", "talent acquisition",
        "employee relations"
    ]):
        return "HR Specialist"

    # Sales
    elif any(skill in skills for skill in [
        "crm", "salesforce",
        "lead generation"
    ]):
        return "Sales Executive"

    else:
        return "General Professional"
 
# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
 
st.markdown(
    """
    <h1 style='text-align:center;color:#4CAF50;'>
    📄 AI Resume Screening System
    </h1>
    <hr>
    """,
    unsafe_allow_html=True
)
 
# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
 
st.sidebar.title("📌 Navigation")
 
page = st.sidebar.radio(
    "Select Module",
    [
        "Resume Classification",
        "Resume Matching",
        "Candidate Ranking"
    ]
)
 
# ===================================================
# PAGE 1 : RESUME CLASSIFICATION
# ===================================================
 
if page == "Resume Classification":
 
    st.header("📄 Resume Classification")
 
    uploaded_resume = st.file_uploader(
        "Upload Resume PDF",
        type=["pdf"]
    )
 
    if uploaded_resume:
 
        if st.button("🔍 Analyze Resume"):
 
            resume_text = extract_text_from_pdf(
                uploaded_resume
            )
 
            skills_found = extract_skills(
                resume_text
            )
 
            st.subheader("🛠 Detected Skills")
 
            if skills_found:
 
                cols = st.columns(4)
 
                for i, skill in enumerate(skills_found):
                    cols[i % 4].success(skill)
 
            else:
                st.warning(
                    "No skills detected"
                )
 
            role = predict_role(
                skills_found
            )
 
            st.subheader(
                "🎯 Predicted Role"
            )
 
            st.success(
                role
            )
 
# ===================================================
# PAGE 2 : RESUME MATCHING
# ===================================================
 
elif page == "Resume Matching":
 
    st.header("📊 Resume Matching")
 
    resume_file = st.file_uploader(
        "Upload Resume PDF",
        type=["pdf"],
        key="resume"
    )
 
    jd_option = st.radio(
        "Job Description Input Method",
        [
            "Upload JD PDF",
            "Paste JD Text"
        ]
    )
 
    jd_text = ""
 
    if jd_option == "Upload JD PDF":
 
        jd_file = st.file_uploader(
            "Upload Job Description PDF",
            type=["pdf"],
            key="jd"
        )
 
        if jd_file:
            jd_text = extract_text_from_pdf(
                jd_file
            )
 
    else:
 
        jd_text = st.text_area(
            "Paste Job Description Here",
            height=200
        )
 
    if resume_file and jd_text:
 
        if st.button("📈 Calculate Match Score"):
 
            resume_text = extract_text_from_pdf(
                resume_file
            )
 
            resume_skills = extract_skills(
                resume_text
            )

            role = predict_role(resume_skills)
 
            jd_skills = extract_skills(
                jd_text
            )
 
            missing_skills = []
 
            for skill in jd_skills:
 
                if skill not in resume_skills:
                    missing_skills.append(skill)
 
            resume_clean = clean_text(
                resume_text
            )
 
            jd_clean = clean_text(
                jd_text
            )
 
            resume_vector = vectorizer.transform(
                [resume_clean]
            )
 
            jd_vector = vectorizer.transform(
                [jd_clean]
            )
 
            similarity = cosine_similarity(
                resume_vector,
                jd_vector
            )
 
            match_score = round(
                similarity[0][0] * 100,
                2
            )

            if match_score >= 80:
                match_level = "🟢 Excellent Match"
            elif match_score >= 60:
                match_level = "🟡 Good Match"
            elif match_score >= 40:
                match_level = "🟠 Average Match"
            else:
                match_level = "🔴 Low Match"
 
            st.subheader("📊 Match Score")
 
            st.metric(
                "Match %",
                f"{match_score}%"
            )

            st.subheader("📋 ATS Evaluation")

            if match_score >= 80:
                st.success("Resume is highly suitable for this job.")
            elif match_score >= 60:
                st.info("Resume is a good match but can be improved.")
            elif match_score >= 40:
                st.warning("Resume needs improvement for this role.")
            else:
                st.error("Resume is not suitable for this job.")
            
            st.progress(
                min(int(match_score), 100)
            )

            st.info(match_level)

            st.subheader("💼 Recommended Role")
            st.success(role)

            st.subheader(
                "🛠 Resume Skills"
            )
 
            if resume_skills:
 
                cols = st.columns(4)
 
                for i, skill in enumerate(
                    resume_skills
                ):
                    cols[i % 4].success(skill)
 
            else:
                st.warning(
                    "No skills detected"
                )
 
            st.subheader(
                "⚠ Missing Skills"
            )
 
            if missing_skills:
 
                cols = st.columns(4)
 
                for i, skill in enumerate(
                    missing_skills
                ):
                    cols[i % 4].warning(skill)
 
            else:
                st.success(
                    "All required skills found"
                )
            
            report = f"""
            Resume Match Report

            Match Score: {match_score}%

            Recommended Role: {role}

            Skills Found:
            {', '.join(resume_skills)}

            Missing Skills:
            {', '.join(missing_skills)}
            """

            st.download_button(
                label="📥 Download Report",
                data=report,
                file_name="resume_report.txt",
                mime="text/plain"
            )

            st.subheader("📌 Resume Summary")

            st.write(f"Total Skills Found: {len(resume_skills)}")
            st.write(f"Missing Skills: {len(missing_skills)}")
            st.write(f"Recommended Role: {role}")

 
# ===================================================
# PAGE 3 : CANDIDATE RANKING
# ===================================================
 
elif page == "Candidate Ranking":
 
    st.header("🏆 Candidate Ranking System")
 
    resume_files = st.file_uploader(
        "Upload Multiple Resume PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )
 
    jd_option = st.radio(
        "Job Description Input Method",
        [
            "Upload JD PDF",
            "Paste JD Text"
        ],
        key="ranking_jd"
    )
 
    jd_text = ""
 
    if jd_option == "Upload JD PDF":
 
        jd_file = st.file_uploader(
            "Upload Job Description PDF",
            type=["pdf"],
            key="ranking_pdf"
        )
 
        if jd_file:
            jd_text = extract_text_from_pdf(
                jd_file
            )
 
    else:
 
        jd_text = st.text_area(
            "Paste Job Description",
            height=200,
            key="ranking_text"
        )
 
    if resume_files and jd_text:
 
        if st.button("🏆 Rank Candidates"):
 
            jd_clean = clean_text(
                jd_text
            )
 
            jd_vector = vectorizer.transform(
                [jd_clean]
            )
 
            results = []
 
            for resume_file in resume_files:
 
                resume_text = extract_text_from_pdf(
                    resume_file
                )
 
                resume_clean = clean_text(
                    resume_text
                )
 
                resume_vector = vectorizer.transform(
                    [resume_clean]
                )
 
                similarity = cosine_similarity(
                    resume_vector,
                    jd_vector
                )
 
                score = round(
                    similarity[0][0] * 100,
                    2
                )
 
                results.append(
                    {
                        "Candidate": resume_file.name,
                        "Match Score (%)": score
                    }
                )
 
            results = sorted(
                results,
                key=lambda x: x["Match Score (%)"],
                reverse=True
            )
 
            st.subheader(
                "🏆 Candidate Rankings"
            )
 
            rank_df = pd.DataFrame(
                results
            )
 
            rank_df.index = rank_df.index + 1
 
            st.dataframe(
                rank_df,
                use_container_width=True
            )
 
            st.subheader(
                "Top Candidate"
            )
 
            st.success(
                f"{results[0]['Candidate']} "
                f"({results[0]['Match Score (%)']}%)"
            )
 
            st.subheader(
                "Ranking Summary"
            )
 
            for i, row in enumerate(
                results,
                start=1
            ):
 
                score = row["Match Score (%)"]
 
                if score >= 80:
                    icon = "🟢"
                    status = "Strong Match"
 
                elif score >= 60:
                    icon = "🟡"
                    status = "Potential Match"
 
                else:
                    icon = "🔴"
                    status = "Low Match"
 
                st.write(
                    f"{icon} Rank {i}: "
                    f"{row['Candidate']} - "
                    f"{score}% ({status})"
                )
 
 
# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
 
st.markdown("---")
 
st.caption(
    "Intelligent Resume Screening Platform for Candidate Classification and Job Role Matching"
)