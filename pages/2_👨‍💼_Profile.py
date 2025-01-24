import streamlit as st
from src.utils.styling import load_css
from src.utils.config import config
import base64

def get_pdf_download_link(pdf_path, filename):
    """Generate download link for PDF file"""
    with open(pdf_path, "rb") as f:
        bytes_data = f.read()
    b64 = base64.b64encode(bytes_data).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}" class="download-button">📥 Download CV</a>'
    return href

st.set_page_config(
    page_title=config.get('PAGE_TITLE_PROFILE'), 
    page_icon=config.get('PAGE_ICON_PROFILE'), 
    layout=config.get('LAYOUT_PROFILE')
)

load_css()

# Custom CSS
st.markdown("""
    <style>
    .css-1v0mbdj.etr89bj1 {
        text-align: center;
    }
    .profile-img {
        border-radius: 50%;
        margin: 0 auto;
        display: block;
    }
    .social-links {
        text-align: center;
        padding: 1rem 0;
    }
    .experience-card {
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
        border-left: 3px solid #0366d6;
    }
    </style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 2])

with col1:
    # Profile Image
    st.image("static/img/foto.jpg", width=200, output_format="auto")
    
    # Contact Information
    st.markdown("""
    ### Contact
    - 📧 andhikaputra1301@gmail.com
    - 📱 082213676520
    """)
    
    # Social Links
    st.markdown("""
    ### Social Links
    - [![LinkedIn](https://img.icons8.com/color/48/000000/linkedin.png)](https://www.linkedin.com/in/andhika-putra-bagaskara13/)
    - [![Instagram](https://img.icons8.com/color/48/000000/instagram-new--v1.png)](https://www.instagram.com/andhikaputrab/)
    - [![Instagram](https://img.icons8.com/color/48/000000/github.png)](https://github.com/andhikaputrab)
    """)
    
with col2:
    st.title("Andhika Putra Bagaskara")
    st.subheader("Data Analyst")
    st.markdown(get_pdf_download_link("cv/CV_Andhika_Putra_Bagaskara.pdf", "cv_andhika_putra_bagaskara.pdf"), unsafe_allow_html=True)
    
    st.markdown("""
    ### Summary
    A data analysis enthusiast with an educational background in Informatics Engineering and certification in Data Analytics, 
    and currently studying machine learning. Proficiency in data processing, visualisation, and analysis using Python, 
    Tableau, and Excel. Good problem-solving and critical thinking skills, committed to providing accurate data insights to support company
    decision-making, and highly motivated to continue learning and developing in the field of data analysis.
    """)
    
# Experience Section
st.header("Professional Experience")

experiences = [
    {
        "role": "Internship",
        "company": "PT Pertamina Persero",
        "period": "October 2022 - October 2023",
        "points": [
            "Automate the management of ticket request data from the website using RPA and send the analysis via email to the relevant department",
            "Create a website dashboard to manage internal asset",
        ]
    }
]

for exp in experiences:
    with st.expander(f"**{exp['company']} - {exp['role']}**", expanded=True):
        st.markdown(f"**Period:** {exp['period']}")
        for point in exp['points']:
            st.markdown(f"- {point}")
            
# Skill Section
st.header("Skill & Expertise")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Technical Skills")
    technical_skills = [
        "Machine Learning",
        "Data Analysis",
        "Python Programming",
        "SQL",
        "Data Visualization"
    ]
    for skill in technical_skills:
        st.markdown(f"- {skill}")

with col2:
    st.subheader("Tools & Technologies")
    tools = [
        "Python",
        "R Studio",
        "Tableau",
        "Microsoft Office",
    ]
    for tool in tools:
        st.markdown(f"- {tool}")

# Education Section
st.header("Education")
st.markdown("""
#### Telkom University
- **Degree:** Master Degree of Informatics Engineering
- **Relevant Coursework:** Data Analysis, Algorithms, Machine Learning, Artificial Intelligence
- **Current GPA:** 3.14/4.00
- **Period:** 2024 - Present
""")
st.markdown("""
#### Universitas Komputer Indonesia
- **Degree:** Bachelor Degree of Informatics Engineering
- **Relevant Coursework:** Software Engineering, Algorithms, Artificial Intelligence
- **GPA:** 3.07/4.00
- **Period:** 2017 - 2022
""")

# Certifications
st.header("Certifications")
certifications = [
    {
        "title": "Big Data using Python",
        "orginizer": "Online Course – Kominfo",
        "period": "2021"
    },
    {
        "title": "Google Data Analytics",
        "orginizer": "Online Course – Kominfo",
        "period": "2024"
    }
]
    
for cert in certifications:
    with st.expander(f"**{cert['title']}**", expanded=True):
        st.markdown(f"{cert['orginizer']}")
        st.markdown(f"Period: {cert['period']}")
        # for point in cert['points']:
        #     st.markdown(f"- {point}")
        
