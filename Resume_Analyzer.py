import streamlit as st
import pickle
import re
import nltk
from PyPDF2 import PdfReader  # For extracting text from PDF

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load pre-trained models
clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

# Function to clean the resume text
def clean_resume(text):
    """Cleans the input resume text."""
    text = re.sub(r'http\S+\s', '', text)  # Remove URLs
    text = re.sub(r'RT|cc', '', text)  # Remove RT and cc
    text = re.sub(r'@\S+', '', text)  # Remove mentions
    text = re.sub(r'\r\n', ' ', text)  # Replace newlines with spaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
    text = re.sub(r'#\S+', '', text)  # Remove hashtags
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Function to map category ID to category name
def get_category_name(category_id):
    """Converts category ID to a human-readable category name."""
    category_mapping = {
        6: 'Data Science', 12: 'HR', 0: 'Advocate', 1: 'Arts',
        24: 'Web Designing', 16: 'Mechanical Engineer',
        22: 'Sales', 14: 'Health and Fitness', 5: 'Civil Engineer',
        15: 'Java Developer', 4: 'Business Analyst',
        21: 'SAP Developer', 2: 'Automation Testing',
        11: 'Electrical Engineering', 18: 'Operations Manager',
        20: 'Python Developer', 8: 'DevOps Engineer',
        17: 'Network Security Engineer', 19: 'PMO',
        7: 'Database', 13: 'Hadoop', 10: 'ETL Developer',
        9: 'DotNet Developer', 3: 'Blockchain', 23: 'Testing'
    }
    return category_mapping.get(category_id, "Unknown")

# Function to extract text from a PDF file
def extract_text_from_pdf(file):
    """Extracts text from a PDF file."""
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Main function to run the Streamlit app
def main():
    # Apply a custom theme and title
    st.markdown(
        """
        <style>
        body {
            font-family: Arial, sans-serif;
        }
        .title {
            text-align: center;
            color: #2C3E50;
        }
        .sidebar .sidebar-content {
            background-color: #F7F9F9;
        }
        </style>
        """, unsafe_allow_html=True
    )

    # Title Section
    st.markdown("<h1 class='title'>Resume Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Upload your resume to find its predicted category.</p>", unsafe_allow_html=True)

    # File Upload Section
    st.sidebar.title("Upload Section")
    uploaded_file = st.sidebar.file_uploader("Upload Resume", type=['pdf', 'txt'])

    if uploaded_file is not None:
        # Extract text from the uploaded file
        if uploaded_file.type == "application/pdf":
            resume_text = extract_text_from_pdf(uploaded_file)
        else:  # Assume it's a text file
            resume_text = uploaded_file.read().decode('utf-8', errors='ignore')
        
        # Clean the extracted text
        cleaned_resume = clean_resume(resume_text)
        
        # Vectorize the cleaned resume
        vectorized_resume = tfidf.transform([cleaned_resume])
        
        # Predict the category ID
        predicted_id = clf.predict(vectorized_resume)[0]
        
        # Get the category name
        predicted_category = get_category_name(predicted_id)
        
        # Display the predicted category with styling
        st.markdown(f"<h2 style='text-align: center; color: #16A085;'>Predicted Category: {predicted_category}</h2>", unsafe_allow_html=True)
    else:
        st.info("Please upload a PDF or TXT file to analyze.")

if __name__ == "__main__":
    main()
