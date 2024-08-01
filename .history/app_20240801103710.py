import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from groq import Groq
import PyPDF2
import json
from textblob import TextBlob

# Load environment variables
load_dotenv()

# Initialize Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Function to read PDF content
def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

# Function to generate AI summary and key issues
def generate_ai_content(document_text):
    prompt = f"""
    Analyze the following document and provide:
    1. A 500-word summary
    2. 800 words on key issues and their impact on users

    Document:
    {document_text[:2000]}  # Limiting input to avoid token limits
    """
    
    response = groq_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="mixtral-8x7b-32768",
        max_tokens=2000,
    )
    
    return response.choices[0].message.content

# Function to analyze sentiment
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0.1:
        return "Positive"
    elif sentiment < -0.1:
        return "Negative"
    else:
        return "Neutral"

# Function to save comment
def save_comment(comment, document_name):
    comment_dir = "comments"
    os.makedirs(comment_dir, exist_ok=True)
    file_name = f"{comment_dir}/{document_name}_{len(os.listdir(comment_dir))}.txt"
    with open(file_name, "w") as f:
        f.write(comment)

# Function to load and analyze comments
def analyze_comments(document_name):
    comments = []
    sentiments = []
    for file in os.listdir("comments"):
        if file.startswith(document_name):
            with open(f"comments/{file}", "r") as f:
                comment = f.read()
                comments.append(comment)
                sentiments.append(analyze_sentiment(comment))
    
    # Perform analysis
    sentiment_counts = pd.Series(sentiments).value_counts()
    
    # Generate summary using AI
    all_comments = "\n".join(comments)
    summary_prompt = f"Summarize the following comments in 200 words:\n\n{all_comments[:5000]}"
    summary_response = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": summary_prompt}],
        model="mixtral-8x7b-32768",
        max_tokens=300,
    )
    summary = summary_response.choices[0].message.content
    
    return len(comments), sentiment_counts, summary

# Streamlit app
st.set_page_config(layout="wide")

# Main layout
col1, col2, col3 = st.columns([1, 2, 1])

# Column 1 (Left)
with col1:
    st.title("Kenya Public Participation Tool")
    
    document_list = [f for f in os.listdir("documents") if f.endswith(".pdf")]
    selected_document = st.selectbox("Select a document", document_list)
    
    st.write("Comment via:")
    st.write("X | TikTok | LinkedIn")
    
    with st.expander("Privacy Statement"):
        st.write("Your privacy is important to us. We collect and process your data in accordance with Kenyan data protection laws.")
    
    st.write("© Bonga Talk Research")

# Column 2 (Middle)
with col2:
    if selected_document:
        st.subheader("Original Document")
        doc_path = f"documents/{selected_document}"
        doc_content = read_pdf(doc_path)
        st.text_area("Document content", value=doc_content[:1000] + "...", height=200, disabled=True)
        
        ai_content = generate_ai_content(doc_content)
        summary, key_issues = ai_content.split("2. 800 words on key issues and their impact on users")
        
        with st.expander("AI-Generated Summary"):
            st.write(summary)
        
        st.subheader("Key Issues and Impact")
        st.write(key_issues)

# Column 3 (Right)
with col3:
    st.subheader("Provide Your Comments")
    user_comment = st.text_area("Your comment", height=150)
    uploaded_file = st.file_uploader("Upload supporting document (max 1MB)", type=["pdf", "docx"], accept_multiple_files=False)
    
    if st.button("Submit Comment"):
        if user_comment:
            save_comment(user_comment, selected_document)
            st.success("Comment submitted successfully!")
            st.session_state['comment_submitted'] = True
        else:
            st.error("Please enter a comment before submitting.")

# Results section (appears after submission)
if st.session_state.get('comment_submitted', False):
    st.header("Results")
    
    comment_count, sentiment_counts, comment_summary = analyze_comments(selected_document)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Number of Comments", comment_count)
        
        # Sentiment Analysis Pie Chart
        fig, ax = plt.subplots()
        ax.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')
        ax.set_title("Sentiment Analysis")
        st.pyplot(fig)
    
    with col2:
        st.subheader("Summary of Comments")
        st.write(comment_summary)

# Add some spacing at the bottom
st.write("\n\n")