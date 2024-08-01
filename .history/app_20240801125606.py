import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from groq import Groq
import PyPDF2
import sqlite3
from textblob import TextBlob
import base64

# Load environment variables
load_dotenv()

# Initialize Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Initialize SQLite database
conn = sqlite3.connect('comments.db')
c = conn.cursor()

# Create table if not exists
c.execute('''CREATE TABLE IF NOT EXISTS comments
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              document_name TEXT,
              comment TEXT,
              sentiment TEXT,
              category TEXT)''')
conn.commit()

# Function to read PDF content
def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

# Function to display PDF
def display_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# AI System Prompt
SYSTEM_PROMPT = """
You are an AI assistant for the Kenya Public Participation Tool. Your role is to help citizens understand government documents, 
provide summaries, answer questions, and analyze public comments. Always strive to be impartial, accurate, and respectful of 
diverse viewpoints. Your goal is to facilitate informed public participation in governance.
"""

# AI Prompts
SUMMARY_PROMPT = """
Provide a 1000 character summary of the following document in natural layman's language. Use Kenyan slang to connect with users:

{document_text}
"""

HIGHLIGHTS_PROMPT = """
Create a bulleted list of key pros and possible cons from the document and for each with a brief explanation of its impact: MAX 2000 characters.

{document_text}
"""

ANSWER_PROMPT = """
Based on the following document, answer the user's question:

Document: {document_text}

User Question: {user_question}
"""

SENTIMENT_PROMPT = """
Analyze the sentiment of the following user comment. 
Categorize it as Positive, Negative, or Neutral, and provide a brief explanation why:

{user_comment}
"""

COMMENT_SUMMARY_PROMPT = """
Summarize and categorize the following user comments. 
Identify 3-5 main categories of feedback and provide a brief summary for each:

{all_comments}
"""

# Function to generate AI content
def generate_ai_content(prompt, max_tokens=2000):
    response = groq_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="mixtral-8x7b-32768",
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content

# Function to save comment to SQLite
def save_comment(document_name, comment, sentiment, category):
    c.execute("INSERT INTO comments (document_name, comment, sentiment, category) VALUES (?, ?, ?, ?)",
              (document_name, comment, sentiment, category))
    conn.commit()

# Function to load and analyze comments
def analyze_comments(document_name):
    c.execute("SELECT comment, sentiment, category FROM comments WHERE document_name=?", (document_name,))
    comments = c.fetchall()
    
    all_comments = "\n".join([comment[0] for comment in comments])
    sentiment_counts = pd.Series([comment[1] for comment in comments]).value_counts()
    category_counts = pd.Series([comment[2] for comment in comments]).value_counts()
    
    summary = generate_ai_content(COMMENT_SUMMARY_PROMPT.format(all_comments=all_comments[:5000]))
    
    return len(comments), sentiment_counts, category_counts, summary

# Streamlit app
st.set_page_config(layout="wide")

# Main layout
col1, col2, col3 = st.columns([1, 2, 1])

# Column 1 (Left)
with col1:
    st.title("Public GPT")
    st.write("The Kenya Public Participation Tool is a web application designed to facilitate public engagement and feedback on government documents such as bills, policies, and regulations. The goal of this tool is to enable citizens to easily access, understand, and provide comments on these documents, ultimately promoting transparency and inclusive governance in Kenya.")
    
    document_list = [f for f in os.listdir("documents") if f.endswith(".pdf")]
    document_list.insert(0, "None")  # Add "None" as the first option
    selected_document = st.selectbox("Select a document", document_list, index=0)  # Set index=0 to default to "None"
    
    st.write("Comment via:")
    st.write("X | TikTok | LinkedIn")
    
    with st.expander("Privacy Statement"):
        st.write("Your privacy is important to us. We collect and process your data in accordance with Kenyan data protection laws.")
    
    st.write("© Bonga Talk Research")

# Column 2 (Middle)
with col2:
    if selected_document and selected_document != "None":
        st.subheader("Original Document")
        doc_path = f"documents/{selected_document}"
        
        # Provide a link to the original document
        st.markdown(f"[Open Original Document](file://{doc_path})", unsafe_allow_html=True)
        
        # Generate AI content
        doc_content = read_pdf(doc_path)
        summary = generate_ai_content(SUMMARY_PROMPT.format(document_text=doc_content[:5000]))
        highlights = generate_ai_content(HIGHLIGHTS_PROMPT.format(document_text=doc_content[:5000]))
        
        # Create accordions for summary and insights
        with st.expander("AI-Generated Summary"):
            st.write(summary)
        
        with st.expander("Key Highlights and Impacts"):
            st.write(highlights)
    else:
        st.write("No document selected")

# Column 3 (Right)
with col3:
    if selected_document and selected_document != "None":
        st.subheader("Chat with AI")
        user_question = st.text_input("Ask a question about the document")
        if user_question:
            ai_answer = generate_ai_content(ANSWER_PROMPT.format(document_text=doc_content[:5000], user_question=user_question))
            st.write("AI Response:", ai_answer)
        
        st.subheader("Provide Your Comments")
        user_comment = st.text_area("Your comment", height=150)
        uploaded_file = st.file_uploader("Upload supporting document (max 1MB)", type=["pdf", "docx"], accept_multiple_files=False)
        
        if st.button("Submit Comment"):
            if user_comment:
                sentiment_analysis = generate_ai_content(SENTIMENT_PROMPT.format(user_comment=user_comment))
                sentiment, category = sentiment_analysis.split('\n')[0], "General"  # You might want to refine category extraction
                save_comment(selected_document, user_comment, sentiment, category)
                st.success("Comment submitted successfully!")
                st.session_state['comment_submitted'] = True
            else:
                st.error("Please enter a comment before submitting.")
    else:
        st.write("No document selected.")

# Results section (appears after submission)
if st.session_state.get('comment_submitted', False) and selected_document != "None":
    st.header("Results")
    
    comment_count, sentiment_counts, category_counts, comment_summary = analyze_comments(selected_document)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Number of Comments", comment_count)
        
        # Sentiment Analysis Pie Chart
        fig, ax = plt.subplots()
        ax.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')
        ax.set_title("Sentiment Analysis")
        st.pyplot(fig)
    
    with col2:
        # Category Distribution Bar Chart
        fig, ax = plt.subplots()
        category_counts.plot(kind='bar', ax=ax)
        ax.set_title("Comment Categories")
        ax.set_ylabel("Number of Comments")
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)
    
    st.subheader("Summary of Comments")
    st.write(comment_summary)

# Close SQLite connection
conn.close()

# Add some spacing at the bottom
st.write("\n\n")