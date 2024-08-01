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
import time

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

# Function to save summary and highlights as Markdown
def save_markdown(document_name, summary, highlights):
    folder_path = "summaries"
    os.makedirs(folder_path, exist_ok=True)
    with open(f"{folder_path}/{document_name}_summary.md", "w") as f:
        f.write("# AI-Generated Summary\n")
        f.write(summary + "\n")
    with open(f"{folder_path}/{document_name}_highlights.md", "w") as f:
        f.write("# Key Highlights and Impacts\n")
        f.write(highlights + "\n")

# Function to display PDF
# def display_pdf(file_path):
#     with open(file_path, "rb") as f:
#         base64_pdf = base64.b64encode(f.read()).decode('utf-8')
#     pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
#     st.markdown(pdf_display, unsafe_allow_html=True)

# AI System Prompt
SYSTEM_PROMPT = """
You are an AI research assistant in public law, policy, and regulation. Your role is to help citizens understand government documents, 
provide summaries, answer questions, and analyze public comments. Always strive to be impartial, accurate, and respectful of 
diverse viewpoints. Your goal is to facilitate informed public participation in governance.
"""

# AI Prompts
SUMMARY_PROMPT = """
Provide a 1000 character summary of the following document in natural layman's language.

{document_text}
"""

HIGHLIGHTS_PROMPT = """
Create a bulleted list of key pros and cons from the document, highlight what is vague or unclear, what are the principles and some counter argument principles. MAX 2000 characters.

{document_text}
"""

ANSWER_PROMPT = """
Based on the following document, answer the user's question in natural layman's language. Be impartial and respectful and not be overtly supportive of the document. Assume the user too has a valid argument and agree with them when they have a valid point. Remember that even if the user has expressed negatively to the document, your role is only to outline what the document says and its possible implications.

Document: {document_text}

User Question: {user_question}
"""

SENTIMENT_PROMPT = """
Analyze the sentiment of the following user comment. 
Categorize it as either Positive, Negative, or Neutral and only one of the three.

{user_comment}
"""

COMMENT_SUMMARY_PROMPT = """
Summarize and categorize the user comments. 
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

# Set the page layout to wide
st.set_page_config(page_title="Chat with AI", layout="wide")

# First Row (Viewport Height Constrained)
row1_col1, row1_col2, row1_col3 = st.columns(3)

# Column 1
with row1_col1:
    st.title("Public GPT")
    st.write("The Kenya Public Participation Tool is a web application designed to facilitate public engagement and feedback on government documents such as bills, policies, and regulations. The goal of this tool is to enable citizens to easily access, understand, and provide comments on these documents, ultimately promoting transparency and inclusive governance in Kenya.")
    
    # Dropdown select
    document_list = [f for f in os.listdir("documents") if f.endswith(".pdf")]
    document_list.insert(0, "None")  # Add "None" as the first option
    selected_document = st.selectbox("Select a document", document_list, index=0)  # Set index=0 to default to "None"
    
    # Social media links
    st.write("Comment via:")
    st.write("X | TikTok | LinkedIn")
    
    # Privacy Statement
    with st.expander("Privacy Statement"):
        st.write("Your privacy is important to us. We collect and process your data in accordance with Kenyan data protection laws.")
    
    st.write("© Bonga Talk Research")

# Column 2
with row1_col2:
    st.write("")  # Empty space
    st.write("")  # Empty space
    if selected_document and selected_document != "None":
        st.subheader("Original Document")
        doc_path = f"documents/{selected_document}"
        st.markdown(f"[Open Original Document](file://{doc_path})", unsafe_allow_html=True)
        
        # Check if summary and highlights already exist
        summary_path = f"summaries/{selected_document}_summary.md"
        highlights_path = f"summaries/{selected_document}_highlights.md"
        
        if os.path.exists(summary_path):
            st.subheader("AI-Generated Summary")
            with open(summary_path, "r") as f:
                st.markdown(f.read())
        else:
            st.write("Summary not available.")
        
        if os.path.exists(highlights_path):
            st.subheader("Key Highlights and Impacts")
            with open(highlights_path, "r") as f:
                st.markdown(f.read())
        else:
            st.write("Highlights not available.")
    else:
        st.write("No document selected")

# Column 3
with row1_col3:
    st.write("")  # Empty space
    st.write("")  # Empty space
    st.subheader("Chat with AI")
    
    # Initialize session state for chat history if it doesn't exist
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Display chat messages from the history
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            with st.chat_message("user"):
                st.markdown(f"**You:** {message['content']}")
        else:
            with st.chat_message("assistant"):
                st.markdown(f"**AI:** {message['content']}")

    # Chat input field
    user_question = st.chat_input("Ask a question about the document")

    # Handle user input
    if user_question:
        # Generate AI response
        ai_answer = generate_ai_content(ANSWER_PROMPT.format(document_text=doc_content[:5000], user_question=user_question))

        # Store the user question and AI response in chat history
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        st.session_state.chat_history.append({"role": "assistant", "content": ai_answer})

        # Display the AI response
        with st.chat_message("assistant"):
            st.markdown(f"**AI:** {ai_answer}")

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

# Second Row (if needed)
# You can add another row here for additional content or functionality
if st.session_state.get('comment_submitted', False) and selected_document != "None":
    st.header("Results")
    
    # Load comments for summary
    comment_count, sentiment_counts, category_counts, comment_summary = analyze_comments(selected_document)
    
    # Display the summary of comments
    st.subheader("Summary of Comments")
    if comment_count == 1:
        st.write(comment_summary)  # Display single user's comment
    else:
        st.write("Multiple comments received. Summary will be based on user feedback.")
    
    # Display aggregate sentiment counts
    st.subheader("Sentiment Analysis")
    st.write("Total Comments:", comment_count)
    st.write("Sentiment Counts:")
    for sentiment, count in sentiment_counts.items():
        st.write(f"{sentiment}: {count}")
    
    # Display category distribution if needed
    st.subheader("Comment Categories")
    st.bar_chart(category_counts)

# Close SQLite connection
conn.close()

# Add some spacing at the bottom
st.write("\n\n")
st.write("\n\n")