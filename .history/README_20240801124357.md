## Kenya Public Participation Tool

The Kenya Public Participation Tool is a web application designed to facilitate public engagement and feedback on government documents such as bills, policies, and regulations. The goal of this tool is to enable citizens to easily access, understand, and provide comments on these documents, ultimately promoting transparency and inclusive governance in Kenya.

### Features

1. **Document Viewing**: Users can select from a list of available government documents and view their content in the app.

2. **AI-Generated Summaries**: The tool uses natural language processing to generate concise summaries of the documents, making it easier for users to understand the key points.

3. **Highlights and Impacts**: The AI system identifies and explains the potential pros and cons of the document, helping users grasp the document's significance and implications.

4. **Q&A with AI**: Users can ask questions about the document, and the AI assistant provides relevant answers based on the document's content.

5. **Public Comments**: Citizens can submit their comments on the document, which are categorized by sentiment (positive, negative, or neutral) and topic.

6. **Comment Analysis**: The tool analyzes the submitted comments and provides visualizations of the sentiment distribution and category breakdown. It also generates a summary of the main themes and feedback from the comments.

7. **Privacy and Security**: The app ensures user privacy by collecting and processing data in accordance with Kenyan data protection laws.

### Technologies Used

- **Streamlit**: A Python library for building interactive web applications
- **Groq**: A natural language processing API for generating AI content
- **PyPDF2**: A library for extracting text from PDF documents
- **SQLite**: A lightweight, serverless database for storing user comments
- **TextBlob**: A library for performing sentiment analysis on text

### Getting Started

To run the Kenya Public Participation Tool locally, follow these steps:

1. **Clone the repository**: `git clone https://github.com/your-username/kenya-public-participation-tool.git`
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Set up environment variables**: Create a `.env` file in the project root directory and add your Groq API key.
4. **Place PDF documents**: Add the government documents you want to make available in the `documents` directory.
5. **Run the app**: `streamlit run app.py`

The app will start running on `http://localhost:8501`.

### Contributing

Contributions to the Kenya Public Participation Tool are welcome! If you find any issues or have suggestions for improvements, please create a new issue or submit a pull request.

### License

This project is licensed under the [MIT License](LICENSE).

