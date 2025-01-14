import streamlit as st
import os
import sqlite3
from datetime import datetime

from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

class SimpleRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.vectorizer = TfidfVectorizer()
        self.doc_vectors = self.vectorizer.fit_transform([doc.page_content for doc in documents])

    def get_relevant_documents(self, query, k=4):
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.doc_vectors)[0]
        top_indices = np.argsort(similarities)[-k:][::-1]
        return [self.documents[i] for i in top_indices]

def init_sqlite_db():
    """Initialize SQLite database with sample data."""
    conn = sqlite3.connect('messages.db')
    cursor = conn.cursor()

    # Create tables
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS channels (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS messages (
        id TEXT PRIMARY KEY,
        content TEXT NOT NULL,
        channel_id TEXT,
        author_id TEXT,
        created_at TIMESTAMP,
        FOREIGN KEY (channel_id) REFERENCES channels (id),
        FOREIGN KEY (author_id) REFERENCES users (id)
    )
    ''')

    # Insert sample data
    sample_channels = [
        ('ch1', 'general'),
        ('ch2', 'fitness'),
        ('ch3', 'business'),
        ('ch4', 'vigilante-stuff'),
        ('ch5', 'philosophy')
    ]

    sample_users = [
        ('u1', 'Patrick Bateman'),
        ('u2', 'Tyler Durden'),
        ('u3', 'Bruce Wayne'),
        ('u4', 'Bane'),
        ('u5', 'Arnold Schwarzenegger')
    ]

    sample_messages = [
        ('m1', 'I need to return some videotapes. But first, let me tell you about Huey Lewis and the News. Their early work was a little too new wave for my taste, but when Sports came out in 83, they really came into their own.', 'ch3', 'u1', '2024-01-01 10:00:00'),
        ('m2', 'Listen up, you are not your job. You are not how much money you have in the bank. You are not the car you drive. You are not the contents of your wallet.', 'ch5', 'u2', '2024-01-01 10:05:00'),
        ('m3', 'Its not who I am underneath, but what I do that defines me. Also, anyone interested in some new tactical equipment? My R&D department just developed something interesting.', 'ch4', 'u3', '2024-01-01 10:10:00'),
        ('m4', 'Ah you think darkness is your ally? You merely adopted the dark. I was born in it, molded by it.', 'ch4', 'u4', '2024-01-01 10:15:00'),
        ('m5', 'Remember when I said Id kill you last? I lied. Also, here is my workout routine: train six days a week, sleep six hours a night, no plan B, no backup plan, no safety net. Just get out there and do it.', 'ch2', 'u5', '2024-01-01 10:20:00')
    ]

    cursor.executemany('INSERT OR REPLACE INTO channels VALUES (?, ?)', sample_channels)
    cursor.executemany('INSERT OR REPLACE INTO users VALUES (?, ?)', sample_users)
    cursor.executemany('INSERT OR REPLACE INTO messages VALUES (?, ?, ?, ?, ?)', sample_messages)

    conn.commit()
    conn.close()

def load_messages_from_db():
    """
    Connect to SQLite database and retrieve messages.
    Return a list of LangChain Document objects.
    """
    conn = sqlite3.connect('messages.db')
    cursor = conn.cursor()

    cursor.execute("""
        SELECT 
            m.content,
            m.channel_id,
            m.author_id,
            m.created_at,
            c.name as channel_name,
            u.name as author_name
        FROM messages m
        LEFT JOIN channels c ON m.channel_id = c.id
        LEFT JOIN users u ON m.author_id = u.id
        WHERE m.content IS NOT NULL AND m.content != ''
    """)
    rows = cursor.fetchall()

    documents = []
    for row in rows:
        content, channel_id, author_id, created_at, channel_name, author_name = row
        metadata = {
            "channel_id": channel_id,
            "channel_name": channel_name,
            "author_id": author_id,
            "author_name": author_name,
            "created_at": str(created_at)
        }
        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)

    cursor.close()
    conn.close()
    return documents

def format_docs(docs):
    """Format documents for display"""
    context_snippets = []
    for i, doc in enumerate(docs, start=1):
        snippet = (
            f"--- Message {i} ---\n"
            f"Channel: {doc.metadata.get('channel_name')} ({doc.metadata.get('channel_id')})\n"
            f"Author: {doc.metadata.get('author_name')} ({doc.metadata.get('author_id')})\n"
            f"Created At: {doc.metadata.get('created_at')}\n"
            f"Content: {doc.page_content}\n"
        )
        context_snippets.append(snippet)
    return "\n".join(context_snippets)

def main():
    st.title("RAG Chat Interface")
    st.write("Query the message database using natural language. The system will find relevant messages and generate a response.")

    # Initialize database and load messages
    if 'documents' not in st.session_state:
        init_sqlite_db()
        st.session_state.documents = load_messages_from_db()
        st.session_state.retriever = SimpleRetriever(st.session_state.documents)

    # Example queries
    st.sidebar.header("Example Queries")
    example_queries = [
        "What philosophical views have been shared?",
        "What advice has been shared about fitness?",
        "What have people said about darkness or night?",
        "Who has talked about business topics?",
        "What are people's thoughts on identity or who they are?"
    ]
    
    for query in example_queries:
        if st.sidebar.button(query):
            st.session_state.user_query = query

    # Query input
    user_query = st.chat_input("Enter your query here...")
    
    if user_query:
        st.session_state.user_query = user_query

    # Process query if it exists in session state
    if 'user_query' in st.session_state:
        query = st.session_state.user_query
        
        # Get relevant documents
        docs = st.session_state.retriever.get_relevant_documents(query)
        context = format_docs(docs)

        # Display relevant messages
        with st.expander("📑 Relevant Messages", expanded=True):
            st.text(context)

        # Generate and display response
        llm = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-3.5-turbo",
            openai_api_key=st.secrets["OPENAI_API_KEY"]
        )

        prompt_template = PromptTemplate(
            input_variables=["query", "context"],
            template="{query}\n\nContext:\n{context}"
        )
        
        prompt_with_context = prompt_template.format(query=query, context=context)
        response = llm.predict(prompt_with_context)

        # Add to chat history
        st.session_state.chat_history.append({"query": query, "response": response})

        # Display chat history
        st.subheader("💬 Chat History")
        for chat in st.session_state.chat_history:
            st.write("🤔 **You asked:**")
            st.write(chat["query"])
            st.write("🤖 **Assistant answered:**")
            st.write(chat["response"])
            st.divider()

if __name__ == "__main__":
    main() 