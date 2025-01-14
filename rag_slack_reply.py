#!/usr/bin/env python3
"""
rag_slack_repl.py

A unified RAG-style REPL that:
1) Uses a local SQLite database for messages
2) Embeds them in Pinecone
3) Allows command-line queries (REPL) against the vector store
4) Generates an LLM answer with relevant context
"""

# Standard library imports
import os
import sys
import sqlite3
from datetime import datetime
from dotenv import load_dotenv

# LangChain imports
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Scikit-learn imports for text similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import pinecone

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
    print("SQLite database initialized with sample data.")

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

    print(f"Retrieved {len(documents)} messages from the database.")
    return documents

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

def embed_and_store_documents(documents):
    """
    Creates a simple TF-IDF based retriever.
    """
    return SimpleRetriever(documents)

def run_repl(retriever):
    """
    A simple REPL loop that:
    - Takes a user query
    - Retrieves the most relevant docs
    - Adds the doc content as context
    - Calls the LLM to get a final answer
    - Repeats until user types 'exit'
    """
    # Create a ChatOpenAI LLM wrapper
    llm = ChatOpenAI(
        temperature=0.7,
        model_name="gpt-3.5-turbo",  # Standard OpenAI chat model
        openai_api_key=os.getenv("LANGCHAIN_API_KEY")
    )

    # The prompt template we can use:
    # You can customize this further, just make sure
    # to pass `query` and `context` in your final `.format()`
    prompt_template = PromptTemplate(
        input_variables=["query", "context"],
        template="{query}\n\nContext:\n{context}"
    )

    while True:
        user_query = input("\nEnter your query (or 'exit' to quit): ")
        if user_query.strip().lower() in ["exit", "quit"]:
            print("Exiting REPL.")
            break

        # Retrieve the most relevant docs from Pinecone
        docs = retriever.get_relevant_documents(user_query)
        # Build a context string out of the docs
        context_snippets = []
        for i, doc in enumerate(docs, start=1):
            snippet = (
                f"--- Document {i} ---\n"
                f"Channel: {doc.metadata.get('channel_name')} ({doc.metadata.get('channel_id')})\n"
                f"Author: {doc.metadata.get('author_name')} ({doc.metadata.get('author_id')})\n"
                f"Created At: {doc.metadata.get('created_at')}\n"
            )
            
            # Add thread context if available
            if doc.metadata.get('thread_name'):
                snippet += f"Thread: {doc.metadata.get('thread_name')}\n"
            
            # Add file information if available
            if doc.metadata.get('file_name'):
                snippet += (
                    f"File: {doc.metadata.get('file_name')} "
                    f"({doc.metadata.get('file_type', 'unknown type')})\n"
                )
            
            snippet += f"Content: {doc.page_content}\n"
            context_snippets.append(snippet)

        context_text = "\n".join(context_snippets)
        print("\nRetrieved context:")
        print(context_text)
        print("__________________________")

        # Format the prompt
        prompt_with_context = prompt_template.format(query=user_query, context=context_text)

        # Call the LLM
        answer = llm.predict(prompt_with_context)
        print("\nLLM Answer:")
        print(answer)

def main():
    # Load .env environment variables
    load_dotenv()

    # Initialize SQLite database with sample data
    init_sqlite_db()

    # 1) Load messages from DB
    documents = load_messages_from_db()
    if not documents:
        print("No documents found. Exiting.")
        return

    # 2) Embed & store in Pinecone, get a retriever
    retriever = embed_and_store_documents(documents)

    # 3) Run local REPL for user queries
    run_repl(retriever)

if __name__ == "__main__":
    main()
