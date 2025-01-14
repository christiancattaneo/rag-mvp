# RAG Slack Reply

A RAG (Retrieval-Augmented Generation) system for querying chat messages using LangChain and OpenAI.

## Features

- Local SQLite database for storing sample messages
- TF-IDF based retrieval system for finding relevant messages
- OpenAI GPT integration for generating responses
- Interactive REPL interface for querying messages

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install langchain langchain-openai python-dotenv scikit-learn
   ```
4. Create a `.env` file with your OpenAI API key:
   ```
   LANGCHAIN_API_KEY=your-api-key-here
   ```

## Usage

Run the script:
```bash
python rag_slack_reply.py
```

The script will:
1. Create a local SQLite database with sample messages
2. Start a REPL interface where you can enter queries
3. Show relevant messages and AI-generated responses

Example queries:
- "What philosophical views have been shared?"
- "What advice has been shared about fitness?"
- "What have people said about darkness or night?"
- "Who has talked about business topics?"

Type 'exit' or 'quit' to end the session.

## Sample Data

The system comes with sample messages from fictional characters:
- Patrick Bateman (American Psycho)
- Tyler Durden (Fight Club)
- Bruce Wayne (Batman)
- Bane (Batman)
- Arnold Schwarzenegger

Messages are organized into channels:
- general
- fitness
- business
- vigilante-stuff
- philosophy 