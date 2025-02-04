# Handles data loading and preprocessing

import os
import uuid
import pandas as pd
import numpy as np
import wikipedia
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configuration & Constants

NEO4J_URL = "bolt://localhost:7687"
NEO4J_DATABASE = "neo4j"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "pass1234"
AUTH = (NEO4J_USER, NEO4J_PASSWORD)

RAW_DATA_DIR = "data/raw"
OUTPUT_DIR = "output"
WIKI_TOPIC = "World War I"
WIKI_OUTPUT_FILE = "World war 1.txt"

def get_wikipedia_content(topic: str) -> str:
    """Fetch and format Wikipedia content."""
    page = wikipedia.page(topic)
    return page.content.replace('==', '').replace('\n', ' ').strip()

def write_text_to_file(file_path: str, text: str) -> None:
    """Write text with UTF-8 encoding."""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Content written to {file_path}")

def load_data(file_path):
    # Get the Wikipedia page content
    text = get_wikipedia_content(WIKI_TOPIC)

    # Print to verify
    print(text)

    # Write to a file
    write_text_to_file(WIKI_OUTPUT_FILE, text)

def load_documents(loader_path: str) -> list:
    loader = DirectoryLoader(loader_path, show_progress=True)
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s) from {loader_path}")
    return documents

def split_documents(documents: list, chunk_size: int = 1500, chunk_overlap: int = 150) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False
    )
    pages = splitter.split_documents(documents)
    print(f"Number of pages after splitting: {len(pages)}")
    return pages

def documents_to_dataframe(pages: list) -> pd.DataFrame:
    page_data = [{
        'Page Content': page.page_content,
        'Source': page.metadata.get('source', 'N/A'),
        'chunk_id': uuid.uuid4().hex
    } for page in pages]
    
    df = pd.DataFrame(page_data)
    print(f"DataFrame created with {len(df)} rows")
    return df
