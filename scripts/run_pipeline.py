import os
import sys

# Ensure the parent directory is in sys.path to resolve modules like config.settings
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np
import pandas as pd

from config.settings import RAW_DATA_DIR, OUTPUT_DIR, WIKI_TOPIC, WIKI_OUTPUT_FILE
from src.text_processor import get_wikipedia_content, write_text_to_file
from src.data_loader import load_documents, split_documents, documents_to_dataframe
from src.graph_handler import df_to_graph, initialise_neo4j_schema, insert_dataframe_to_neo4j, execute_with_fallback

def main():
    # Step 1: Get content from Wikipedia
    wiki_text = get_wikipedia_content(WIKI_TOPIC)
    write_text_to_file(WIKI_OUTPUT_FILE, wiki_text)
    
    # Step 2: Load raw documents and split into pages
    documents = load_documents(RAW_DATA_DIR)
    pages = split_documents(documents)
    
    # Step 3: Create DataFrame from pages and generate unique chunk IDs
    df_chunks = documents_to_dataframe(pages)
    
    # Process a subset for testing
    df_graph = df_to_graph(df_chunks.head(20), model="llama3.2:3B")
    df_graph.replace("", np.nan, inplace=True)
    df_graph.dropna(subset=["node_1", "node_2", "edge"], inplace=True)
    df_graph["count"] = 4  # default value
    
    # Save intermediate outputs
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_graph.to_csv(os.path.join(OUTPUT_DIR, "graph.csv"), index=False)
    df_chunks.to_csv(os.path.join(OUTPUT_DIR, "chunks.csv"), index=False)
    print("CSV files exported.")
    
    # Step 4: Initialise Neo4j and insert graph data
    initialise_neo4j_schema()
    insert_dataframe_to_neo4j(df_graph)
    
    # Step 5: Execute a sample query on the graph
    from langchain.graphs import Neo4jGraph
    from langchain.chains import GraphCypherQAChain
    from langchain_ollama import ChatOllama

    graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="pass1234")
    chain = GraphCypherQAChain.from_llm(
        llm=ChatOllama(model="llama3:8b", temperature=0),
        graph=graph,
        verbose=True,
        allow_dangerous_requests=True
    )
    
    sample_query = "How Machine Guns and Artillery are related"
    result = execute_with_fallback(sample_query, chain)
    print("Query Result:", result)

if __name__ == "__main__":
    main()
