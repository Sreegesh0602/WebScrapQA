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
from src.create_graph import generate_chain

def main():
    # Step 1: Get content from Wikipedia
    if not os.path.exists(WIKI_OUTPUT_FILE):
        print("Fetching content from Wikipedia...")
        wiki_text = get_wikipedia_content(WIKI_TOPIC)
        write_text_to_file(WIKI_OUTPUT_FILE, wiki_text)

        print("Wikipedia content saved.")
        # Step 2: Load raw documents and split into pages
        documents = load_documents(RAW_DATA_DIR)
        pages = split_documents(documents)


        print("Documents loaded and split into pages.")
        # Step 3: Create DataFrame from pages and generate unique chunk IDs
        df_chunks = documents_to_dataframe(pages)


        print("DataFrame created from pages.")
# Process a subset for testing
    if not os.path.exists(os.path.join(OUTPUT_DIR, "graph.csv")):
        df_graph = df_to_graph(df_chunks.head(1), model="llama3")
        df_graph.replace("", np.nan, inplace=True)
        # df_graph.dropna(subset=["node_1", "node_2", "edge"], inplace=True)
        df_graph["count"] = 4  # default value
    
    # Save intermediate outputs
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        df_graph.to_csv(os.path.join(OUTPUT_DIR, "graph.csv"), index=False)
        df_chunks.to_csv(os.path.join(OUTPUT_DIR, "chunks.csv"), index=False)
        print("CSV files exported.")
    else:
        df_graph = pd.read_csv(os.path.join(OUTPUT_DIR, "graph.csv"))
        
    # Step 4: Initialise Neo4j and insert graph data
    initialise_neo4j_schema()
    insert_dataframe_to_neo4j(df_graph)
    
    # Step 5: Execute a sample query on the graph

    chain = generate_chain()
    # sample_query = "tell about Hitler"
    # result = execute_with_fallback(sample_query, chain)
    # print("Query Result:", result)
    # ✅ Invoke the Chain
    query = "tell about Japan and China"
    response = chain.invoke(query)  # OR response = chain.run(query)

    # ✅ Extract Full Context (Avoid Iterating Over String)
    full_context = response.get("full_context") or response.get("result", [])

    # ✅ Check if full_context is a list (structured data) or just a plain string
    if isinstance(full_context, list):  
        print("\n🔹 **Extracted Relationships** 🔹\n")
        for item in full_context:
            n = item.get("n", {})
            relatedNode = item.get("relatedNode", {})
            relationshipType = item.get("type(r)", "Unknown")  # Adjusted key
            relationshipProperties = item.get("properties(r)", {})

            print(f"🌍 Entity: {n.get('name', 'Unknown')} ({n.get('category', 'N/A')})")
            print(f"🔗 Relationship: {relationshipType}")
            print(f"➡️ Related To: {relatedNode.get('name', 'Unknown')} ({relatedNode.get('category', 'N/A')})")
            print(f"📜 Relationship Details: {relationshipProperties.get('relationship', 'No details')}")
            print("-" * 50)

    elif isinstance(full_context, str):  
        print("📝 **Summary Response:**", full_context)  # Handle plain text responses properly

    else:
        print("⚠️ Unexpected response format:", response)


if __name__ == "__main__":
    main()
