import os
import sys
import streamlit as st
import numpy as np
import pandas as pd

# Ensure the parent directory is in sys.path to resolve modules like config.settings
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config.settings import RAW_DATA_DIR, OUTPUT_DIR, WIKI_TOPIC, WIKI_OUTPUT_FILE
from src.text_processor import get_wikipedia_content, write_text_to_file
from src.data_loader import load_documents, split_documents, documents_to_dataframe
from src.graph_handler import df_to_graph, initialise_neo4j_schema, insert_dataframe_to_neo4j
from src.create_graph import generate_chain

# 🌟 --- Streamlit UI ---
st.set_page_config(page_title="Graph Query Pipeline", page_icon="🕵️‍♂️", layout="wide")

# 🌟 Sidebar
with st.sidebar:
    # st.image("https://upload.wikimedia.org/wikipedia/commons/8/8e/Neo4j-logo.png", width=200)
    st.markdown("### ⚡ **Graph Query Pipeline**")
    st.info("This app extracts Wikipedia content, builds a Neo4j graph, and translates natural language into Cypher queries.")
    # st.markdown("#### 🔗 **Quick Links**")
    # st.markdown("[📖 Neo4j Docs](https://neo4j.com/docs/)")
    # st.markdown("[🛠 LangChain](https://python.langchain.com/)")
    # st.markdown("[📊 Streamlit](https://streamlit.io/)")

# 🌟 Header Section
st.title("🕵️ Graph Query Pipeline")
st.subheader("🔎 Extract insights from Wikipedia, build a knowledge graph, and explore relationships!")

# ✅ Step 1: Get Wikipedia content
with st.expander("📖 Fetch Wikipedia Content"):
    if not os.path.exists(WIKI_OUTPUT_FILE):
        st.info("Fetching content from Wikipedia... ⏳")
        wiki_text = get_wikipedia_content(WIKI_TOPIC)
        write_text_to_file(WIKI_OUTPUT_FILE, wiki_text)
        st.success("✅ Wikipedia content saved successfully.")
    else:
        st.success("✅ Wikipedia content already exists.")

# ✅ Step 2: Load and process documents
with st.expander("📂 Load & Process Documents"):
    documents = load_documents(RAW_DATA_DIR)
    pages = split_documents(documents)
    df_chunks = documents_to_dataframe(pages)

    st.success("✅ Documents successfully loaded and split into pages!")
    st.write("📊 **Data Preview:**")
    st.dataframe(df_chunks.head(5))

# ✅ Step 3: Create or load graph DataFrame
with st.expander("📊 Generate Graph DataFrame"):
    graph_csv = os.path.join(OUTPUT_DIR, "graph.csv")

    if not os.path.exists(graph_csv):
        st.info("Generating graph DataFrame... ⏳")
        df_graph = df_to_graph(df_chunks.head(1), model="llama3")
        df_graph.replace("", np.nan, inplace=True)
        df_graph["count"] = 4  # Default count value

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        df_graph.to_csv(graph_csv, index=False)
        df_chunks.to_csv(os.path.join(OUTPUT_DIR, "chunks.csv"), index=False)

        st.success("✅ Graph data saved!")
    else:
        st.success("✅ Graph CSV already exists, loading data...")
        df_graph = pd.read_csv(graph_csv)

    st.write("📌 **Graph Data Preview:**")
    st.dataframe(df_graph.head(5))

# ✅ Step 4: Initialize Neo4j Schema & Insert Data
with st.expander("🛠 Initialize & Insert Data into Neo4j"):
    st.info("Setting up Neo4j schema & inserting data...")
    initialise_neo4j_schema()
    insert_dataframe_to_neo4j(df_graph)
    st.success("✅ Data successfully inserted into Neo4j!")

# ✅ Step 5: Query the Graph using Natural Language
st.markdown("---")
st.subheader("🔍 Query the Knowledge Graph")

query = st.text_input("✍️ **Enter your graph query:**", "")

if st.button("🚀 Run Query"):
    with st.spinner("Processing your query... ⏳"):
        chain = generate_chain()
        response = chain.invoke(query)

        # Extract full context from the response
        full_context = response.get("full_context") or response.get("result", [])

        # ✅ Display Results in a Better UI
        if isinstance(full_context, list):
            st.subheader("📊 **Extracted Relationships**")
            for item in full_context:
                n = item.get("n", {})
                relatedNode = item.get("relatedNode", {})
                relationshipType = item.get("type(r)", "Unknown")
                relationshipProperties = item.get("properties(r)", {})

                with st.expander(f"🔗 {n.get('name', 'Unknown')} ➝ {relatedNode.get('name', 'Unknown')}"):
                    st.markdown(f"**🔹 Entity:** `{n.get('name', 'Unknown')}` ({n.get('category', 'N/A')})")
                    st.markdown(f"**🔹 Relationship:** `{relationshipType}`")
                    st.markdown(f"**🔹 Related To:** `{relatedNode.get('name', 'Unknown')}` ({relatedNode.get('category', 'N/A')})")
                    st.markdown(f"**🔹 Relationship Details:** `{relationshipProperties.get('relationship', 'No details')}`")

        elif isinstance(full_context, str):
            st.subheader("📜 **Summary Response**")
            st.write(full_context)
        else:
            st.error("⚠️ Unexpected response format!")
            st.write(response)

st.markdown("---")
st.markdown("👨‍💻 **Built with ❤️ using Streamlit, Neo4j & LangChain** 🚀")
