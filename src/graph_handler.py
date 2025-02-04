import json
import sys
import numpy as np
import pandas as pd
from neo4j import GraphDatabase
from langchain.chains import GraphCypherQAChain
from langchain_ollama import ChatOllama
from config.settings import NEO4J_URL, NEO4J_DATABASE, NEO4J_USER, NEO4J_PASSWORD, AUTH

# Handles graph-related operations

from langchain.graphs import Neo4jGraph

def connect_to_graph():
    return Neo4jGraph()

import pandas as pd
import json

def graph_prompt(input_text: str, metadata: dict = None, model: str = "mistral-openorca:latest") -> list:
    metadata = metadata or {}

    model_instance = ChatOllama(model=model, format="json")
    sys_prompt = (
        "You are a network graph maker who extracts terms and their relations from a given context. "
        "You are provided with a context chunk (delimited by ```). Your task is two-fold: "
        "first, extract the ontology of key terms in the context along with their relationships; "
        "second, automatically generate a corresponding Neo4j Cypher query that retrieves these nodes and their relationships. \n"
        "Important: When generating the Cypher query, ensure every relationship variable (e.g., 'r') used in the pattern is properly defined. "
        "For example, use patterns like [r:RELATIONSHIP] instead of omitting the variable. \n"
        "Instructions: \n"
        "  - Identify relevant nodes (terms) and potential relationships between them. \n"
        "  - Formulate a Cypher query that uses MERGE or MATCH clauses appropriately, ensuring that any relationship (like 'r') is explicit and defined. \n"
        "  - Strictly output your final response in valid JSON format as a list of objects. Each object should include keys such as 'node_1', 'node_2', 'edge', 'entity', 'importance', and 'category'. \n"
        "Example format:\n"
        "   [{\n"
        '       "node_1": "Entity A",\n'
        '       "node_2": "Entity B",\n'
        '       "edge": "Relation description",\n'
        '       "entity": "Concept type",\n'
        '       "importance": 4,\n'
        '       "category": "Category type"\n'
        "   }, ...]\n"
        "Make sure the Cypher query within your JSON response properly defines all used variables."
    )
    user_prompt = f"context: ```{input_text}```"
    full_prompt = f"{sys_prompt}\n\n{user_prompt}"
    print("Getting graph prompt response...")
    response = model_instance.invoke(full_prompt).content
    try:
        result = json.loads(response)
        print("Graph prompt result:", result)
        return result
    except json.JSONDecodeError as e:
        print("Failed to parse JSON:", e)
        return None

def df_to_graph(df: pd.DataFrame, model: str = "llama3.2:3B") -> pd.DataFrame:
    
    results = df.apply(
        lambda row: graph_prompt(row['Page Content'], {"chunk_id": row['chunk_id']}, model),
        axis=1
    )
    
    # Dynamically handle both 'nodes' and 'ontology'
    nodes_data = []
    for result in results:
        if result:
            # Check if 'nodes' or 'ontology' is in the result
            if 'nodes' in result:
                nodes_data.append(result['nodes'])
            elif 'ontology' in result:
                nodes_data.append(result['ontology'])
    
    # Flatten the list of lists and convert to a DataFrame
    flattened_nodes = [item for sublist in nodes_data for item in sublist]
    
    graph_data = pd.DataFrame(flattened_nodes)
    
    # Clean the DataFrame by dropping any NaN values and resetting the index
    df_cleaned = graph_data.dropna().reset_index(drop=True)
    
    # Save the cleaned data to a CSV
    df_cleaned.to_csv("graph_data.csv", index=False)
    
    return df_cleaned

def initialise_neo4j_schema():
    with GraphDatabase.driver(NEO4J_URL, auth=AUTH) as driver:
        with driver.session(database=NEO4J_DATABASE) as session:
            print("Schema initialized successfully.")

def insert_dataframe_to_neo4j(df: pd.DataFrame) -> None:
    with GraphDatabase.driver(NEO4J_URL, auth=AUTH) as driver:
        with driver.session(database=NEO4J_DATABASE) as session:
            for _, row in df.iterrows():
                node_1_props = {
                    'name': row['node_1'],
                    'entity': row['entity'],
                    'importance': row['importance'],
                    'category': row['category']
                }
                node_2_props = {
                    'name': row['node_2'],
                    'entity': row['entity'],
                    'importance': row['importance'],
                    'category': row['category']
                }
                edge_props = {
                    'type': row['edge'],
                    'relationship': row['edge'],
                    'importance': row['importance'],
                    'category': row['category']
                }
                query = """
                MERGE (n1:Node {name: $node1_name})
                SET n1 += $node1_props
                MERGE (n2:Node {name: $node2_name})
                SET n2 += $node2_props
                MERGE (n1)-[r:RELATIONSHIP {type: $edge_type}]->(n2)
                SET r += $edge_props
                """
                session.run(query,
                            node1_name=node_1_props['name'],
                            node1_props=node_1_props,
                            node2_name=node_2_props['name'],
                            node2_props=node_2_props,
                            edge_type=edge_props['type'],
                            edge_props=edge_props)
            print("Data inserted into Neo4j successfully.")

def execute_with_fallback(query: str, chain) -> str:
    try:
        response = chain.invoke(query)
        if response == "I don't know the answer.":
            return "No result found in the graph."
        
    except Exception as e:
        print("Graph query failed, falling back to Ollama:", e)
        return "No result found in the graph. Fallback to Ollama."
        fallback_model = ChatOllama(model="llama3")
        response = fallback_model.invoke(query)
        return response.get("content", "No response.")
    return response
