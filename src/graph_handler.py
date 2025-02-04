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

def graph_prompt(input_text: str, metadata: dict = None, model: str = "mistral-openorca:latest") -> list:
    metadata = metadata or {}
    print("Input for graph prompt:", input_text)
    model_instance = ChatOllama(model=model, format="json")
    sys_prompt = (
        "You are a network graph maker who extracts terms and their relations from a given context. "
        "You are provided with a context chunk (delimited by ```) "
        "Your task is to extract the ontology of terms mentioned in the given context. These terms should represent "
        "the key concepts as per the context.\n"
        "Thought 1: While traversing through each sentence, think about the key terms mentioned in it. "
        "Terms may include object, entity, location, organization, person, condition, acronym, documents, service, concept, etc. "
        "Terms should be as atomistic as possible.\n\n"
        "Thought 2: Think about how these terms can have one on one relation with other terms. "
        "Terms that are mentioned in the same sentence or paragraph are typically related to each other. "
        "Terms can be related to many other terms.\n\n"
        "Thought 3: Find out the relation between each such related pair of terms.\n\n"
        "Format your output as a list of JSON objects. Each element should contain keys: "
        "'node_1', 'node_2', 'edge', 'entity', 'importance', and 'category'. "
        "Strictly respond in JSON format."
    )
    
    user_prompt = f"context: ```{input_text}```"
    full_prompt = f"{sys_prompt}\n\n{user_prompt}"
    
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
    results = results.dropna().reset_index(drop=True)
    graph_data = pd.DataFrame(results.to_list())
    return graph_data

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

def execute_with_fallback(query: str, chain: GraphCypherQAChain) -> str:
    try:
        response = chain.run(query)
        if response == "I don't know the answer.":
            raise ValueError("No result found in the graph.")
    except (ValueError, KeyError) as e:
        print("Graph query failed, falling back to Ollama:", e)
        fallback_model = ChatOllama(model="llama3.2:3b")
        response = fallback_model.invoke(query)
        return response.get("content", "No response.")
    return response
