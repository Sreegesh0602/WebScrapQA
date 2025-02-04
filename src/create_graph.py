from langchain.chains import GraphCypherQAChain
from langchain.graphs import Neo4jGraph
from langchain.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM






def generate_chain():
    # ✅ Use ChatOllama from LangChain
    llm = ChatOllama(model="llama3", temperature=0)

    # ✅ Connect to Neo4j
    graph = Neo4jGraph(
        url="bolt://localhost:7687",
        username="neo4j",
        password="password"
    )

    # ✅ Step 1: Get existing labels and relationships
    schema_query = """
    CALL db.labels() YIELD label
    RETURN collect(label) AS labels;
    """
    labels_result = graph.query(schema_query)
    existing_labels = labels_result[0]["labels"]

    relationship_query = """
    CALL db.relationshipTypes() YIELD relationshipType
    RETURN collect(relationshipType) AS relationships;
    """
    relationships_result = graph.query(relationship_query)
    existing_relationships = relationships_result[0]["relationships"]

    print("Existing Labels:", existing_labels)
    print("Existing Relationships:", existing_relationships)

    # ✅ Step 2: Use an advanced Cypher prompt



    # MATCH (n:Node {name: "Germany"})-[r:RELATIONSHIP]->(relatedNodes)
    # RETURN n, collect(DISTINCT relatedNodes) AS relatedNodes, collect(DISTINCT r) AS relationships, collect(DISTINCT r.property_name) AS relationshipProperties




    cypher_prompt = PromptTemplate(
        input_variables=["query"],
        template = f"""
        You are a Neo4j Cypher expert. Generate an optimized Cypher query to retrieve information for the user's question.
        
        Ensure:
        - Use the existing labels: {existing_labels}
        - Use the existing relationships: {existing_relationships}
        - If a label or relationship doesn't exist, modify the query accordingly.
        - Do not include importance.
        - Get the relatedNode, relationship (`r`), relationship type (`r:type`), and any relevant properties of the relationship only if exists.

        User Query: {{query}}

        Cypher Query:
    """
    )
    llm = OllamaLLM(model="llama3")
    # ✅ Step 3: Use GraphCypherQAChain with Schema Awareness
    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        cypher_prompt=cypher_prompt,
        verbose=True,
        allow_dangerous_requests=True
    )
    return chain
