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

    # ✅ Step 2: Advanced Cypher Prompt without Keyword Dependency
    cypher_prompt = PromptTemplate(
    input_variables=["query"],
    template=f"""
    You are a Neo4j Cypher expert. Generate an optimized Cypher query to retrieve information for the user's question.

    Ensure and follow the following guidelines strictly:
    - Use the existing labels: {existing_labels}
    - Use the existing relationships: {existing_relationships}
    - Ensure that all nodes and relationships involved are correctly bound.
    - Always return the relationship variable (`r`) and explicitly define it in the MATCH clause.
    - Use `COLLECT(DISTINCT ...)` to avoid duplicates.
    - Dont miss out where clause to filter the results based on the user's query.
    - Use the 'OR','AND','IS', 'CONTAINS', 'not' 'Null' operator within the where clause not with the match clause
    - you should use the IS NOT NULL syntax to check if a property exists.
    - you need to explicitly define the relationship variable r in the MATCH clause.
    - you cannot introduce new variables or '*' directly in the WHERE clause when they are part of pattern expressions

    The query structure should be:

    ```
    MATCH (n:Node)-[r:RELATIONSHIP]->(relatedNode)
    WHERE n.name CONTAINS '{{query}}' OR relatedNode.name CONTAINS '{{query}}'
    RETURN n, relatedNode, type(r)  , properties(r)  
    ```

    User Query: {{query}}

    Cypher Query:
"""
)
    llm = OllamaLLM(model="llama3")

    # ✅ Step 3: Use GraphCypherQAChain without keyword dependency
    chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    cypher_prompt=cypher_prompt,
    verbose=True,
    include_run_info=True,  # ✅ Ensures we capture raw execution info
    return_direct=True,  # ✅ Forces returning the raw Cypher query result
    allow_dangerous_requests=True
)

    return chain

