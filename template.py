import os

def create_project_structure(base_path=r"./"):
    directories = [
        "src",
        "config",
        "scripts",
        "data",
        "output"
    ]
    
    files = {
        "src/__init__.py": "",
        "src/data_loader.py": "# Handles data loading and preprocessing\n\nimport pandas as pd\nimport numpy as np\n\ndef load_data(file_path):\n    return pd.read_csv(file_path)\n",
        "src/text_processor.py": "# Processes text data\n\nimport re\n\ndef clean_text(text):\n    return re.sub(r'[^a-zA-Z0-9 ]', '', text)\n",
        "src/graph_handler.py": "# Handles graph-related operations\n\nfrom langchain.graphs import Neo4jGraph\n\ndef connect_to_graph():\n    return Neo4jGraph()\n",
        "src/ai_model.py": "# AI model integration\n\nfrom langchain_ollama import ChatOllama\n\ndef get_ai_response(prompt):\n    model = ChatOllama()\n    return model.run(prompt)\n",
        "config/settings.py": "# Configuration settings\n\nDB_URI = 'your_database_uri_here'\n",
        "scripts/run_pipeline.py": "# Script to run the full pipeline\n\nfrom src.data_loader import load_data\nfrom src.text_processor import clean_text\n\nif __name__ == '__main__':\n    print('Pipeline running...')\n",
        "requirements.txt": "pandas\nnumpy\nlangchain\nNeo4j\nollama\n",
        "README.md": "# Project Overview\n\nThis project processes text, interacts with a graph database, and uses AI models."
    }
    
    os.makedirs(base_path, exist_ok=True)
    
    for directory in directories:
        os.makedirs(os.path.join(base_path, directory), exist_ok=True)
    
    for file_path, content in files.items():
        full_path = os.path.join(base_path, file_path)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
    
    print(f"Project structure created at {base_path}")

if __name__ == "__main__":
    create_project_structure()
