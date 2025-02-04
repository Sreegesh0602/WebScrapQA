# Processes text data

import re
import wikipedia

def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9 ]', '', text)

def get_wikipedia_content(topic: str) -> str:
    """Fetch and format Wikipedia content."""
    page = wikipedia.page(topic)
    return page.content.replace('==', '').replace('\n', ' ').strip()

def write_text_to_file(file_path: str, text: str) -> None:
    """Write text with UTF-8 encoding."""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Content written to {file_path}")
