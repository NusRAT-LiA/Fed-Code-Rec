"""
AST-based Code Chunking using Tree-sitter
Extracts functions and classes from Python code
"""

from tree_sitter import Language, Parser, Query, QueryCursor
import tree_sitter_python as tspython
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Python parser
PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)


def extract_functions_and_classes(code: str, filename: str = "unknown") -> List[Tuple[str, str, str]]:
    """
    Extract functions and classes from Python code using AST parsing.
    """
    chunks = []

    try:
        # Parse source
        tree = parser.parse(bytes(code, "utf8"))
        root_node = tree.root_node

        # Build query
        query = Query(PY_LANGUAGE, """
            (function_definition
                name: (identifier) @func.name
            ) @func.def

            (class_definition
                name: (identifier) @class.name
            ) @class.def
        """)

        # Create cursor
        cursor = QueryCursor(query)

        # Get captures as dict
        captures_dict = cursor.captures(root_node)

        # Process functions
        func_names = captures_dict.get('func.name', [])
        func_defs = captures_dict.get('func.def', [])
        for name_node, def_node in zip(func_names, func_defs):
            current_name = code[name_node.start_byte:name_node.end_byte]
            chunk_code = code[def_node.start_byte:def_node.end_byte]
            chunks.append(("function", current_name, chunk_code))

        # Process classes
        class_names = captures_dict.get('class.name', [])
        class_defs = captures_dict.get('class.def', [])
        for name_node, def_node in zip(class_names, class_defs):
            current_name = code[name_node.start_byte:name_node.end_byte]
            chunk_code = code[def_node.start_byte:def_node.end_byte]
            chunks.append(("class", current_name, chunk_code))

        logger.info(f"Extracted {len(chunks)} chunks from {filename}")

    except Exception as e:
        logger.error(f"Error parsing {filename}: {e}")
        return []

    return chunks


def chunk_code_file(filepath: str) -> List[Tuple[str, str, str]]:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        return extract_functions_and_classes(code, filepath)

    except Exception as e:
        logger.error(f"Error reading {filepath}: {e}")
        return []


def get_chunk_summary(chunk_type: str, name: str, code: str, max_lines: int = 5) -> str:
    lines = code.strip().split('\n')
    preview = '\n'.join(lines[:max_lines])
    return f"[{chunk_type.upper()}] {name}\n{preview}\n..."


if __name__ == "__main__":
    test_code = """
def fibonacci(n):
    '''Calculate fibonacci number'''
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class Calculator:
    '''Simple calculator class'''
    def __init__(self):
        self.result = 0
    
    def add(self, x, y):
        return x + y
"""

    chunks = extract_functions_and_classes(test_code)

    print(f"Found {len(chunks)} chunks:")
    for chunk_type, name, code in chunks:
        print(f"\n{chunk_type}: {name}")
        print(f"Lines: {len(code.split(chr(10)))}")
        print(get_chunk_summary(chunk_type, name, code))