import ast

def extract_structure(file_path, output_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        tree = ast.parse(f.read())

    lines = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            lines.append(f"Class: {node.name}")
            doc = ast.get_docstring(node)
            if doc:
                lines.append(f"    Docstring: {doc}")

            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    lines.append(f"    Function: {item.name}")
                    func_doc = ast.get_docstring(item)
                    if func_doc:
                        lines.append(f"        Docstring: {func_doc}")
        elif isinstance(node, ast.FunctionDef):
            lines.append(f"Function: {node.name}")
            doc = ast.get_docstring(node)
            if doc:
                lines.append(f"    Docstring: {doc}")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

    print(f"\n✅ Structure extracted and written to: {output_path}")


if __name__ == "__main__":
    file_path = input("Enter the path to the Python file: ").strip()
    output_path = input("Enter the path to the output TXT file: ").strip()
    extract_structure(file_path, output_path)
