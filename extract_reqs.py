# save as extract_reqs.py
import ast, sys, pkg_resources

def get_imports(filename):
    with open(filename, "r") as f:
        tree = ast.parse(f.read())
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                imports.add(n.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])
    return imports

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_reqs.py your_file.py")
        sys.exit(1)

    filename = sys.argv[1]
    imports = get_imports(filename)

    for pkg in imports:
        try:
            dist = pkg_resources.get_distribution(pkg)
            print(f"{dist.project_name}=={dist.version}")
        except:
            print(f"# Not installed: {pkg}")
