"""Generate docs pages from example python files"""

import ast
import tokenize
from pathlib import Path
import mkdocs_gen_files
from itertools import chain

# Where to look for examples
example_folders = (Path("matfact/examples"), Path("hmm_synthetic/examples"))

def get_docstring_and_content(file):
    file_content = file.readlines()
    file.seek(0)
    docstring = ast.get_docstring(ast.parse(''.join(file_content)))

    tokens = tokenize.generate_tokens(file.readline)
    first_token = next(tokens)
    if not docstring or docstring not in first_token.string:
        raise Exception("Make sure the examples have a module docstring!")
    docstring_end = first_token.end[0]
    
    return docstring, ''.join(file_content[docstring_end:])
    

for example in chain(*map(lambda path: path.glob("*.py"), example_folders)):
    with open(example) as file:
        docstring, content = get_docstring_and_content(file) 
    with mkdocs_gen_files.open(f"examples/{example.stem}.md", "w") as f:
        print(f"# {docstring}", file=f)

        print(("```python\n"
               f"# {example.name}\n"
               "\n"
               f"{content}"
               "```"), file=f)
