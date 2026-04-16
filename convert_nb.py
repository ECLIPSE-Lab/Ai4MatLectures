import json
import glob
import os

files = [
    '/home/philipp/projects/Ai4MatLectures/autograd.ipynb',
    '/home/philipp/projects/Ai4MatLectures/linear_algebra.ipynb',
    '/home/philipp/projects/Ai4MatLectures/ndarray.ipynb',
    '/home/philipp/projects/Ai4MatLectures/pandas.ipynb'
]

for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    py_lines = []
    for cell in nb.get('cells', []):
        cell_source = ''.join(cell.get('source', []))
        if cell['cell_type'] == 'markdown':
            # Add markdown content as comments
            py_lines.append("\n" + "\n".join(f"# {line}" for line in cell_source.split('\n')) + "\n")
        elif cell['cell_type'] == 'code':
            # Add code content
            py_lines.append("\n" + cell_source + "\n")
            
    out_file = file.replace('.ipynb', '.py')
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(py_lines))

print("Conversion complete.")
