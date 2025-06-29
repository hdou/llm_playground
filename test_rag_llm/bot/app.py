from flask import Flask, render_template, request
import sys
from pathlib import Path

this_file = Path(__file__).resolve()
rag_llm_dir = this_file.parents[1]

# Ensure the parent directory is in the path to import generate_answer
sys.path.append(str(rag_llm_dir))
from generating import generate_answer

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    answer = None
    question = ''
    if request.method == 'POST':
        question = request.form.get('question', '')
        if question:
            answer = generate_answer(question)
    return render_template('index.html', answer=answer, question=question)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)