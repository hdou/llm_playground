from flask import Flask, render_template, request
import sys
from pathlib import Path
import re

this_file = Path(__file__).resolve()
rag_llm_dir = this_file.parents[1]

# Ensure the parent directory is in the path to import generate_answer
sys.path.append(str(rag_llm_dir))
from generating import generate_answer

app = Flask(__name__)

# Custom Jinja filter to bold text between pairs of ** on the same line
def bold_asterisks(value):
    # Only replace pairs of ** on the same line
    def replacer(match):
        return f"<b>{match.group(1)}</b>"
    return re.sub(r"\*\*(.+?)\*\*", replacer, value)

app.jinja_env.filters['bold_asterisks'] = bold_asterisks

@app.route('/', methods=['GET', 'POST'])
def index():
    answer = None
    question = ''
    if request.method == 'POST':
        question = request.form.get('question', '')
        if question:
            answer = generate_answer(question)
        print(answer)
    return render_template('index.html', answer=answer, question=question)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)