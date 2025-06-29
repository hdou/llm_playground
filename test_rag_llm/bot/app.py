from flask import Flask, render_template, request, redirect, url_for
import sys
from pathlib import Path
import re
from flask import session

this_file = Path(__file__).resolve()
rag_llm_dir = this_file.parents[1]

# Ensure the parent directory is in the path to import generate_answer
sys.path.append(str(rag_llm_dir))
from generating import generate_answer

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Needed for session

# Custom Jinja filter to bold text between pairs of ** on the same line
def bold_asterisks(value):
    # Only replace pairs of ** on the same line
    def replacer(match):
        return f"<b>{match.group(1)}</b>"
    return re.sub(r"\*\*(.+?)\*\*", replacer, value)

app.jinja_env.filters['bold_asterisks'] = bold_asterisks

@app.route('/', methods=['GET', 'POST'])
def index():
    if 'history' not in session:
        session['history'] = []
    history = session['history']
    answer = None
    question = ''
    if request.method == 'POST':
        question = request.form.get('question', '')
        if question:
            answer = generate_answer(question)
            history.append({'question': question, 'answer': answer})
            session['history'] = history
    return render_template('index.html', history=history, answer=answer, question=question)

@app.route('/clear', methods=['POST'])
def clear():
    session['history'] = []
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)