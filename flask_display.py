from flask import Flask, render_template_string, jsonify, request
import threading

app = Flask(__name__)


data = {'word': '—', 'sentence': ''}

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/get_state')
def get_state():
    return jsonify(data)

@app.route('/update', methods=['POST'])
def update():
    global data
    d = request.get_json()
    if d:
        data['word'] = d.get('word', '—') or '—'
        data['sentence'] = d.get('sentence', '')
    return '', 204

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Sign Language Interpreter</title>
    <style>
        body{background:#0f172a;color:#e2e8f0;font-family:monospace;text-align:center;padding:20px}
        .card{background:#1e293b;padding:20px;margin:20px auto;max-width:600px;border-radius:16px}
        .word{font-size:2.5em;color:#00e896;word-break:break-all}
        .sentence{font-size:1.2em;background:#0f172a;padding:15px;border-radius:12px;text-align:left}
        button{background:#3b82f6;border:none;padding:8px 16px;margin:5px;border-radius:30px;color:white;cursor:pointer}
        .danger{background:#ef4444}
        .success{background:#10b981}
    </style>
</head>
<body>
    <h1>Sign Language Interpreter</h1>
    <div class="card">
        <div>Current Word</div>
        <div class="word" id="word">—</div>
        <button id="resetWord" class="danger">Clear word</button>
    </div>
    <div class="card">
        <div>Sentence</div>
        <div class="sentence" id="sentence"></div>
        <button id="resetAll" class="danger">Reset everything</button>
        <button id="speak" class="success">🔊 Speak</button>
    </div>
    <script>
        async function fetchData() {
            let r = await fetch('/get_state');
            let d = await r.json();
            document.getElementById('word').innerText = d.word || '—';
            document.getElementById('sentence').innerText = d.sentence || '';
        }
        async function resetWord() {
            await fetch('/reset_word', {method:'POST'});
            fetchData();
        }
        async function resetAll() {
            await fetch('/reset_all', {method:'POST'});
            fetchData();
        }
        function speak() {
            let t = document.getElementById('sentence').innerText;
            if (t) { let u = new SpeechSynthesisUtterance(t); window.speechSynthesis.cancel(); window.speechSynthesis.speak(u); }
        }
        document.getElementById('resetWord').onclick = resetWord;
        document.getElementById('resetAll').onclick = resetAll;
        document.getElementById('speak').onclick = speak;
        setInterval(fetchData, 200);
        fetchData();
    </script>
</body>
</html>
"""

@app.route('/reset_word', methods=['POST'])
def reset_word():
    global data
    data['word'] = '—'
    return '', 204

@app.route('/reset_all', methods=['POST'])
def reset_all():
    global data
    data['word'] = '—'
    data['sentence'] = ''
    return '', 204

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5001)