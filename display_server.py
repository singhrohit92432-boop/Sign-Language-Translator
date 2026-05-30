from flask import Flask, jsonify, render_template_string
import json
import os

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head><title>SignLens Display</title>
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
</div>
<div class="card">
    <div>Sentence</div>
    <div class="sentence" id="sentence"></div>
    <button id="speak" class="success">🔊 Speak</button>
    <button id="reset" class="danger">Reset (clear file)</button>
</div>
<script>
    async function fetchData() {
        const res = await fetch('/data');
        const d = await res.json();
        document.getElementById('word').innerText = d.word || '—';
        document.getElementById('sentence').innerText = d.sentence || '';
    }
    async function resetData() {
        await fetch('/reset', {method:'POST'});
        fetchData();
    }
    function speak() {
        const t = document.getElementById('sentence').innerText;
        if(t) { const u = new SpeechSynthesisUtterance(t); window.speechSynthesis.cancel(); window.speechSynthesis.speak(u); }
    }
    document.getElementById('speak').onclick = speak;
    document.getElementById('reset').onclick = resetData;
    setInterval(fetchData, 200);
    fetchData();
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/data')
def data():
    try:
        with open('sign_data.json', 'r') as f:
            d = json.load(f)
    except:
        d = {'word': '', 'sentence': ''}
    return jsonify(d)

@app.route('/reset', methods=['POST'])
def reset():
    with open('sign_data.json', 'w') as f:
        json.dump({'word': '', 'sentence': ''}, f)
    return '', 204

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)