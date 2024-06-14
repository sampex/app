from flask import Flask, request, render_template
from model import generate_text

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['text']
    result = generate_text(input_text)
    return render_template('result.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)