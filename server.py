from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
	return "Hello, TF Embeddings!"

if __name__ == "__main__":
        # Only for debugging while developing
        app.run(host='0.0.0.0', debug=True, port=3333)
