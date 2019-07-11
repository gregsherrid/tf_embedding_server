from flask import Flask
from flask_cors import CORS

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

import json

use_module_url = "https://tfhub.dev/google/universal-sentence-encoder/2" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]
embed = hub.Module(use_module_url)

tf.logging.set_verbosity(tf.logging.ERROR)

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
	return "Hello TF Embeddings!"

@app.route("/encode")
def encode():
	params = get_request_params()

	params["texts"]

	with tf.Session() as session:
		session.run([tf.global_variables_initializer(), tf.tables_initializer()])
		message_embeddings = session.run(embed(messages))

		embeddings = np.array(message_embeddings).tolist()

	return json.dumps({ "results": embeddings })

# Merges request.args and request.form
def get_request_params():
	merged_params = {}

	merged_params.update(request.args.to_dict())
	merged_params.update(request.form.to_dict())

	json_data = request.get_json()
	if json_data:
		merged_params.update(json_data)

	return merged_params

if __name__ == "__main__":
	# Only for debugging while developing
	app.run(host='0.0.0.0', debug=True, port=3333)
