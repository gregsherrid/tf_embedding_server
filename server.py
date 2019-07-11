from flask import Flask, request, jsonify
from flask_cors import CORS

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

import json

tf.logging.set_verbosity(tf.logging.ERROR)
use_module_url = "https://tfhub.dev/google/universal-sentence-encoder/2" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]

g = tf.Graph()
with g.as_default():
	text_input = tf.placeholder(dtype=tf.string, shape=[None])
	embed = hub.Module(use_module_url)
	embedded_text = embed(text_input)
	init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
g.finalize()

session = tf.Session(graph=g)
session.run(init_op)

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
	return "Hello TF Embeddings!"

@app.route("/encode")
def encode():
	params = get_request_params()

	text = params.get("text")
	if not text:
		return jsonify({ "error": "Please provide 'text' argument." })

	result = session.run(text, feed_dict={ text_input: ["Hello world"] })
	return jsonify({ "result": result })

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
