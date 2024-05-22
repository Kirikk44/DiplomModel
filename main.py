from doctest import debug
from flask import Flask, jsonify, request
from DatasetHandler import DatasetHandler
from Predictor import Predictor
from flask import json
from assistant2 import preprocess_text
import webview
import config

app = Flask(__name__)

@app.route("/")
def index():
    print ("Base url without port " + request.remote_addr)
    print ("Base url with port " + request.host_url)
    response = app.response_class(
        response=json.dumps(["success"]),
        status=200,
        mimetype='application/json'
        )
    return response

@app.route("/predicte", methods=['GET'])
def predicte():
    uid = request.args.get('uid', None)
    moc = request.args.get('moc', "1")
    mod_name = request.args.get('mod_name', None)
    
    if uid:
        if mod_name:
            predictor.set_mod(mod_name)
        uids = predictor.predict(uid, moc=moc)
        response = app.response_class(
        response=json.dumps(uids),
        status=200,
        mimetype='application/json'
        )
        return  response
    else:
        response = app.response_class(
        response=json.dumps("uid отсутствует"),
        status=400,
        mimetype='application/json'
        )
        return 'response'

@app.route("/update", methods=['POST'])
def update_docs():
    data = request.get_json()

    if 'docs' in data:
        strings_array = data['docs']
        if isinstance(strings_array, list):
            predictor.updateDocs(strings_array)
        else:
            return jsonify({'error': 'Значение поля strings_array должно быть массивом'}), 400
    else:
        return jsonify({'error': 'Отсутствует поле strings_array в JSON данных'}), 400

if __name__ == "__main__":
    
    dataset_handler = DatasetHandler()
    predictor = Predictor(dataset_handler=dataset_handler)
    app.run(debug=True, host="0.0.0.0", port=5000)