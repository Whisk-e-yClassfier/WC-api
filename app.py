from flask import Flask
import pickle

app = Flask(__name__)
model = pickle.load(open('./model/model.pkl', 'rb'))


@app.route("/")
def hello_world():
    return {"data": "<p>Hello, World!</p>"}
