from flask import Flask
from onnx_model import ONNX_MODEL
app = Flask(__name__)

model =  ONNX_MODEL("onnx_transformer.onnx")

@app.route('/')
def hello():
    
    return f'<h2> Hello Flask {model.classify("a.wav")} </h2>'
