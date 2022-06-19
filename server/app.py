from numpy import int32
from onnx_model import ONNX_MODEL
from flask import Flask, request, jsonify
app = Flask(__name__)

model =  ONNX_MODEL("onnx_transformer.onnx")

# @app.route('/')
# def hello():    
#     return f'<h2> Hello Flask {model.classify("a.wav")} </h2>'




@app.route("/", methods=["POST"])
def hello_world():
    data = request.get_json(force=True)['samples']
    resp = {"status":200,"prediction":0,"prediction_prob":int(model.classify(data))}
    print(resp)
    return jsonify(resp)


# if __name__=="__main__":
#     app.run(debug=True)