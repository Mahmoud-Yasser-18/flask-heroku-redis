from onnx_model import ONNX_MODEL
from flask import Flask, request, jsonify
app = Flask(__name__)

model =  ONNX_MODEL("onnx_transformer.onnx")

@app.route('/',methods=["GET"])
def hello():    
    return f'<h2> Hello Flask {model.classify("a.wav")} </h2>'





@app.route("/", methods=["POST"])
def hello_world():
    data = request.get_json(force=True)['samples']
    class_id, probability =model.classify(data)
    resp = {"status":200,"prediction":int(class_id),"prediction_prob":float(probability)}
    return jsonify(resp)


# if __name__=="__main__":
#     app.run(debug=True)