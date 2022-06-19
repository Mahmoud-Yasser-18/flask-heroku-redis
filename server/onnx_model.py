from unicodedata import name
import onnxruntime
import numpy as np
#from scipy.io import wavfile
import librosa

from transformers import  Wav2Vec2FeatureExtractor

# compute ONNX Runtime output prediction

class ONNX_MODEL: 
    def __init__(self,model_path) -> None:
        self.max_duration = 1
        self.feature_extractor =  Wav2Vec2FeatureExtractor.from_pretrained( "facebook/wav2vec2-base")
        self.ort_session = onnxruntime.InferenceSession(model_path)

    def classify(self,input_audio_file):
        if type(input_audio_file)==str: 
            ort_inputs = {self.ort_session.get_inputs()[0].name: 
                
                    self.feature_extractor(
                        librosa.load(input_audio_file,sr=None)[0], 
                        sampling_rate=self.feature_extractor.sampling_rate, 
                        max_length=int(self.feature_extractor.sampling_rate * self.max_duration), 
                        truncation=True, 
                        padding="max_length"
                        )["input_values"][0][None,:]

                }
        else: 
            ort_inputs = {self.ort_session.get_inputs()[0].name: 
                
                    self.feature_extractor(
                        input_audio_file, 
                        sampling_rate=self.feature_extractor.sampling_rate, 
                        max_length=int(self.feature_extractor.sampling_rate * self.max_duration), 
                        truncation=True, 
                        padding="max_length"
                        )["input_values"][0][None,:]

                }

        return np.argmax(self.ort_session.run(None, ort_inputs)[0],axis=1)[0]
2
if __name__ == "__main__":
    model = ONNX_MODEL("onnx_transformer.onnx")
    print(model.classify("a.wav"))
    print(model.classify("b.wav"))