max_duration = 1.0
from transformers import  Wav2Vec2FeatureExtractor,Wav2Vec2ForSequenceClassification #, AutoModel, AutoTokenizer, AudioClassificationPipeline
import torch

feature_extractor =  Wav2Vec2FeatureExtractor.from_pretrained( "facebook/wav2vec2-base")

def preprocess_function(example):
    
    audio_arrays = [x["array"] for x in example["audio"]]
    print(f"processing {len(audio_arrays)} example")
    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=feature_extractor.sampling_rate, 
        max_length=int(feature_extractor.sampling_rate * max_duration), 
        truncation=True, 
        return_tensor="pt"
    )
    inputs["input_values"]=torch.FloatTensor(inputs["input_values"][0])[None,:]
    return inputs

from datasets import Dataset,Features,Value,Audio,ClassLabel
nums = list(range(1,84+1))    
num_labels = len(nums)
num_classes = len(nums)
names = [str(i) for i in range(num_classes)]

file_name="a.wav"

d_train = {
    'audio':[file_name],}

ds_train = Dataset.from_dict(mapping=d_train, features=Features({
                                                    'audio':Audio()}))

tensor_input= preprocess_function(ds_train)
if __name__=="__main__":
    model = Wav2Vec2ForSequenceClassification.from_pretrained("Mahmoud1816Yasser/tmp_trainer",num_labels=num_labels)
    outputs = model(**tensor_input)
    print(torch.argmax(outputs[0],axis=1))