import numpy as np
import librosa
import requests

# input_audio_file_l= [f"server/{f}.ogg" for f in range(5)]
input_audio_file= "server/a.wav"
# for input_audio_file in input_audio_file_l:
audio = librosa.load(input_audio_file,sr=16000)[0].tolist()

# def silence(wave):
#     wave = np.array(wave)
#     thr = max(np.abs(wave))/100
#     idx =  np.where((wave>thr) | (wave<-thr))[0]
#     print(min(idx)/len(wave)*100,"  ",max(idx)/len(wave)*100)
#     res = wave[min(idx):max(idx)]
#     return res

# print (len(silence(audio)))

r=  requests.post(url = "https://speakar.herokuapp.com/8080",json={"samples":audio})
print(r.text)
print(r.json())