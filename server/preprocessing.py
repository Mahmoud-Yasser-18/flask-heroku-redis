import numpy as np
 
def silence(wave):
  wave = np.array(wave)
  thr = max(np.abs(wave))/5
  idx =  np.where((wave>thr) | (wave<-thr))[0]
  res = wave[min(idx):max(idx)]
  return res


