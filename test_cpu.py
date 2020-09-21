
import h5py
import numpy as np
from keras.models import load_model

model = load_model('/mnt/intStorage/deeplearning/myDP/logs/fulltrain/weights-improvement-15.hdf5')


res=[]
def prediction(test1, test2):

    pred_labels = model.predict([test1, test2])
    for item in zip(pred_labels):
        if np.round(item) == 1:
            print(item, "Item interacted")
            result = 1
        else:
            print(item)
            result = 0
        res.append([result, "%.8f"% item[0][0]])
    return (result, "%.8f"% item[0][0])

with h5py.File("/mnt/intStorage/deeplearning/myDP/regions_matrics.h5", 'r') as hf:
    s1_e1 = np.array(hf.get('region_s1_e1'))
    s2_e2 = np.array(hf.get('region_s2_e2'))

s1_e1=s1_e1.transpose(0, 2, 1)
s2_e2=s2_e2.transpose(0, 2, 1)



prediction(s1_e1, s2_e2)

print(res[1:5])
print(len(res))
