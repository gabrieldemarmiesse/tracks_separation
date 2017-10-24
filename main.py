# import keras
import numpy as np
from scipy.io.wavfile import read, write
from resampy import resample
from time import time
from keras_functions import get_generator_model


def preprocess(array):
    if len(array.shape):
        array = np.expand_dims(array, 0)
    new_array = np.sum(array, axis=2, keepdims=True)
    new_array = np.broadcast_to(new_array, (10, new_array.shape[1], 1))
    return new_array/(2**14)


def postprocess(array):
    array = array[0]
    array *= 2**14
    array = np.clip(array, -(2**14), 2**14)
    array = array.astype(np.int16)
    return np.broadcast_to(array, (array.shape[0], 2))


folder = "./data/DSD100_16kHz/Sources/Dev/051 - AM Contra - Heart Peripheral/"

in_data_bass = read(folder + "bass.wav")[1].astype(np.float32)
in_data_drums = read(folder + "drums.wav")[1].astype(np.float32)
in_data_other = read(folder + "other.wav")[1].astype(np.float32)
in_data_vocals = read(folder + "vocals.wav")[1].astype(np.float32)

in_data = in_data_bass + in_data_drums + in_data_other + in_data_vocals
in_data = preprocess(in_data)

out_data = in_data_vocals
out_data = preprocess(out_data)

model = get_generator_model(256, 2, 2)

dummy_size = 30000
dummy_input = np.zeros((1,dummy_size,1))
padding = (dummy_size - model.predict(dummy_input).shape[1])//2
print(padding)



for i in range(out_data.shape[1]):
    if out_data[0,i,0] !=0:
        break
print(i)
model.compile("adam", "mae")
print(np.sum(np.abs(out_data)))
print(np.max(np.abs(out_data)))
model.fit(in_data[:,i:i+20000], out_data[:,i+padding:i+20000-padding], 1, 2)
model.save("./data/models/first_model.h5")
