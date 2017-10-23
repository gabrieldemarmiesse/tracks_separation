from glob import glob
from tqdm import tqdm as tq
from scipy.io.wavfile import read, write
from resampy import resample
new_rate = 16000
path = "./data/DSD100_16kHz/Sources/*/*/*.wav"

for file in tq(glob(path)):
    rate, array = read(file)
    new_array = resample(array,rate, new_rate, axis=0)
    write(file, new_rate, new_array)
