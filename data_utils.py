import numpy as np
from scipy.io.wavfile import read, write
from glob import glob
from os.path import join
from tqdm import tqdm as tq
from utils import *
from random import randint
import pickle


def preprocess(filename):
    base_array = read(filename)[1].astype(np.float64)
    array = (base_array[:, 0] + base_array[:, 1])[:, np.newaxis]
    array /= (2 ** 15)
    np.clip(array, -1, 1, array)
    return array.astype(np.float32)


def postprocess(array, filename):
    new_array = array * (2 ** 14)
    np.around(new_array, out=new_array)
    new_array = new_array.astype(np.int16)
    new_array = np.broadcast_to(new_array, (new_array.shape[0], 2))
    write(filename, 16000, new_array)


def mix_tracks(list_arrays):
    result = sum(list_arrays)
    np.clip(result, -1, 1, result)
    return result


def get_instru_voice(directory):
    bass_array = preprocess(join(directory, "bass.wav"))
    drums_array = preprocess(join(directory, "drums.wav"))
    other_array = preprocess(join(directory, "other.wav"))
    instru_array = mix_tracks([bass_array, drums_array, other_array])
    vocals_array = preprocess(join(directory, "vocals.wav"))
    return instru_array, vocals_array


class Data:
    def __init__(self, dir="./data/DSD100_16kHz/Sources/", padding=0):
        print("init")
        cache_dir = "./data/cache/"
        mk(cache_dir)
        pickle_file = cache_dir + "train.p"
        try:
            self.training_set = pickle.load(open(pickle_file, "rb"))
        except FileNotFoundError:
            self.training_set = []
            for folder in tq(glob(dir + "Dev/*/")):
                self.training_set.append(get_instru_voice(folder))
            pickle.dump(self.training_set, open(pickle_file, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

        pickle_file = cache_dir + "val.p"
        try:
            self.validation_set = pickle.load(open(pickle_file, "rb"))
        except FileNotFoundError:
            self.validation_set = []
            for folder in tq(glob(dir + "Test/*/")):
                self.validation_set.append(get_instru_voice(folder))
            pickle.dump(self.validation_set, open(pickle_file, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

        self.padding = padding

        self.sanity_check()

    def sanity_check(self):
        """ Save wav files to be sure that we did the operations correctly."""
        dir_checks = "./checks/"
        mk(dir_checks)
        for file in os.listdir(dir_checks):
            os.remove(join(dir_checks, file))

        postprocess(self.training_set[0][0], dir_checks + "instru_train.wav")
        postprocess(self.training_set[0][1], dir_checks + "voice_train.wav")
        postprocess(mix_tracks(self.training_set[0]), dir_checks + "mix_train.wav")

        postprocess(self.validation_set[0][0], dir_checks + "instru_val.wav")
        postprocess(self.validation_set[0][1], dir_checks + "voice_val.wav")
        postprocess(mix_tracks(self.validation_set[0]), dir_checks + "mix_val.wav")

        print("lenght of training set:", len(self.training_set))
        print("lenght of validation set:", len(self.validation_set))

        mix, voice = next(self.training_generator(8, 20000))
        for i in range(mix.shape[0]):
            postprocess(mix[i], dir_checks + "batch_train_" + str(i) + "_mix.wav")
            postprocess(voice[i], dir_checks + "batch_train_" + str(i) + "_voice.wav")

        mix, voice = next(self.validation_generator(8, 20000))
        for i in range(mix.shape[0]):
            postprocess(mix[-1], dir_checks + "batch_val_" + str(i) + "_mix.wav")
            postprocess(voice[-1], dir_checks + "batch_val_" + str(i) + "_voice.wav")

        print("All tests done.")

    def make_generator(self, list_tracks, batch_size, size_seq):
        indices = np.arange(len(list_tracks))
        np.random.shuffle(indices)
        start = 0

        first = True

        while 1:
            end = start + batch_size
            mix_array = np.zeros((batch_size, size_seq, 1))
            voice_array = np.zeros((batch_size, size_seq - 2 * self.padding, 1))

            for i in range(batch_size):
                idx = indices[(start + i) % len(indices)]
                max_start = list_tracks[idx][0].shape[0] - size_seq
                start_seq = randint(0, max_start)
                mix_array[i] = mix_tracks([list_tracks[idx][0][start_seq: start_seq + size_seq],
                                           list_tracks[idx][1][start_seq: start_seq + size_seq]])
                voice_array[i] = list_tracks[idx][1][start_seq + self.padding: start_seq + size_seq - self.padding]

            if first:
                print("Size of the mix array: %.2f Mo" % (mix_array.nbytes / (2 ** 20)))
                first = False

            yield mix_array, voice_array

            start = end % len(indices)

    def training_generator(self, batch_size, size_sequence):
        return self.make_generator(self.training_set, batch_size, size_sequence)

    def validation_generator(self, batch_size, size_sequence):
        return self.make_generator(self.validation_set, batch_size, size_sequence)


if __name__ == "__main__":
    da = Data("./data/DSD100_16kHz/Sources/")
