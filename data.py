import numpy as np
from scipy.io.wavfile import read, write
from glob import glob
from os.path import join
from tqdm import tqdm as tq
from utils import *
from random import randint


def preprocess(filename):
    base_array = read(filename)[1].astype(np.float64)
    array = (base_array[:, 0] + base_array[:, 1])[:, np.newaxis]
    array /= (2 ** 14)
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
    vocals_array = preprocess(join(directory, "vocals.wav"))
    instru_array = mix_tracks([bass_array, drums_array, other_array])
    return instru_array, vocals_array


class Data:
    def __init__(self, dir="./data/DSD100_16kHz/Sources/", model=None):
        print("init")
        self.training_set = []
        self.validation_set = []

        for folder in tq(glob(dir + "Dev/*/")):
            self.training_set.append(get_instru_voice(folder))

        for folder in tq(glob(dir + "Test/*/")):
            self.validation_set.append(get_instru_voice(folder))

        if model is None:
            self.padding = 0
            print("No model was given, the padding is set to 0.")
        else:
            dummy_array = np.zeros((1, 100000, 1))
            output_size = model.predict(dummy_array).shape[1]
            self.padding = output_size // 2
            assert output_size % 2 == 0
            print("Padding found:", self.padding)

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

        mix, voice = next(self.training_generator(32, 20000))
        for i in range(mix.shape[0]):
            postprocess(mix[i], dir_checks + "batch_train_" + str(i) + "_mix.wav")
            postprocess(voice[i], dir_checks + "batch_train_" + str(i) + "_voice.wav")

        mix, voice = next(self.validation_generator(32, 20000))
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
            end = min((start + batch_size, len(list_tracks)))
            current_bs = end - start
            mix_array = np.zeros((current_bs, size_seq, 1))
            voice_array = np.zeros((current_bs, size_seq - 2 * self.padding, 1))

            for i in range(current_bs):
                idx = indices[start + i]
                max_start = list_tracks[idx][0].shape[0] - size_seq
                start_seq = randint(0, max_start)
                mix_array[i] = mix_tracks([list_tracks[idx][0][start_seq: start_seq + size_seq],
                                           list_tracks[idx][1][start_seq: start_seq + size_seq]])
                voice_array[i] = list_tracks[idx][1][start_seq + self.padding: start_seq + size_seq - self.padding]

            if first:
                print("Size of the mix array: %.2f Go" % (mix_array.nbytes / (2 ** 30)))
                first = False

            yield mix_array, voice_array

            if end == len(list_tracks):
                start = 0
            else:
                start = end

    def training_generator(self, batch_size, size_sequence):
        return self.make_generator(self.training_set, batch_size, size_sequence)

    def validation_generator(self, batch_size, size_sequence):
        return self.make_generator(self.validation_set, batch_size, size_sequence)


if __name__ == "__main__":
    da = Data("./data/DSD100_16kHz/Sources/")
