# import keras
from keras.callbacks import TerminateOnNaN, TensorBoard
import numpy as np
from data_utils import Data, postprocess, mix_tracks
from keras_models import get_gan
import os
from glob import glob
from tqdm import tqdm as tq
from utils import *
from keras_augmented import load_model


def predict_song(tup, model):
    list_ = [x[:16000 * 40] for x in tup]
    mix_array = mix_tracks(list_)[np.newaxis]
    return model.predict([mix_array, sample_latent_noise(mix_array.shape)])[0]


def sample_latent_noise(shape, min_=-1, max_=1):
    return (max_ - min_) * np.random.random_sample(shape) + min_


def generator_latent(data_generator):
    while 1:
        mix, voice = next(data_generator)
        yield [mix, sample_latent_noise(mix.shape)], voice


def train_generator(gan, data_generator_train):
    mix, voice = next(data_generator_train)
    latent_noise = sample_latent_noise(mix.shape)
    target = np.ones((mix.shape[0], 1), dtype=np.float32)
    return gan.train_on_batch([mix, latent_noise], target)


def train_discriminator(generator, discriminator, data_generator_train):
    mix1, voice1 = next(data_generator_train)
    mix2, voice2 = next(data_generator_train)

    assert mix1.shape[0] == mix2.shape[0]

    latent_noise = sample_latent_noise(mix1.shape)
    artificial_voice = generator.predict([mix1, latent_noise])

    mae = np.mean(np.abs(voice1 - artificial_voice))

    mix = np.concatenate([mix1, mix2], axis=0)
    voice = np.concatenate([artificial_voice, voice2])

    target = np.ones((mix.shape[0], 1), dtype=np.float32)
    half = target.shape[0] / 2
    assert int(half) == half
    target[:int(half)] = 0
    stats = discriminator.train_on_batch([mix, voice], target)
    stats.append(mae)
    return stats


def training_procedure(gan, generator, discriminator, data_generator_train, data_generator_val, nb_steps_gan, database):
    # Some setup on the first run. It's just for the timing.
    train_generator(gan, data_generator_train)
    train_discriminator(generator, discriminator, data_generator_train)
    result_dir = "./result/"
    mk(result_dir)

    generator.fit_generator(generator_latent(data_generator_train), 30, 2)
    song = predict_song(database.validation_set[-1], generator)
    postprocess(song, result_dir + "first.wav")

    for i in tq(range(nb_steps_gan)):

        stats_generator = train_generator(gan, data_generator_train)
        stats_discriminator = train_discriminator(generator, discriminator, data_generator_train)
        tq.write(str(stats_generator + stats_discriminator))

        if i % 10 == 0:
            song = predict_song(database.validation_set[-1], generator)
            postprocess(song, result_dir + "it_" + str(i) + ".wav")


def main():
    models_directory = "./models/"
    mk(models_directory)
    nb_steps_gan = 10000

    gan, generator, discriminator = get_gan(24, nb_blocks=2, conv_per_block=10, padding_generator="valid")
    generator.compile("adam", "mae")
    discriminator.compile("adam", "binary_crossentropy", metrics=["accuracy"])

    # We freeze the discriminator in the gan.
    discriminator.trainable = False
    gan.compile("adam", "binary_crossentropy", metrics=["accuracy"])

    database = Data(padding=generator.padding)

    training_procedure(gan, generator, discriminator,
                       database.training_generator(16, 10000),
                       database.validation_generator(16, 10000), nb_steps_gan, database)

    generator.save(models_directory + "model1.h5")
    load_model(models_directory + "model1.h5")  # to be sure that we can load it afterwards.

    postprocess(predict_song(database.training_set[0], generator), "./checks/train_predict.wav")
    postprocess(predict_song(database.validation_set[-1], generator), "./checks/val_predict.wav")


if __name__ == "__main__":
    main()
