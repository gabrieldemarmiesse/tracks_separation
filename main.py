# import keras
import numpy as np
from scipy.io.wavfile import read, write
from resampy import resample
from time import time
from keras_functions import get_generator, get_full_model
from keras.callbacks import TerminateOnNaN, ModelCheckpoint, TensorBoard
import data

nb_steps_MAE = 10000
nb_steps_GAN = 10000



def training_procedure(full_model, generator, discriminator, data_generator_train, data_generator_val):


    # Callbacks:
    generator.fit_generator(data_generator_train, 30, callbacks=[TerminateOnNaN(), TensorBoard()])

    # We need to first train the generator, using only MAE, then, we move on to using the discriminator too.
    generator.fit_generator(data_generator_train, 1000, epochs=10, validation_data=data_generator_val)


full_model, generator, discriminator = get_full_model(16, 3, 2)

generator.compile("adam", "mae")

database = data.Data(model=generator)

training_procedure(full_model, generator, discriminator,
                   database.training_generator(32,10000),
                   database.validation_generator(32,10000))
