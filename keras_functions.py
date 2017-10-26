import keras
import numpy as np
from keras.models import Model
from keras.layers import *
import keras.backend as K


def add_centered(list_of_tensors):
    """ Sum a list of tensors of different sizes. The annoying part is
        to be sure that they are centered when they are summed."""

    cst_2 = K.constant(2, dtype="int32")
    result = list_of_tensors[-1]
    smallest_shape = K.shape(result)[1]

    for tensor in list_of_tensors[:-1]:
        diff = K.shape(tensor)[1] - smallest_shape
        print(K.dtype(diff))
        pad = diff // cst_2
        print(K.dtype(pad))
        result += tensor[:, pad:-pad, :]

    return result


def apply_conv(x, dilatation_rate, filters, padding="valid"):
    tanh_out = Conv1D(filters, 3,
                      dilation_rate=dilatation_rate,
                      activation="tanh",
                      padding=padding)(x)
    sigm_out = Conv1D(filters, 3,
                      dilation_rate=dilatation_rate,
                      activation="sigmoid",
                      padding=padding)(x)

    return Multiply()([tanh_out, sigm_out])


def residual_block(x, dilatation_rate, filters, padding="valid"):
    out_act = apply_conv(x, dilatation_rate)

    res_x = Conv1D(filters, 1, padding=padding)(out_act)
    res_x = Lambda(add_centered, lambda x: x[-1])([x, res_x])

    skip_x = Conv1D(filters, 1, padding=padding)(out_act)
    return res_x, skip_x


def get_generator(filters, nb_blocks=3, conv_per_block=8):
    print("last conv is dilated by a factor of", 2 ** (conv_per_block - 1))
    print("receptive field:", nb_blocks * (2 ** conv_per_block))

    input_tensor = Input(shape=(None, 1), name='input_part')

    x = input_tensor
    skip_connections = []

    for i in range(nb_blocks):
        for j in range(conv_per_block):
            x, skip_x = residual_block(x, 2 ** j, filters)
            skip_connections.append(skip_x)

    out = Lambda(add_centered, lambda x: x[-1])(skip_connections)
    out = Activation("relu")(out)
    out = Conv1D(filters, 3, activation="relu")(out)
    out = Conv1D(1, 1, activation="linear")(out)
    model = Model(input_tensor, out)

    print(model.layers[-1])

    return model


def average_all(tensor):
    x = K.mean(tensor, axis=(1, 2))
    return K.expand_dims(x, 1)


def average_layer(tensor):
    return Lambda(average_all, lambda x: (x[0], 1))(tensor)


def get_discriminator(filters, nb_blocks=3, conv_per_block=8):
    print("last conv is dilated by a factor of", 2 ** (conv_per_block - 1))
    print("receptive field:", nb_blocks * (2 ** conv_per_block))

    input_0 = Input(shape=(None, 1), name='input_0')
    input_1 = Input(shape=(None, 1), name='input_1')

    x = input_0
    skip_connections_0 = []

    for i in range(nb_blocks):
        for j in range(conv_per_block):
            x, skip_x = residual_block(x, 2 ** j, filters, padding="same")
            skip_connections_0.append(skip_x)

    features_extractor = Model(input_0, skip_connections_0)

    skip_connections_1 = features_extractor(input_1)

    out_0 = [average_layer(a) for a in skip_connections_0]
    out_0 = Average()(out_0)

    out_1 = [average_layer(a) for a in skip_connections_1]
    out_1 = Average()(out_1)

    out_sub = Subtract()([out_0, out_1])
    out = Activation("sigmoid")(out_sub)
    return Model([input_0, input_1], out)


def get_full_model(filters, nb_blocks, conv_per_block):
    generator = get_generator(filters, nb_blocks, conv_per_block)
    discriminator = get_discriminator(filters, nb_blocks, conv_per_block)

    # We need two inputs
    input_mixed = Input(shape=(None, 1), name='input_mixed')

    input_true_voice = Input(shape=(None, 1), name='input_true_voice')

    predicted_voice = generator(input_mixed)

    predicted_original = discriminator([predicted_voice, input_true_voice])

    full_model = Model([input_mixed, input_true_voice], [predicted_voice, predicted_original])
    return full_model, generator, discriminator
