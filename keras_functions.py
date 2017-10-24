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


def get_generator_model(filters, nb_blocks=3, conv_per_block=8):
    print("last conv is dilated by a factor of", 2 ** (conv_per_block - 1))
    print("receptive field:", nb_blocks * (2 ** conv_per_block))

    def apply_conv(x, dilatation_rate):
        tanh_out = Conv1D(filters, 3,
                          dilation_rate=dilatation_rate,
                          activation="tanh")(x)
        sigm_out = Conv1D(filters, 3,
                          dilation_rate=dilatation_rate,
                          activation="sigmoid")(x)

        return Multiply()([tanh_out, sigm_out])

    def residual_block(x, dilatation_rate):
        out_act = apply_conv(x, dilatation_rate)

        res_x = Conv1D(filters, 1)(out_act)
        res_x = Lambda(add_centered, lambda x: x[-1])([x, res_x])

        skip_x = Conv1D(filters, 1)(out_act)
        return res_x, skip_x

    input_tensor = Input(shape=(None, 1), name='input_part')

    x = input_tensor
    skip_connections = []

    for i in range(nb_blocks):
        for j in range(conv_per_block):
            x, skip_x = residual_block(x, 2 ** j)
            skip_connections.append(skip_x)

    out = Lambda(add_centered, lambda x: x[-1])(skip_connections)
    out = Activation("relu")(out)
    out = Conv1D(filters, 3, activation="relu")(out)
    out = Conv1D(1, 1, activation="linear")(out)
    model = Model(input_tensor, out)

    print(model.layers[-1])

    return model
