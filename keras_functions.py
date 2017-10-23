import keras
import numpy as np
from keras.models import Model
from keras.layers import *


def get_model(filters, nb_blocks=3, conv_per_block=8):

    print("last conv is dilated by a factor of", 2**(conv_per_block-1))
    print("receptive field:", nb_blocks*(2**conv_per_block))

    def apply_conv(x, dilatation_rate):
        tanh_out = Conv1D(filters, 2,
                          dilation_rate=dilatation_rate,
                          activation="tanh",
                          padding="same")(x)
        sigm_out = Conv1D(filters, 2,
                          dilation_rate=dilatation_rate,
                          activation="sigmoid",
                          padding="same")(x)

        return Multiply()([tanh_out, sigm_out])

    def residual_block(x, dilatation_rate):
        out_act = apply_conv(x, dilatation_rate)

        res_x = Conv1D(filters, 1, padding="same")(out_act)
        res_x = Add()([x, res_x])

        skip_x = Conv1D(filters, 1, padding="same")(out_act)
        return res_x, skip_x


    input_tensor = Input(shape=(None, 1), name='input_part')

    x=input_tensor
    skip_connections = []

    for i in range(nb_blocks):
        for j in range(conv_per_block):
            x, skip_x = residual_block(x, 2**j)
            skip_connections.append(skip_x)

    out = Add()(skip_connections)
    out = Activation("relu")(out)
    out = Conv1D(filters, 3, padding="same", activation="relu")(out)
    out = Conv1D(1, 1, padding="same", activation="linear")(out)
    model = Model(input_tensor, out)
    return model