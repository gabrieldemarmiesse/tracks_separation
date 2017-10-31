import keras
import numpy as np
from keras.models import Model
from keras.layers import *
import keras.backend as K
from keras_augmented import AddCentered, MeanCentered


def freeze_model(model):
    model.trainable = False
    for layer in model.layers:
        layer.trainable = False


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
    out_act = apply_conv(x, dilatation_rate, filters, padding=padding)

    res_x = Conv1D(filters, 1, padding=padding)(out_act)

    if padding == "valid":
        layer = AddCentered()
    elif padding == "same":
        layer = Average()
    else:
        print(padding)
        assert False
    res_x = layer([x, res_x])

    skip_x = Conv1D(filters, 1, padding=padding)(out_act)
    return res_x, skip_x


def wavenet(input_tensor, filters, nb_blocks=3, conv_per_block=8, padding="valid"):
    x = Conv1D(filters, 1, activation="relu")(input_tensor)
    skip_connections = []

    for i in range(nb_blocks):
        for j in range(conv_per_block):
            x, skip_x = residual_block(x, 2 ** j, filters, padding=padding)
            skip_connections.append(skip_x)

    if padding == "valid":
        layer = MeanCentered()
    elif padding == "same":
        layer = Average()
    else:
        print(padding)
        assert False

    out = layer(skip_connections)
    out = Activation("relu")(out)
    out = Conv1D(filters, 3, activation="relu", padding=padding)(out)
    out = Conv1D(1, 1, activation="linear")(out)
    return out


def get_generator(filters, nb_blocks=3, conv_per_block=8, padding="valid"):
    print("last conv is dilated by a factor of", 2 ** (conv_per_block - 1))
    print("receptive field:", nb_blocks * (2 ** conv_per_block))

    input_mix = Input(shape=(None, 1), name='input_mix_gen')
    input_latent = Input(shape=(None, 1), name='input_latent_gen')

    full_input = Concatenate(axis=-1)([input_mix, input_latent])

    out = wavenet(full_input, filters, nb_blocks, conv_per_block, padding)  # Shape is (BS, timesteps, 1)
    model = Model([input_mix, input_latent], out)

    print(model.layers[-1])

    # We find the padding induced.
    dummy_array = np.zeros((1, 10000, 1))
    output_size = model.predict([dummy_array, dummy_array]).shape[1]
    model.padding = (dummy_array.shape[1] - output_size) // 2
    assert output_size % 2 == 0
    print("Padding found:", model.padding)

    return model


def get_discriminator(filters, nb_blocks=3, conv_per_block=8, padding="valid", padding_gen=0):
    print("last conv is dilated by a factor of", 2 ** (conv_per_block - 1))
    print("receptive field:", nb_blocks * (2 ** conv_per_block))

    input_mix = Input(shape=(None, 1), name='input_mix_disc')
    input_voice = Input(shape=(None, 1), name='input_voice_disc')

    # It's possible that input_voice has a smaller size than input_mix.
    if padding_gen == 0:
        input_voice_pad = input_voice
    else:
        input_voice_pad = ZeroPadding1D(padding_gen)(input_voice)

    full_input = Concatenate(axis=-1)([input_mix, input_voice_pad])

    out = wavenet(full_input, filters, nb_blocks, conv_per_block, padding)
    out = GlobalAveragePooling1D()(out)
    out = Activation("sigmoid")(out)  # Shape is (BS, 1)

    return Model([input_mix, input_voice], out)


def get_gan(filters, nb_blocks, conv_per_block, padding_generator="valid", padding_discriminator="valid"):
    generator = get_generator(filters, nb_blocks, conv_per_block, padding=padding_generator)
    discriminator = get_discriminator(filters, nb_blocks, conv_per_block,
                                      padding=padding_discriminator,
                                      padding_gen=generator.padding)

    # We need two inputs
    input_mixed = Input(shape=(None, 1), name='input_mixed_gan')
    input_latent = Input(shape=(None, 1), name='input_latent_gan')

    predicted_voice = generator([input_mixed, input_latent])
    disciminator_decision = discriminator([input_mixed, predicted_voice])

    gan = Model([input_mixed, input_latent], disciminator_decision)
    return gan, generator, discriminator
