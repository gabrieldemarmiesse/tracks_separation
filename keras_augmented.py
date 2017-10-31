from keras import backend as K
from keras.engine.topology import Layer
from keras import models
import inspect
import sys


def add_centered(list_of_tensors):
    """ Sum a list of tensors of different sizes. The annoying part is
        to be sure that they are centered when they are summed."""

    cst_2 = K.constant(2, dtype="int32")
    result = list_of_tensors[-1]
    smallest_shape = K.shape(result)[1]

    for tensor in list_of_tensors[:-1]:
        diff = K.shape(tensor)[1] - smallest_shape
        pad = diff // cst_2
        result += tensor[:, pad:-pad, :]

    return result


class AddCentered(Layer):
    def __init__(self, **kwargs):
        super(AddCentered, self).__init__(**kwargs)

    def build(self, input_shape):
        super(AddCentered, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return add_centered(x)

    def compute_output_shape(self, input_shape):
        return input_shape[-1]


class MeanCentered(Layer):
    def __init__(self, **kwargs):
        super(MeanCentered, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MeanCentered, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return add_centered(x) / len(x)

    def compute_output_shape(self, input_shape):
        return input_shape[-1]


def get_classes():
    clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)
    dic = {}
    for name, class_ in clsmembers:
        dic[name] = class_
    return dic


def load_model(file):
    return models.load_model(file, custom_objects=get_classes())


if __name__ == "__main__":
    print(get_classes())
