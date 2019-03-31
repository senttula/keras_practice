import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.models import Model
from keras.layers import Input, Lambda
from keras.layers import Reshape, Add, Layer
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.losses import mse
from keras.initializers import RandomUniform
from keras import backend as K

from keras.datasets import mnist

padding_size = 3
filters = 8

(x_train, _), (x_test, _) = mnist.load_data()
x_data = np.concatenate((x_train, x_test), axis=0)
x_data = x_data.astype(np.float32) / 255
size = x_data.shape[1]

x_data = np.expand_dims(x_data, -1)
y_data = np.zeros((x_data.shape[0], 1))

class mse_layer(Layer):
    def __init__(self, filters, **kwargs):
        super(mse_layer, self).__init__(**kwargs)
        self.filters = filters

    def build(self, input_shape):
        # bias size is image size * filtercount
        input_shape = list(input_shape)
        input_shape.pop(0)
        input_shape.pop(-1)
        input_shape[-1] = self.filters

        self.bias = self.add_weight(shape=input_shape,
                                    initializer=RandomUniform(minval=0, maxval=1),
                                    name='bias')
        self.built = True

    def call(self, inputs):
        # repeat dimensions to match each biasfilter to each window
        inputs = tf.tile(inputs, [1, 1, 1, 1, filters])
        biases = tf.tile(K.expand_dims(self.bias, -2), [1, 1, (padding_size*2)**2, 1])

        output = K.mean(K.square(inputs - biases), axis=[1, 2])

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], padding_size*2, self.filters)


class min_layer(Layer):
    def __init__(self, axis, keepdims=False, **kwargs):
        super(min_layer, self).__init__(**kwargs)
        self.axis = axis
        self.keepdims = keepdims

    def call(self, inputs):
        return K.min(inputs, axis=[self.axis], keepdims=self.keepdims)

    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        if self.keepdims:
            input_shape[self.axis] = 1
        else:
            input_shape.pop(self.axis)
        return tuple(input_shape)


def sliding_window(inputs):
    target = inputs

    window_shape = K.int_shape(target)[1:3]

    pattern = [[0, 0],
               [padding_size, padding_size],
               [padding_size, padding_size],
               [0, 0]]

    # Using tf pad insted of keras pad gives us the mode options
    target = tf.pad(inputs, pattern, mode="SYMMETRIC")

    start_indices = K.arange(padding_size * 2)
    repeated_elements = K.repeat_elements(start_indices, padding_size*2, axis=0)
    repeated_range = tf.tile(start_indices, [padding_size*2])

    repeated_elements = K.expand_dims(repeated_elements)
    repeated_range = K.expand_dims(repeated_range)
    # repeated_elements: 111222333
    # repeated_range:    123123123
    # concentate to get each coordinate
    start_coordinates = K.concatenate((repeated_elements, repeated_range))

    window_select = lambda index: target[:, index[0]:index[0]+window_shape[0], index[1]:index[1]+window_shape[1], :]

    windows = K.map_fn(window_select, start_coordinates, dtype=np.float32)

    # reshape for shape: (samples, imrows, imcolumns, mse_score, 1)
    windows = K.permute_dimensions(windows, (1, 2, 3, 0, 4))
    windows = Reshape((size, size, (padding_size * 2) ** 2, 1))(windows)
    return windows


inputs = Input((size, size, 1))
windows = Lambda(sliding_window)(inputs)

layer = mse_layer(filters)(windows)

layer = min_layer(1)(layer)  # global min over windows

# residual pass, forces every filter to update a bit
residual = Lambda(lambda x: K.sum(x, axis=1, keepdims=True)/20)(layer)

layer = min_layer(1, keepdims=True)(layer)
layer = Add()([layer, residual])

model = Model(inputs, layer)
model.summary()
opt = Adam(0.01)

model.compile(loss=mse, optimizer=opt)

# init weights to mean of small sample
start_size = 4
random_start_indices = np.random.choice(x_data.shape[0], start_size*filters)
random_samples = x_data[random_start_indices]
random_samples = np.transpose(random_samples, axes=[1, 2, 3, 0])
init_weigths = np.zeros((x_data.shape[1], x_data.shape[2], filters))

for i in range(filters):
    weigth_slice = np.mean(random_samples[:, :, :, start_size*i:start_size*(i+1)], axis=3)
    init_weigths[:, :, i] = weigth_slice[:, :, 0]

model.set_weights([init_weigths])

earlystop = EarlyStopping(monitor='loss', patience=5)

model.fit(x_data, y_data,
            batch_size=256,
            epochs=100,
            verbose=2,
            callbacks=[earlystop]
          )

weigths = model.get_weights()[0]
for i in range(filters):
    ww = weigths[:, :, i]
    plt.subplot(201+filters*5+i)
    plt.imshow(ww, cmap='gray')
    plt.axis('off')
plt.savefig('filters.png')

