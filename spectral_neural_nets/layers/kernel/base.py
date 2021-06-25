import tensorflow as tf


class ParametricFourierConvolutionBase(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(ParametricFourierConvolutionBase, self).__init__(**kwargs)
        self.filters = filters

    def add_filter_weights(self, h, w):
        raise 'add_filter_weights must be defined when subclassing ' + \
            'ParametricFourierConvolutionBase'

    def expand_filter(self, h, w):
        raise 'expand_filter must be defined when subclassing ' + \
            'ParametricFourierConvolutionBase'

    def compute_output_shape(self, input_shape):
        # channels last
        assert input_shape and len(input_shape) == 4
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.filters*2
        return tuple(output_shape)

    def build(self, input_shape):
        h = input_shape[1]
        w = input_shape[2]
        self.h = h
        self.w = w
        self.add_filter_weights(h, w)

    def call(self, inputs, training=False):
        # Update the filter based on the new weights when training
        h = inputs.shape[1]
        w = inputs.shape[2]
        real_W, imag_W = self.expand_filter(h, w)

        # Proceed like we did with the normal fourier conv layers
        real_input = inputs[..., 0]
        imag_input = inputs[..., 1]

        real_times_real = tf.einsum('ijkl,jko->ijko', real_input, real_W)
        imag_times_imag = tf.einsum('ijkl,jko->ijko', imag_input, imag_W)
        y_real = real_times_real-imag_times_imag

        real_times_imag = tf.einsum('ijkl,jko->ijko', real_input, imag_W)
        imag_times_real = tf.einsum('ijkl,jko->ijko', imag_input, real_W)
        y_imag = real_times_imag + imag_times_real
        return tf.stack([y_real, y_imag], axis=-1)
