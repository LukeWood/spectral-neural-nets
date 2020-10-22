import tensorflow as tf
import tensorflow.keras.initializers as initializers
from .base import KernelFourierConvolutionBase


class Gaussian2DKernelFourierLayer(KernelFourierConvolutionBase):

    def __init__(self, filters, **kwargs):
        super(Gaussian2DKernelFourierLayer, self).__init__(filters)

    def add_kernel_weights(self, h, w):
        # , x0, y0, a, b, c, s0, s1, s2
        weights_shape = (self.filters, 8)
        self.real_terms = self.add_weight(
            shape=weights_shape,
            initializer=initializers.RandomNormal(mean=0, stddev=0.05),
            trainable=True
        )

        # One kernel set for row, one for column entry
        self.imag_terms = self.add_weight(
            shape=weights_shape,
            initializer=initializers.RandomNormal(mean=0, stddev=0.05),
            trainable=True
        )

    # f(x, y) = A*exp(-(a(s1*(x-x0))^2) + 2b(x-x0)(y-y0) + c((s2y-y0))^2)
    def expand_one_kernel(self, kernel, h, w):
        terms = None
        if kernel == 'real':
            terms = self.real_terms
        else:
            terms = self.imag_terms
        res = []

        cols = tf.range(h, dtype=tf.float32)/h - 0.5
        rows = tf.range(w, dtype=tf.float32)/w

        for filter in range(self.filters):
            variables = terms[filter]
            x0 = variables[0]
            y0 = variables[1]
            a = variables[2]
            b = variables[3]
            c = variables[4]
            s0 = variables[5]
            s1 = variables[6]
            s2 = variables[7]

            x_minus_x0 = s1*(cols - x0)
            y_minus_y0 = s2*(rows - y0)

            x_minus_x0_repeated = tf.expand_dims(y_minus_y0, axis=0)
            y_minus_y0_repeated = tf.expand_dims(x_minus_x0, axis=1)
            x_minus_x0_repeated = tf.repeat(x_minus_x0_repeated, h, axis=0)
            y_minus_y0_repeated = tf.repeat(y_minus_y0_repeated, w, axis=1)

            inner_term = tf.math.multiply(
                x_minus_x0_repeated, y_minus_y0_repeated)
            final = s0*tf.math.exp(
                -a*tf.math.square(x_minus_x0_repeated) +
                (2*b*inner_term) +
                -c*tf.math.square(y_minus_y0_repeated)
            )
            res.append(final)
        result = tf.stack(res, axis=-1)
        return result

    def expand_kernel(self, h, w):
        return self.expand_one_kernel('real', h, w), \
            self.expand_one_kernel('imag', h, w)
        # expand on row
        # expand on col
        # concat together into a single axis
        # apply kernel on an element-wise basis
