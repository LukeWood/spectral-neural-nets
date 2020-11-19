import tensorflow as tf
import tensorflow.keras.initializers as initializers

from .base import ParametricFourierConvolutionBase


class LinearFourierConvolution(ParametricFourierConvolutionBase):
    def __init__(self, filters, order=1, **kwargs):
        super(LinearFourierConvolution, self).__init__(filters, **kwargs)
        self.order = order

    def add_filter_weights(self, h, w):
        weights_shape = (2, self.filters, self.order+1)
        # One filter set for row, one for column entry
        self.real_terms = self.add_weight(
            shape=weights_shape,
            initializer=initializers.RandomNormal(mean=0, stddev=0.005),
            trainable=True
        )

        # One filter set for row, one for column entry
        self.imag_terms = self.add_weight(
            shape=weights_shape,
            initializer=initializers.RandomNormal(mean=0, stddev=0.005),
            trainable=True
        )

    def expand_filter(self, h, w):
        terms = self.real_terms
        row_terms = terms[0]
        col_terms = terms[1]

        cols = tf.range(h, dtype=tf.float32)/h - 0.5
        rows = tf.range(w, dtype=tf.float32)/w - 0.5

        res = []
        for filter in range(self.filters):
            coeffs_row = tf.unstack(row_terms[filter])
            coeffs_col = tf.unstack(col_terms[filter])

            r_res = tf.math.polyval(coeffs_row, rows)
            c_res = tf.math.polyval(coeffs_col, cols)
            r_res = tf.expand_dims(r_res, axis=0)
            c_res = tf.expand_dims(c_res, axis=1)
            r_res = tf.repeat(r_res, h, axis=0)
            c_res = tf.repeat(c_res, w, axis=1)
            res.append(r_res + c_res)
        real_filter = tf.stack(res, axis=-1)

        terms = self.imag_terms
        row_terms = terms[0]
        col_terms = terms[1]
        res = []
        for filter in range(self.filters):
            coeffs_row = tf.unstack(row_terms[filter])
            coeffs_col = tf.unstack(col_terms[filter])

            r_res = tf.math.polyval(coeffs_row, rows)
            c_res = tf.math.polyval(coeffs_col, cols)
            r_res = tf.expand_dims(r_res, axis=0)
            c_res = tf.expand_dims(c_res, axis=1)
            r_res = tf.repeat(r_res, h, axis=0)
            c_res = tf.repeat(c_res, w, axis=1)
            res.append(r_res + c_res)
        imag_filter = tf.stack(res, axis=-1)
        return real_filter, imag_filter
