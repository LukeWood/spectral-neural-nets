import tensorflow as tf


class FourierDomainConv2D(tf.keras.layers.Layer):
    def __init__(self, kernels, **kwargs):
        super(FourierDomainConv2D, self).__init__(**kwargs)
        self.kernels = kernels

    def build(self, input_shape):
        # print(input_shape) => (None, 150, 76, 6)
        # 6 chans, 3 real 3 imag
        h = input_shape[1]
        w = input_shape[2]
        kernel_shape = (h, w, self.kernels)

        self.real_kernel = self.add_weight(
            shape=kernel_shape,
            trainable=True
        )

        self.imag_kernel = self.add_weight(
            shape=kernel_shape,
            trainable=True
        )

        def compute_output_shape(self, input_shape):
            # channels last
            assert input_shape and len(input_shape) == 4
            assert input_shape[-1]
            output_shape = list(input_shape)
            output_shape[-1] = self.kernels*2
            return tuple(output_shape)

    def call(self, inputs):
        # (None, 150, 76, 3) (150, 76, 7)
        real_times_real = tf.einsum(
            'ijkc,jko->ijko', inputs[..., 0], self.real_kernel)
        imag_times_imag = tf.einsum(
            'ijkc,jko->ijko', inputs[..., 1], self.imag_kernel)
        y_real = real_times_real-imag_times_imag

        real_times_imag = tf.einsum(
            'ijkc,jko->ijko', inputs[..., 0], self.imag_kernel)
        imag_times_real = tf.einsum(
            'ijkc,jko->ijko', inputs[..., 1], self.real_kernel)
        y_imag = real_times_imag + imag_times_real

        return tf.stack([y_real, y_imag], axis=-1)
