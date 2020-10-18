import tensorflow as tf
from tensorflow.keras.layers import Lambda


def do_from_complex(x):
    real = tf.math.real(x)
    imag = tf.math.imag(x)
    zed = tf.stack([real, imag], axis=-1)
    return zed


def do_to_complex(x):
    real = x[..., 0]
    imag = x[..., 1]
    result = tf.complex(real, imag)
    return result


from_complex = Lambda(lambda x: do_from_complex(x), name='from_complex')
to_complex = Lambda(lambda x: do_to_complex(x), name='to_complex')
