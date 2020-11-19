import tensorflow as tf
from tensorflow.keras.layers import Lambda


def fft_on_axis(x):
    x = tf.transpose(x, perm=[0, 3,  1, 2],)
    x_fft = tf.signal.rfft2d(x)
    result = tf.transpose(x_fft, perm=[0, 2, 3, 1])
    return result


def ifft_on_axis(x, target_size=None):
    x = tf.transpose(x, perm=[0, 3,  1, 2],)
    x_fft = tf.signal.irfft2d(x, target_size)
    result = tf.transpose(x_fft, perm=[0, 2, 3, 1])
    return result


fft_layer = Lambda(fft_on_axis, name='fft2d')
ifft_layer = Lambda(ifft_on_axis, name='irfft2d')
