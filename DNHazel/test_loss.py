import numpy as np
from keras import objectives
from keras import backend as K
import tensorflow as tf
from ipdb import set_trace as stop
import scipy.stats as st
import scipy.misc as mi

_EPSILON = K.epsilon()

def _loss_tensor(y_true, y_pred):
    y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
    out = -(y_true * K.log(y_pred) + (1.0 - y_true) * K.log(1.0 - y_pred))
    return K.mean(out, axis=-1)

def _loss_np(y_true, y_pred):
    y_pred = np.clip(y_pred, _EPSILON, 1.0-_EPSILON)
    out = -(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))
    return np.mean(out, axis=-1)

def _loss_tensor2(y_true, y_pred):
    dist = tf.contrib.distributions.Normal(mu=K.zeros_like(y_true), sigma=K.ones_like(y_true))
    n_dim = len(y_pred.get_shape().as_list())
    return tf.reduce_logsumexp(dist.log_pdf(y_pred), reduction_indices=[n_dim-1])

def _loss_np2(y_true, y_pred):
    log_pdf = st.norm.logpdf(y_pred, np.zeros_like(y_true), np.ones_like(y_true))    
    return mi.logsumexp(log_pdf, axis=-1)

def check_loss(_shape):
    if _shape == '2d':
        shape = (6, 7)
    elif _shape == '3d':
        shape = (5, 6, 7)
    elif _shape == '4d':
        shape = (8, 5, 6, 7)
    elif _shape == '5d':
        shape = (9, 8, 5, 6, 7)

    y_a = np.random.random(shape)
    y_b = np.random.random(shape)

    out1 = K.eval(_loss_tensor2(K.variable(y_a), K.variable(y_b)))
    out2 = _loss_np2(y_a, y_b)

    assert out1.shape == out2.shape
    assert out1.shape == shape[:-1]
    print(np.linalg.norm(out1))
    print(np.linalg.norm(out2))
    print(np.linalg.norm(out1-out2))


def test_loss():
    shape_list = ['2d', '3d', '4d', '5d']
    for _shape in shape_list:
        check_loss(_shape)
        print('======================')

if __name__ == '__main__':
    test_loss()