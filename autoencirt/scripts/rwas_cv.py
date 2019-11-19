#!/usr/bin/env python3
import math
import os
from os import path, system
from copy import copy, deepcopy

import numpy as np
import pandas as pd
import pandas as pd
from sklearn.model_selection import KFold

import autoencirt
from autoencirt.irt import GRModel

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python import util as tfp_util

from tensorflow_probability.python import util as tfp_util
from tensorflow_probability.python.mcmc.transformed_kernel import (
    make_transform_fn, make_transformed_log_prob, make_log_det_jacobian_fn)
from tensorflow_probability.python.experimental.vi.surrogate_posteriors import (
    build_factored_surrogate_posterior)

from tensorflow.python import tf2
if not tf2.enabled():
    import tensorflow.compat.v2 as tf
    tf.enable_v2_behavior()
    assert tf2.enabled()


tfd = tfp.distributions

tfd = tfp.distributions
tfb = tfp.bijectors


def get_data():
    if not path.exists('RWAS/data.csv'):
        system("wget https://openpsychometrics.org/_rawdata/RWAS.zip")
        system("unzip RWAS.zip")
    data = pd.read_csv('RWAS/data.csv', low_memory=False)
    item_responses = data.loc[:, map(lambda x: 'Q'+str(x), list(range(1, 23)))]
    # system("rm -r RWAS")
    return item_responses


def main():
    data = get_data()
    kf = KFold(n_splits=4)
    kf.get_n_splits(data)
    for train_index, test_index in kf.split(data):
        grm = GRModel()
        grm.load_data(data.iloc[train_index, :])
        grm.set_dimension(2)
        grm.create_distributions()
        losses = grm.calibrate_advi(10)
        print(losses)

        test_data_tensor = tf.cast(
            data.iloc[test_index, :].to_numpy(), tf.int32)
        scores = grm.score(test_data_tensor)
        prediction_loss = grm.loss(test_data_tensor, scores)


if __name__ == "__main__":
    main()
