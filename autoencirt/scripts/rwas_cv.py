#!/usr/bin/env python3
import math

from copy import copy, deepcopy

import numpy as np
import pandas as pd
import pandas as pd
from sklearn.model_selection import KFold

from factor_analyzer import (
    ConfirmatoryFactorAnalyzer,
    FactorAnalyzer,
    ModelSpecificationParser)

import autoencirt
from autoencirt.irt import GRModel
from autoencirt.data.rwa import item_text, get_data

import tensorflow as tf
import tensorflow_probability as tfp
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


def main():
    data = get_data(reorient=True).iloc[range(1000), :]
    kf = KFold(n_splits=2)
    kf.get_n_splits(data)
    for train_index, test_index in kf.split(data):
        grm = GRModel(
            auxiliary_parameterization=True,
            xi_scale=1e-2,
            eta_scale=1e-1,
            kappa_scale=1e-1,
            weight_exponent=1
        )
        grm.set_dimension(2)
        grm.load_data(data.iloc[train_index, :])
        grm.create_distributions()

        discriminations = []
        losses = []
        # Use ADVI to get us close
        for _ in range(2):
            losses += [grm.calibrate_advi(300,
                                          initial_learning_rate=5e-3).numpy()]
            print(losses[-1])
            discriminations += [grm.calibrated_expectations['discriminations']
                                [0, ..., 0].numpy()]
            print(discriminations[-1])

        grm.waic()

        # MCMC sample from here
        traces, stats = grm.calibrate_mcmc(
            step_size=5e-2,
            num_steps=5000,
            burnin=3000,
            nuts=False)

        grm.score(data.iloc[test_index, :].to_numpy())
        print(grm.calibrated_expectations['discriminations'][0, ..., 0])

        test_data_tensor = tf.cast(
            data.iloc[test_index, :].to_numpy(), tf.int32)
        scores = grm.score(test_data_tensor)
        prediction_loss = grm.loss(test_data_tensor, scores)

        # Do things the traditional way for comparison

        fa = FactorAnalyzer(n_factors=grm.dimensions)
        fa.fit(data.iloc[train_index, :])


if __name__ == "__main__":
    main()
