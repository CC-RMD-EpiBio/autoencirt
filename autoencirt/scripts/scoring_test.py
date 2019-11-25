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
    grm = GRModel(auxiliary_parameterization=True)
    grm.set_dimension(2)
    grm.num_items = 25
    grm.num_people = 1000
    grm.response_cardinality = 5
    grm.create_distributions()
    synthetic_data, discriminations, abilities = grm.simulate_data(5)
    scores = grm.score(synthetic_data)
    # print(scores[0] - abilities)


if __name__ == "__main__":
    main()