#!/usr/bin/env python3
import math

from copy import copy, deepcopy

import numpy as np
import pandas as pd

from autoencirt.irt import GRModel
from autoencirt.data.rwa import item_text, get_data


import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfd = tfp.distributions
tfb = tfp.bijectors


def main():
    data, num_people = get_data(reorient=False)
    item_names = [f"Q{j}" for j in range(1, 23)]
    grm = GRModel(
        data=data.shuffle(buffer_size=200),
        item_keys=item_names,
        num_people=num_people,
        dim=2,
        xi_scale=1e-4,
        eta_scale=1e-4,
        weight_exponent=1.,
        response_cardinality=10,
    )
    # ds = next(iter(data.batch(121)))
    # p = grm.surrogate_distribution.sample(13)
    # grm.log_likelihood(**p, responses=ds)
    # grm.unormalized_log_prob(**p, data=ds)

    losses = grm.calibrate_advi(
        num_epochs=5, rel_tol=1e-4,
        learning_rate=.01, clip_value=4.,
        data_batches=10
    )

    print(
        grm.calibrated_expectations['discriminations'][0, ..., 0]
    )

    grm.calibrate_mcmc(
        num_steps=1000, burnin=500)

    print(
        grm.calibrated_expectations['discriminations'][0, ..., 0]
    )


if __name__ == "__main__":
    main()
