#!/usr/bin/env python3
import math

from copy import copy, deepcopy

import numpy as np
import pandas as pd

import autoencirt
from autoencirt.irt import GRModel
from autoencirt.data.rwa import item_text, get_data


import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfd = tfp.distributions
tfb = tfp.bijectors


def main():
    tfdata, num_people = get_data(reorient=False)
    item_names = [f"Q{j}" for j in range(1, 23)]
    grm = GRModel(
        item_keys=item_names,
        num_people=num_people,
        dim=2,
        eta_scale=1e-1,
        kappa_scale=1e-1,
        weight_exponent=1.,
        response_cardinality=10,
    )
    batch_size = 121

    def data_factory_factory(batch_size=batch_size, repeat=False, shuffle=False):
        def data_factory(batch_size=batch_size):
            if shuffle:
                out = tfdata.shuffle(batch_size*10)
            else:
                out = tfdata
            
            if repeat:
                out = out.repeat()
            return out.batch(batch_size)
        return data_factory


    losses = grm.fit(
            data_factory_factory(shuffle=False, repeat=True),
            dataset_size=num_people,
            batches_per_step=1,
            check_every=int(num_people/batch_size),
            batch_size=batch_size,
            num_steps=3000,
            max_decay_steps=100,
            max_plateau_epochs=100,
            sample_size=32, 
            learning_rate=0.0005)


if __name__ == "__main__":
    main()
