#!/usr/bin/env python3
import itertools

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

from factor_analyzer import FactorAnalyzer

from autoencirt.irt import IRTModel, GRModel
from bayesianquilts.vi.advi import (
    build_trainable_InverseGamma_dist,
    build_trainable_normal_dist,
    build_surrogate_posterior,
    build_trainable_concentration_distribution,
)

from bayesianquilts.distributions import SqrtInverseGamma, AbsHorseshoe

from tensorflow_probability.python import util as tfp_util
from tensorflow_probability.python.bijectors import softplus as softplus_lib

from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python.distributions import LogNormal


class FactorizedGRModel(GRModel):
    """Implement and store the graded response model for IRT

    Arguments:
        IRTModel {[type]} -- [description]

    Returns:
        [type] -- [description]
    """

    response_type = "polytomous"

    def __init__(self, scale_indices, *args, **kwargs):
        """Initialize model based on scale indices

        Args:
            scale_indices (list(list(int))): Indices for the items per scale
        """
        self.scale_indices = scale_indices
        super(FactorizedGRModel, self).__init__(*args, **kwargs)
        self.dim = len(scale_indices)
        self.create_distributions()

    def create_distributions(self, grouping_params=None):
        """Joint probability with measure over observations
        groupings: Vector of same size as the data

        Returns:
            tf.distributions.JointDistributionNamed -- Joint distribution
        """
        self.bijectors = {
            k: tfp.bijectors.Identity() for k in ["abilities", "mu", "difficulties0"]
        }

        self.bijectors["eta"] = tfp.bijectors.Softplus()
        self.bijectors["kappa"] = tfp.bijectors.Softplus()

        self.bijectors["discriminations"] = tfp.bijectors.Softplus()
        self.bijectors["ddifficulties"] = tfp.bijectors.Softplus()

        self.bijectors["eta_a"] = tfp.bijectors.Softplus()
        self.bijectors["kappa_a"] = tfp.bijectors.Softplus()

        K = self.response_cardinality
        difficulties0 = np.sort(
            np.random.normal(size=(1, self.dimensions, self.num_items, K - 1)), axis=-1
        )

        grm_joint_distribution_dict = dict(
            mu=tfd.Independent(
                tfd.Normal(
                    loc=tf.zeros(
                        (1, self.dimensions, self.num_items, 1), dtype=self.dtype
                    ),
                    scale=tf.ones(
                        (1, self.dimensions, self.num_items, 1), dtype=self.dtype
                    ),
                ),
                reinterpreted_batch_ndims=4,
            ),  # mu
            difficulties0=lambda mu: tfd.Independent(
                tfd.Normal(
                    loc=mu,
                    scale=tf.ones(
                        (1, self.dimensions, self.num_items, 1), dtype=self.dtype
                    ),
                ),
                reinterpreted_batch_ndims=4,
            ),  # difficulties0
            ddifficulties=tfd.Independent(
                tfd.HalfNormal(
                    scale=tf.ones(
                        (
                            1,
                            self.dimensions,
                            self.num_items,
                            self.response_cardinality - 2,
                        ),
                        dtype=self.dtype,
                    )
                ),
                reinterpreted_batch_ndims=4,
            ),
        )

        for j, indices in enumerate(self.scale_indices):
            grm_joint_distribution_dict = {
                **grm_joint_distribution_dict,
                **self.gen_discrim_prior(j),
            }
        if grouping_params is not None:
            grm_joint_distribution_dict["probs"] = tfd.Independent(
                tfd.Dirichlet(tf.cast(grouping_params, self.dtype)),
                reinterpreted_batch_ndims=1,
            )
            grm_joint_distribution_dict["mu_ability"] = lambda sigma: tfd.Independent(
                tfd.Normal(
                    loc=tf.zeros((self.dimensions, self.num_groups), self.dtype),
                    scale=sigma,
                ),
                reinterpreted_batch_ndims=2,
            )
            self.bijectors["sigma"] = tfp.bijectors.Softplus()
            grm_joint_distribution_dict["sigma"] = tfd.Independent(
                tfd.HalfNormal(
                    scale=0.5 * tf.ones((self.dimensions, self.num_groups), self.dtype)
                ),
                reinterpreted_batch_ndims=2,
            )

            grm_joint_distribution_dict["abilities"] = (
                lambda probs, mu_ability, sigma: tfd.Independent(
                    tfd.Mixture(
                        cat=tfd.Categorical(probs=probs),
                        components=[
                            tfd.Independent(
                                tfd.Normal(
                                    loc=(
                                        tf.squeeze(
                                            mu_ability[..., tf.newaxis, :, 0:1]
                                            + tf.zeros(
                                                shape=(
                                                    1,
                                                    self.num_people,
                                                    (
                                                        self.dimensions
                                                        if not self.include_independent
                                                        else self.dimensions - 1
                                                    ),
                                                    1,
                                                ),
                                                dtype=self.dtype,
                                            )
                                        )
                                    )[..., tf.newaxis, tf.newaxis],
                                    scale=(
                                        tf.squeeze(
                                            sigma[..., tf.newaxis, :, 0:1]
                                            + tf.zeros(
                                                shape=(
                                                    1,
                                                    self.num_people,
                                                    (
                                                        self.dimensions
                                                        if not self.include_independent
                                                        else self.dimensions - 1
                                                    ),
                                                    1,
                                                ),
                                                dtype=self.dtype,
                                            )
                                        )
                                    )[..., tf.newaxis, tf.newaxis],
                                ),
                                reinterpreted_batch_ndims=3,
                            ),
                            tfd.Independent(
                                tfd.Normal(
                                    loc=(
                                        tf.squeeze(
                                            mu_ability[..., tf.newaxis, :, 1:2]
                                            + tf.zeros(
                                                (
                                                    1,
                                                    self.num_people,
                                                    (
                                                        self.dimensions
                                                        if not self.include_independent
                                                        else self.dimensions - 1
                                                    ),
                                                    1,
                                                ),
                                                self.dtype,
                                            )
                                        )
                                    )[..., tf.newaxis, tf.newaxis],
                                    scale=(
                                        tf.squeeze(
                                            sigma[..., tf.newaxis, :, 1:2]
                                            + tf.zeros(
                                                (
                                                    1,
                                                    self.num_people,
                                                    (
                                                        self.dimensions
                                                        if not self.include_independent
                                                        else self.dimensions - 1
                                                    ),
                                                    1,
                                                ),
                                                self.dtype,
                                            )
                                        )
                                    )[..., tf.newaxis, tf.newaxis],
                                ),
                                reinterpreted_batch_ndims=3,
                            ),
                        ],
                    ),
                    reinterpreted_batch_ndims=1,
                )
            )
        else:
            grm_joint_distribution_dict["abilities"] = tfd.Independent(
                tfd.Normal(
                    loc=tf.zeros(
                        (
                            self.num_people,
                            (
                                self.dimensions
                                if not self.include_independent
                                else self.dimensions - 1
                            ),
                            1,
                            1,
                        ),
                        dtype=self.dtype,
                    ),
                    scale=tf.ones(
                        (
                            self.num_people,
                            (
                                self.dimensions
                                if not self.include_independent
                                else self.dimensions - 1
                            ),
                            1,
                            1,
                        ),
                        dtype=self.dtype,
                    ),
                ),
                reinterpreted_batch_ndims=4,
            )
        discriminations0 = (
            tfp.math.softplus_inverse(tf.cast(self.discrimination_guess, self.dtype))
            if self.discrimination_guess is not None
            else (
                -2.0
                * tf.ones((1, self.dimensions, self.num_items, 1), dtype=self.dtype)
            )
        )

        surrogate_distribution_dict = {
            "abilities": build_trainable_normal_dist(
                tf.zeros(
                    (
                        self.num_people,
                        (
                            self.dimensions
                            if not self.include_independent
                            else self.dimensions - 1
                        ),
                        1,
                        1,
                    ),
                    dtype=self.dtype,
                ),
                1e-3
                * tf.ones(
                    (
                        self.num_people,
                        (
                            self.dimensions
                            if not self.include_independent
                            else self.dimensions - 1
                        ),
                        1,
                        1,
                    ),
                    dtype=self.dtype,
                ),
                4,
                name="abilities",
            ),
            "ddifficulties": self.bijectors["ddifficulties"](
                build_trainable_normal_dist(
                    tf.cast(
                        difficulties0[..., 1:] - difficulties0[..., :-1],
                        dtype=self.dtype,
                    ),
                    1e-3
                    * tf.ones(
                        (
                            1,
                            self.dimensions,
                            self.num_items,
                            self.response_cardinality - 2,
                        ),
                        dtype=self.dtype,
                    ),
                    4,
                    name="ddifficulties",
                )
            ),
            "mu": build_trainable_normal_dist(
                tf.zeros((1, self.dimensions, self.num_items, 1), dtype=self.dtype),
                1e-3
                * tf.ones((1, self.dimensions, self.num_items, 1), dtype=self.dtype),
                4,
                name="mu",
            ),
        }

        for j, indices in enumerate(self.scale_indices):
            surrogate_distribution_dict[f"discriminations_{j}"] = self.bijectors[
                "discriminations"
            ](
                build_trainable_normal_dist(
                    # tf.cast(
                    #    (1.+np.abs(self.factor_loadings.T)),
                    #    self.dtype)[tf.newaxis, ..., tf.newaxis],
                    np.array(discriminations0)[..., j : (j + 1), indices, :1],
                    5e-2 * tf.ones((1, 1, len(indices), 1), dtype=self.dtype),
                    4,
                    name=f"discriminations_{j}",
                )
            )
            surrogate_distribution_dict[f"kappa_{j}"] = self.bijectors["kappa"](
                build_trainable_InverseGamma_dist(
                    0.5 * tf.ones((1, self.dimensions, 1, 1), dtype=self.dtype),
                    tf.ones((1, self.dimensions, 1, 1), dtype=self.dtype),
                    4,
                    name=f"kappa_{j}",
                )
            )
            surrogate_distribution_dict[f"kappa_a_{j}"] = self.bijectors["kappa_a"](
                build_trainable_InverseGamma_dist(
                    2 * tf.ones((1, self.dimensions, 1, 1), dtype=self.dtype),
                    tf.ones((1, self.dimensions, 1, 1), dtype=self.dtype),
                    4,
                    name=f"kappa_a_{j}",
                )
            )
        surrogate_distribution_dict = {
            **surrogate_distribution_dict,
            "difficulties0": build_trainable_normal_dist(
                tf.ones((1, self.dimensions, self.num_items, 1), dtype=self.dtype),
                1e-3
                * tf.ones((1, self.dimensions, self.num_items, 1), dtype=self.dtype),
                4,
                name="difficulties0",
            ),
        }

        if grouping_params is not None:
            surrogate_distribution_dict = {
                **surrogate_distribution_dict,
                "mu_ability": build_trainable_normal_dist(
                    tf.zeros(
                        (
                            (
                                self.dimensions
                                if not self.include_independent
                                else self.dimensions - 1
                            ),
                            self.num_groups,
                        ),
                        dtype=self.dtype,
                    ),
                    1e-2
                    * tf.ones(
                        (
                            (
                                self.dimensions
                                if not self.include_independent
                                else self.dimensions - 1
                            ),
                            self.num_groups,
                        ),
                        dtype=self.dtype,
                    ),
                    2,
                    name="mu_ability",
                ),
                "sigma": build_trainable_InverseGamma_dist(
                    tf.ones(
                        (
                            (
                                self.dimensions
                                if not self.include_independent
                                else self.dimensions - 1
                            ),
                            self.num_groups,
                        ),
                        dtype=self.dtype,
                    ),
                    tf.ones(
                        (
                            (
                                self.dimensions
                                if not self.include_independent
                                else self.dimensions - 1
                            ),
                            self.num_groups,
                        ),
                        dtype=self.dtype,
                    ),
                    2,
                    name="sigma",
                ),
                "probs": build_trainable_concentration_distribution(
                    tf.cast(grouping_params, self.dtype), 1, name="probs"
                ),
            }

        self.joint_prior_distribution = tfd.JointDistributionNamed(
            grm_joint_distribution_dict
        )
        self.surrogate_distribution = tfd.JointDistributionNamed(
            surrogate_distribution_dict
        )

        if self.vi_mode == "asvi":
            self.surrogate_distribution = (
                tfp.experimental.vi.build_asvi_surrogate_posterior(
                    prior=self.joint_prior_distribution
                )
            )

        self.surrogate_vars = self.surrogate_distribution.variables
        self.var_list = list(surrogate_distribution_dict.keys())
        self.set_calibration_expectations()

    def transform(self, params):
        # re-assemble the discriminations
        p_shape = params["discriminations_0"].shape.as_list()
        discriminations = tf.zeros(
            p_shape[:-3] + [self.dim] + [self.num_items, 1], dtype=self.dtype
        )
        for j, indices in enumerate(self.scale_indices):
            pass
        params["discriminations"] = discriminations
        return params

    def gen_discrim_prior(self, j):
        out = {}
        model_string = f"""lambda kappa_{j}: tfd.Independent(
            AbsHorseshoe(scale=kappa_{j}), reinterpreted_batch_ndims=4)"""
        out[f"discriminations_{j}"] = eval(
            model_string,
            {"self": self, "tfd": tfd, "tf": tf, "AbsHorseshoe": AbsHorseshoe},
        )
        model_string = f"""lambda kappa_a_{j}: tfd.Independent(
            SqrtInverseGamma(
                0.5 * tf.ones((1, self.dimensions, 1, 1), dtype=self.dtype),
                1.0 / kappa_a_{j}),
            reinterpreted_batch_ndims=4)"""
        out[f"kappa_{j}"] = eval(
            model_string,
            {"self": self, "tfd": tfd, "tf": tf, "SqrtInverseGamma": SqrtInverseGamma},
        )
        out[f"kappa_a_{j}"] = tfd.Independent(
            tfd.InverseGamma(
                0.5 * tf.ones((1, self.dimensions, 1, 1), dtype=self.dtype),
                tf.ones((1, self.dimensions, 1, 1), dtype=self.dtype)
                / self.kappa_scale**2,
            ),
            reinterpreted_batch_ndims=4,
        )
        return out
