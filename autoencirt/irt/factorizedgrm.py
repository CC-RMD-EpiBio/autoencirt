#!/usr/bin/env python3

import jax.numpy as jnp
from bayesianquilts.distributions import AbsHorseshoe, SqrtInverseGamma
from bayesianquilts.util import training_loop
from bayesianquilts.vi.advi import build_factored_surrogate_posterior_generator
from jax.scipy.special import xlogy
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.substrates.jax import bijectors as tfb
from tensorflow_probability.substrates.jax import distributions as tfd
from tensorflow_probability.substrates.jax import tf2jax as tf

from autoencirt.irt import IRTModel


class FactorizedGRModel(IRTModel):
    """Implement and store the graded response model for IRT

    Arguments:
        IRTModel {[type]} -- [description]

    Returns:
        [type] -- [description]
    """

    response_type = "polytomous"

    def __init__(self, scale_indices, kappa_scale, *args, **kwargs):
        """Initialize model based on scale indices

        Args:
            scale_indices (list(list(int))): Indices for the items per scale
        """
        self.scale_indices = scale_indices
        super(FactorizedGRModel, self).__init__(*args, **kwargs)
        self.kappa_scale = kappa_scale
        self.dimensions = len(scale_indices)
        self.create_distributions()

    def create_distributions(self):
        """Joint probability with measure over observations
        groupings: Vector of same size as the data

        Returns:
            tf.distributions.JointDistributionNamed -- Joint distribution
        """
        self.bijectors = {
            k: tfb.Identity() for k in ["abilities", "mu", "difficulties0"]
        }

        self.bijectors["kappa"] = tfb.Softplus()
        self.bijectors["kappa_a"] = tfb.Softplus()

        self.bijectors["discriminations"] = tfb.Softplus()
        self.bijectors["ddifficulties"] = tfb.Softplus()  # make_shifted_softplus(1e-3)

        K = self.response_cardinality

        grm_joint_distribution_dict = {}

        for j, indices in enumerate(self.scale_indices):
            self.bijectors[f"discriminations_{j}"] = tfb.Softplus()
            self.bijectors[f"ddifficulties_{j}"] = tfb.Softplus()
            grm_joint_distribution_dict = {
                **grm_joint_distribution_dict,
                **self.gen_discrim_prior(j, indices),
                **self.gen_difficulty_prior(j, indices),
            }

        grm_joint_distribution_dict["abilities"] = tfd.Independent(
            tfd.Normal(
                loc=jnp.zeros((self.num_people, self.dimensions, 1, 1), self.dtype),
                scale=jnp.ones((self.num_people, self.dimensions, 1, 1), self.dtype),
            ),
            reinterpreted_batch_ndims=4,
        )

        self.joint_prior_distribution = tfd.JointDistributionNamed(
            grm_joint_distribution_dict
        )
        self.surrogate_distribution_generator, self.surrogate_parameter_initializer = (
            build_factored_surrogate_posterior_generator(
                self.joint_prior_distribution,
                bijectors=self.bijectors,
                dtype=self.dtype,
            )
        )
        self.params = self.surrogate_parameter_initializer()
        return

    def grm_model_prob(self, abilities, discriminations, difficulties):
        logits = difficulties - abilities  # N x D x I x K-1
        logits = logits * discriminations
        probs = 1.0 / (1 + jnp.exp(logits))
        probs = jnp.pad(
            probs,
            ([(0, 0)] * (len(probs.shape) - 1) + [(1, 0)]),
            mode="constant",
            constant_values=1,
        )
        probs = jnp.pad(
            probs,
            ([(0, 0)] * (len(probs.shape) - 1) + [(0, 1)]),
            mode="constant",
            constant_values=0,
        )
        probs = probs[..., :-1] - probs[..., 1:]
        return probs

    def grm_model_prob_d(
        self, abilities, discriminations, difficulties0, ddifficulties
    ):
        d0 = jnp.concat([difficulties0, ddifficulties], axis=-1)
        difficulties = jnp.cumsum(d0, axis=-1)
        return self.grm_model_prob(abilities, discriminations, difficulties)

    def fit_dim(self, *args, dim: int, **kwargs):
        if dim >= self.dimensions:
            raise ValueError("Dimension to fit must be less than model dimensions")
        # Only optimize parameters for the selected dimension `dim`
        optimizing_keys = [
            k
            for k in self.params.keys()
            if (
                not any(
                    k.startswith(prefix) and not k.startswith(f"{prefix}{dim}")
                    for prefix in [
                        "discriminations_",
                        "ddifficulties_",
                        "difficulties0_",
                        "kappa_",
                        "kappa_a_",
                    ]
                )
            )
        ]
        return self.fit(*args, **kwargs, optimize_keys=optimizing_keys)

    # @jax.jit
    def transform(self, params):
        # re-assemble the discriminations
        discriminations = []
        d0 = []
        dd = []
        for j, indices in enumerate(self.scale_indices):
            update = jnp.transpose(params[f"discriminations_{j}"], [3, 0, 1, 2, 4])[
                :, :, 0, 0, 0
            ]
            update_d0 = jnp.transpose(params[f"difficulties0_{j}"], [3, 0, 1, 2, 4])[
                :, :, 0, 0, 0
            ]
            update_dd = jnp.transpose(params[f"ddifficulties_{j}"], [4, 3, 0, 1, 2])[
                :, :, :, 0, 0
            ]
            S = update.shape[1]
            output_array = jnp.zeros((S, self.num_items), dtype=update.dtype)
            output_array_d0 = jnp.zeros((S, self.num_items), dtype=update.dtype)
            output_array_dd = jnp.zeros(
                (S, self.num_items, self.response_cardinality - 2), dtype=update.dtype
            )

            discriminations += [
                output_array.at[:, indices].set(update.T)[..., jnp.newaxis]
            ]
            d0 += [output_array_d0.at[:, indices].set(update_d0.T)[..., jnp.newaxis]]
            dd += [output_array_dd.at[:, indices].set(update_dd.T)[..., jnp.newaxis]]
        discriminations = jnp.concat(discriminations, axis=-1)
        d0 = jnp.concat(d0, axis=-1)
        dd = jnp.concat(dd, axis=-1)
        _shape = discriminations.shape
        _rank = len(_shape)
        dd = jnp.transpose(
            dd, [t for t in range(_rank - 2)] + [_rank, _rank - 2, _rank - 1]
        )
        discriminations = jnp.transpose(
            discriminations, [t for t in range(_rank - 2)] + [_rank - 1, _rank - 2]
        )
        discriminations = discriminations[..., jnp.newaxis, :, :, jnp.newaxis]

        d0 = jnp.transpose(d0, [t for t in range(_rank - 2)] + [_rank - 1, _rank - 2])
        d0 = d0[..., jnp.newaxis, :, :, jnp.newaxis]
        dd = dd[..., jnp.newaxis, :, :, :]
        params["discriminations"] = discriminations
        params["difficulties0"] = d0
        params["ddifficulties"] = dd
        diff = jnp.concat([d0, dd], axis=-1)
        diff = jnp.cumsum(diff, axis=-1)
        params["difficulties"] = diff
        # assemble the difficulties

        return params

    def gen_discrim_prior(self, j, indices):
        out = {}
        model_string = f"""lambda kappa_{j}: tfd.Independent(
            AbsHorseshoe(scale=kappa_{j}*tf.ones((1, 1, {len(indices)}, 1))), reinterpreted_batch_ndims=4)"""
        out[f"discriminations_{j}"] = eval(
            model_string,
            {"self": self, "tfd": tfd, "tf": tf, "AbsHorseshoe": AbsHorseshoe},
        )
        model_string = f"""lambda kappa_a_{j}: tfd.Independent(
            SqrtInverseGamma(
                0.5 * tf.ones((1, 1, 1, 1), dtype=self.dtype),
                1.0 / kappa_a_{j}),
            reinterpreted_batch_ndims=4)"""
        out[f"kappa_{j}"] = eval(
            model_string,
            {"self": self, "tfd": tfd, "tf": tf, "SqrtInverseGamma": SqrtInverseGamma},
        )
        out[f"kappa_a_{j}"] = tfd.Independent(
            tfd.InverseGamma(
                0.5 * tf.ones((1, 1, 1, 1), dtype=self.dtype),
                tf.ones((1, 1, 1, 1), dtype=self.dtype)
                / self.kappa_scale**2,
            ),
            reinterpreted_batch_ndims=4,
        )
        return out

    def gen_difficulty_prior(self, j, indices):
        out = {}
        out[f"difficulties0_{j}"] = tfd.Independent(
            tfd.Normal(
                loc=3
                * tf.ones(
                    (1, 1, len(indices), 1), dtype=self.dtype
                ),  # mu[..., indices, :],
                scale=tf.ones((1, 1, len(indices), 1), dtype=self.dtype),
            ),
            reinterpreted_batch_ndims=4,
        )
        out[f"ddifficulties_{j}"] = tfd.Independent(
            tfd.HalfNormal(
                scale=tf.ones(
                    (
                        1,
                        1,
                        len(indices),
                        self.response_cardinality - 2,
                    ),
                    dtype=self.dtype,
                ),
            ),
            reinterpreted_batch_ndims=4,
        )
        return out

    def predictive_distribution(
        self, data, discriminations, difficulties0, ddifficulties, abilities, **kwargs
    ):
        ddifficulties = jnp.where(
            ddifficulties < 1e-1, 1e-1 * jnp.ones_like(ddifficulties), ddifficulties
        )
        # check if responses are a batch
        difficulties = jnp.concat([difficulties0, ddifficulties], axis=-1)
        difficulties = jnp.cumsum(difficulties, axis=-1)

        # gather abilities and item parameters corresponding to responses

        rank = len(abilities.shape)
        batch_shape = abilities.shape[: (rank - 4)]
        batch_ndims = len(batch_shape)

        people = data[self.person_key].astype(jnp.int32)
        choices = jnp.concat([data[i][:, jnp.newaxis] for i in self.item_keys], axis=-1)

        bad_choices = (choices < 0) | (choices >= self.response_cardinality) | jnp.isnan(choices)

        for _ in range(batch_ndims):
            choices = choices[jnp.newaxis, ...]

        abilities = abilities[..., people, :, :, :]

        response_probs = self.grm_model_prob(abilities, discriminations, difficulties)
        discrimination_weights = jnp.abs(discriminations) / jnp.sum(
            jnp.abs(discriminations), axis=-3, keepdims=True)

        response_probs = jnp.sum(response_probs*discrimination_weights, axis=-3)
        imputed_lp = jnp.sum(xlogy(response_probs, response_probs), axis=-1)

        rv_responses = tfd.Categorical(probs=response_probs)

        log_probs = rv_responses.log_prob(choices)
        log_probs = jnp.where(bad_choices[jnp.newaxis, ...], imputed_lp, log_probs)

        log_probs = jnp.sum(log_probs, axis=-1)

        return {
            "log_likelihood": log_probs,
            "discriminations": discriminations,
            "rv": rv_responses,
        }

    def fit_projection(
        self,
        other,
        batched_data_factory,
        steps_per_epoch,
        num_epochs,
        samples=32,
        **kwargs,
    ):
        def objective(data):
            this_prediction = self.predictive_distribution(
                data, **self.sample(samples)
            )["rv"]
            other_prediction = other.predictive_distribution(
                data, **other.sample(samples)
            )["rv"]
            delta = other_prediction.kl_divergence(this_prediction)
            return tf.reduce_mean(delta)

        return training_loop(
            self.params,
            objective,
            data_iterator=batched_data_factory,
            steps_per_epoch=steps_per_epoch,
            num_epochs=num_epochs,
            trainable_variables=self.surrogate_distribution.variables,
            **kwargs,
        )

    def unormalized_log_prob(self, data, prior_weight=1.0, **params):
        _params = params.copy()
        if 'difficulties' not in _params:
            _params = self.transform(_params)
        log_prior = self.joint_prior_distribution.log_prob(params)
        prediction = self.predictive_distribution(data, **_params)
        log_likelihood = prediction["log_likelihood"]

        finite_portion = jnp.where(
            jnp.isfinite(log_likelihood),
            log_likelihood,
            jnp.zeros_like(log_likelihood),
        )

        min_val = jnp.min(finite_portion) - 5.0
        log_likelihood = jnp.where(
            jnp.isfinite(log_likelihood),
            log_likelihood,
            jnp.ones_like(log_likelihood) * min_val,
        )
        return prior_weight * log_prior + jnp.sum(log_likelihood, axis=-1)
