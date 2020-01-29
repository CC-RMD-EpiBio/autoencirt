from tensorflow_probability.python.vi import csiszar_divergence
from tensorflow_probability.python import math as tfp_math
import numpy as np

from tensorflow_probability.python import util as tfp_util
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import dtype_util
import tensorflow as tf
import tensorflow_probability as tfp
import functools
from tensorflow_probability.python.experimental.vi.surrogate_posteriors import(
    build_trainable_location_scale_distribution
)
from tensorflow_probability.python.bijectors import softplus as softplus_lib
from tensorflow_probability.python.distributions.transformed_distribution import (
    TransformedDistribution
)

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer


tfd = tfp.distributions
tfb = tfp.bijectors


class SqrtCauchy(TransformedDistribution):
    def __init__(self, loc, scale, validate_args=False,
                 allow_nan_stats=True, name="SqrtCauchy"):
        parameters = dict(locals())
        with tf.name_scope(name) as name:
            super(SqrtCauchy, self).__init__(
                distribution=tfd.HalfCauchy(loc=loc, scale=scale),
                bijector=tfb.Invert(tfb.Square()),
                validate_args=validate_args,
                parameters=parameters,
                name=name)

    @classmethod
    def _params_event_ndims(cls):
        return dict(loc=0, scale=0)

    @property
    def loc(self):
        """Distribution parameter for the pre-transformed mean."""
        return self.distribution.loc

    @property
    def scale(self):
        """Distribution parameter for the
           pre-transformed standard deviation."""
        return self.distribution.scale


class SqrtInverseGamma(TransformedDistribution):
    def __init__(self, concentration, scale, validate_args=False,
                 allow_nan_stats=True, name="SqrtInverseGamma"):
        parameters = dict(locals())
        with tf.name_scope(name) as name:
            super(SqrtInverseGamma, self).__init__(
                distribution=tfd.InverseGamma(
                    concentration=concentration, scale=scale),
                bijector=tfb.Invert(tfb.Square()),
                validate_args=validate_args,
                parameters=parameters,
                name=name)

    @classmethod
    def _params_event_ndims(cls):
        return dict(loc=0, scale=0)

    @property
    def loc(self):
        """Distribution parameter for the pre-transformed mean."""
        return self.distribution.loc

    @property
    def scale(self):
        """Distribution parameter for the
           pre-transformed standard deviation."""
        return self.distribution.scale


def clip_gradients(fn, clip_value):
    def wrapper(*args, **kwargs):
        @tf.custom_gradient
        def grad_wrapper(*flat_args_kwargs):
            with tf.GradientTape() as tape:
                tape.watch(flat_args_kwargs)
                new_args, new_kwargs = tf.nest.pack_sequence_as(
                    (args, kwargs),
                    flat_args_kwargs)
                ret = fn(*new_args, **new_kwargs)

            def grad_fn(*dy):
                flat_grads = tape.gradient(
                    ret, flat_args_kwargs, output_gradients=dy)
                flat_grads = tf.nest.map_structure(
                    lambda g: tf.where(tf.math.is_finite(g),
                                       g, tf.zeros_like(g)),
                    flat_grads)
                return tf.clip_by_global_norm(flat_grads, clip_value)[0]
            return ret, grad_fn
        return grad_wrapper(*tf.nest.flatten((args, kwargs)))
    return wrapper


@tf.function
def run_chain(
        init_state, step_size, target_log_prob_fn,
        unconstraining_bijectors, num_steps=500,
        burnin=50, num_leapfrog_steps=5, nuts=True
):
    if nuts:
        def trace_fn(_, pkr):
            return (
                pkr.inner_results.inner_results.target_log_prob,
                pkr.inner_results.inner_results.leapfrogs_taken,
                pkr.inner_results.inner_results.has_divergence,
                pkr.inner_results.inner_results.energy,
                pkr.inner_results.inner_results.log_accept_ratio
            )

        kernel = tfp.mcmc.TransformedTransitionKernel(
            inner_kernel=tfp.mcmc.NoUTurnSampler(
                target_log_prob_fn,
                step_size=step_size),
            bijector=unconstraining_bijectors)

        hmc = tfp.mcmc.DualAveragingStepSizeAdaptation(
            inner_kernel=kernel,
            num_adaptation_steps=burnin,
            step_size_setter_fn=lambda pkr, new_step_size: pkr._replace(
                inner_results=pkr.inner_results._replace(
                    step_size=new_step_size)),
            step_size_getter_fn=lambda pkr: pkr.inner_results.step_size,
            log_accept_prob_getter_fn=lambda pkr: (
                pkr.inner_results.log_accept_ratio)
        )

        # Sampling from the chain.
        chain_state, sampler_stat = tfp.mcmc.sample_chain(
            num_results=num_steps,
            num_burnin_steps=burnin,
            current_state=init_state,
            kernel=hmc,
            trace_fn=trace_fn)
    else:
        def trace_fn_hmc(_, pkr):
            return (pkr.inner_results.inner_results.is_accepted,
                    pkr.inner_results.inner_results.accepted_results.step_size)
        hmc = tfp.mcmc.TransformedTransitionKernel(
            inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=target_log_prob_fn,
                num_leapfrog_steps=num_leapfrog_steps,
                step_size=step_size,
                state_gradients_are_stopped=True),
            bijector=unconstraining_bijectors)
        kernel = tfp.mcmc.SimpleStepSizeAdaptation(
            inner_kernel=hmc, num_adaptation_steps=int(0.8*burnin))
        chain_state, sampler_stat = tfp.mcmc.sample_chain(
            num_results=num_steps,
            num_burnin_steps=burnin,
            current_state=init_state,
            kernel=kernel,
            trace_fn=trace_fn_hmc)

    return chain_state, sampler_stat


class LossLearningRateScheduler(tf.keras.callbacks.History):
    """
    A learning rate scheduler that relies on changes in loss function
    value to dictate whether learning rate is decayed or not.
    LossLearningRateScheduler has the following properties:
    base_lr: the starting learning rate
    lookback_epochs: the number of epochs in the past to compare
        with the loss function at the current epoch to determine if
        progress is being made.
    decay_threshold / decay_multiple: if loss function has not improved
        by a factor of decay_threshold * lookback_epochs, then decay_multiple
        will be applied to the learning rate.
    spike_epochs: list of the epoch numbers where you want to spike
        the learning rate.
    spike_multiple: the multiple applied to the current learning
        rate for a spike.
    """

    def __init__(
            self, base_lr, lookback_epochs, spike_epochs=None,
            spike_multiple=10, decay_threshold=0.002, decay_multiple=0.5,
            loss_type='val_loss'):

        super(LossLearningRateScheduler, self).__init__()

        self.base_lr = base_lr
        self.lookback_epochs = lookback_epochs
        self.spike_epochs = spike_epochs
        self.spike_multiple = spike_multiple
        self.decay_threshold = decay_threshold
        self.decay_multiple = decay_multiple
        self.loss_type = loss_type

    def on_epoch_begin(self, epoch, logs=None):

        if len(self.epoch) > self.lookback_epochs:
            current_lr = tf.keras.backend.get_value(self.model.optimizer.lr)
            target_loss = self.history[self.loss_type]
            loss_diff = target_loss[-int(self.lookback_epochs)
                                    ] - target_loss[-1]
            if loss_diff <= np.abs(
                target_loss[-1]
            ) * (
                self.decay_threshold
                    * self.lookback_epochs):

                print(' '.join(('Changing learning rate from', str(
                    current_lr), 'to', str(current_lr * self.decay_multiple))))
                tf.keras.backend.set_value(self.model.optimizer.lr,
                                           current_lr * self.decay_multiple)
                current_lr = current_lr * self.decay_multiple

            else:
                print(' '.join(('Learning rate:', str(current_lr))))

            if self.spike_epochs is not None and len(
                    self.epoch) in self.spike_epochs:
                print(' '.join(('Spiking learning rate from', str(
                    current_lr), 'to', str(current_lr * self.spike_multiple))))
                tf.keras.backend.set_value(self.model.optimizer.lr,
                                           current_lr * self.spike_multiple)

        else:

            print(' '.join(('Setting learning rate to', str(self.base_lr))))
            tf.keras.backend.set_value(self.model.optimizer.lr, self.base_lr)

        return tf.keras.backend.get_value(self.model.optimizer.lr)


def build_trainable_concentration_scale_distribution(
        initial_concentration,
        initial_scale,
        event_ndims,
        distribution_fn=tfd.InverseGamma,
        validate_args=False,
        name=None):
    """Builds a variational distribution from a location-scale family.
    Args:
      initial_concentration: Float `Tensor` initial concentration.
      initial_scale: Float `Tensor` initial scale.
      event_ndims: Integer `Tensor` number of event dimensions
        in `initial_concentration`.
      distribution_fn: Optional constructor for a `tfd.Distribution` instance
        in a location-scale family. This should have signature `dist =
        distribution_fn(loc, scale, validate_args)`.
        Default value: `tfd.Normal`.
      validate_args: Python `bool`. Whether to validate input with asserts.
        This imposes a runtime cost. If `validate_args` is `False`, and the
        inputs are invalid, correct behavior is not guaranteed.
        Default value: `False`.
      name: Python `str` name prefixed to ops created by this function.
        Default value: `None` (i.e.,
          'build_trainable_location_scale_distribution').
    Returns:
      posterior_dist: A `tfd.Distribution` instance.
    """
    with tf.name_scope(
            name or 'build_trainable_concentration_scale_distribution'):
        dtype = dtype_util.common_dtype([initial_concentration, initial_scale],
                                        dtype_hint=tf.float32)
        initial_concentration = tf.convert_to_tensor(
            initial_concentration, dtype=dtype)
        initial_scale = tf.convert_to_tensor(initial_scale, dtype=dtype)

        loc = tfp_util.TransformedVariable(
            initial_concentration,
            softplus_lib.Softplus(),
            name='concentration')
        scale = tfp_util.TransformedVariable(
            initial_scale, softplus_lib.Softplus(), name='scale')
        posterior_dist = distribution_fn(concentration=loc, scale=scale,
                                         validate_args=validate_args)

        # Ensure the distribution has the desired number of event dimensions.
        static_event_ndims = tf.get_static_value(event_ndims)
        if static_event_ndims is None or static_event_ndims > 0:
            posterior_dist = tfd.Independent(
                posterior_dist,
                reinterpreted_batch_ndims=event_ndims,
                validate_args=validate_args)

    return posterior_dist


build_trainable_InverseGamma_dist = functools.partial(
    build_trainable_concentration_scale_distribution,
    distribution_fn=tfd.InverseGamma
)

build_trainable_normal_dist = functools.partial(
    build_trainable_location_scale_distribution,
    distribution_fn=tfd.Normal)


def _trace_loss(loss, grads, variables): return loss


def _trace_variables(loss, grads, variables): return loss, variables


def auto_minimize(loss_fn,
                  num_steps=1000,
                  max_plateaus=10,
                  abs_tol=1e-4,
                  rel_tol=1e-4,
                  trainable_variables=None,
                  trace_fn=_trace_loss,
                  learning_rate=1.,
                  check_every=25,
                  decay_rate=0.95,
                  name='minimize'):

    def learning_rate_schedule_fn(step):
        return decay_rate**step

    decay_step = 0

    optimizer = tf.optimizers.Adam(
        learning_rate=lambda: learning_rate_schedule_fn(decay_step)
    )
    opt = tfa.optimizers.Lookahead(optimizer)

    learning_rate = 1.0 if learning_rate is None else learning_rate

    abs_tol = 1e-8 if abs_tol is None else abs_tol
    rel_tol = 1e-8 if rel_tol is None else rel_tol

    @tf.function(autograph=False)
    def train_loop_body(old_result, step):  # pylint: disable=unused-argument
        """Run a single optimization step."""
        with tf.GradientTape(
                watch_accessed_variables=trainable_variables is None) as tape:
            for v in trainable_variables or []:
                tape.watch(v)
            loss = loss_fn()
        watched_variables = tape.watched_variables()
        grads = tape.gradient(loss, watched_variables)
        train_op = opt.apply_gradients(zip(grads, watched_variables))
        with tf.control_dependencies([train_op]):
            state = trace_fn(tf.identity(loss),
                             [tf.identity(g) for g in grads],
                             [tf.identity(v) for v in watched_variables])
        return state

    with tf.name_scope(name) as name:
        # Compute the shape of the trace without executing
        # the graph, if possible.
        concrete_loop_body = train_loop_body.get_concrete_function(
            tf.TensorSpec([]), tf.TensorSpec([]))  # Inputs ignored.
        if all([tensorshape_util.is_fully_defined(shape)
                for shape in tf.nest.flatten(concrete_loop_body.output_shapes)]):
            state_initializer = tf.nest.map_structure(
                lambda shape, dtype: tf.zeros(shape, dtype=dtype),
                concrete_loop_body.output_shapes,
                concrete_loop_body.output_dtypes)
            initial_trace_step = None
        else:
            state_initializer = concrete_loop_body(
                tf.convert_to_tensor(0.), tf.convert_to_tensor(0.))  # Inputs ignored.
            max_steps = max_steps - 1
            initial_trace_step = state_initializer

        converged = False
        results = []
        losses = []
        avg_losses = [1e10]*3
        deviations = [1e10]*3
        min_loss = 1e10
        min_state = None
        # Test the first step, and make sure we can initialize safely

        losses += [
            train_loop_body(state_initializer, 0)
        ]

        step = tf.cast(1, tf.int32)
        while (step < num_steps) and not converged:
            losses += [
                train_loop_body(state_initializer, step)
            ]
            if step % check_every == 0:

                if losses[-1] < min_loss:
                    min_loss = losses[-1]

                """Check for convergence
                """
                recent_losses = tf.convert_to_tensor(losses[-check_every:])
                avg_loss = tf.reduce_mean(recent_losses).numpy()

                if not np.isfinite(avg_loss):
                    print("Backtracking due to inf or nan")

                avg_losses += [avg_loss]
                deviation = tf.math.reduce_std(recent_losses).numpy()
                deviations += [deviation]
                rel = deviation/avg_loss
                status = f"Iteration {step} -- loss: {losses[-1].numpy()}, "
                status += f"abs_err: {deviation}, rel_err: {rel}"

                #
                print(status)

                """Check for plateau
                """
                if (
                        avg_losses[-1] > avg_losses[-3]
                ) and (
                        avg_losses[-1] > avg_losses[-2]
                ):
                    decay_step += 1
                    if decay_step >= max_plateaus:
                        converged = True
                        print(
                            f"We have plateaued {decay_step} times so quitting"
                        )
                    else:
                        status = "We are in a loss plateau"
                        status += f" learning rate: {optimizer.lr}"
                        print(status)

                    # converged = True

                if deviation < abs_tol:
                    print(
                        f"Converged in {step} iterations with acceptable absolute tolerance")
                    converged = True
                elif rel < rel_tol:
                    print(
                        f"Converged in {step} iterations with acceptable relative tolerance")
                    converged = True
            step += 1
            if step >= num_steps:
                print("Terminating because we are out of iterations")

        trace = tf.stack(losses)
        if initial_trace_step is not None:
            trace = tf.nest.map_structure(
                lambda a, b: tf.concat([a[tf.newaxis, ...], b], axis=0),
                initial_trace_step, trace)
        return trace


# Silent fallback to score-function gradients leads to difficult-to-debug
# failures, so we force reparameterization gradients by default.
_reparameterized_elbo = functools.partial(
    csiszar_divergence.monte_carlo_variational_loss,
    discrepancy_fn=csiszar_divergence.kl_reverse,
    use_reparameterization=True)


def fit_surrogate_posterior(target_log_prob_fn,
                            surrogate_posterior,
                            num_steps,
                            trace_fn=_trace_loss,
                            variational_loss_fn=_reparameterized_elbo,
                            sample_size=25,
                            learning_rate=1.0,
                            trainable_variables=None,
                            seed=None,
                            abs_tol=None,
                            rel_tol=None):

    def complete_variational_loss_fn():
        return variational_loss_fn(
            target_log_prob_fn,
            surrogate_posterior,
            sample_size=sample_size,
            seed=seed)

    return auto_minimize(complete_variational_loss_fn,
                         num_steps=num_steps,
                         trace_fn=trace_fn,
                         learning_rate=learning_rate,
                         trainable_variables=trainable_variables,
                         abs_tol=abs_tol,
                         rel_tol=rel_tol)


def build_surrogate_posterior(joint_distribution_named,
                              bijectors=None,
                              exclude=[],
                              num_samples=1000):
    prior_sample = joint_distribution_named.sample(int(num_samples))
    surrogate_dict = {}
    means = {k: tf.reduce_mean(v, axis=0) for k, v in prior_sample.items()}
    prior_sample = joint_distribution_named.sample()
    for k, v in joint_distribution_named.model.items():
        if k in exclude:
            continue
        if callable(v):
            test_input = {
                a: prior_sample[a] for a in inspect.getfullargspec(v).args}
            test_distribution = v(**test_input)
        else:
            test_distribution = v
        if isinstance(
            test_distribution.distribution, tfd.InverseGamma
        ) or isinstance(test_distribution.distribution, SqrtInverseGamma):
            surrogate_dict[k] = bijectors[k](
                build_trainable_InverseGamma_dist(
                    2.*tf.ones(test_distribution.event_shape),
                    tf.ones(test_distribution.event_shape),
                    len(test_distribution.event_shape)
                )
            )
        else:
            surrogate_dict[k] = bijectors[k](
                build_trainable_normal_dist(
                    tfb.invert(bijectors[k])(means[k]),
                    1e-2*tf.ones(test_distribution.event_shape),
                    len(test_distribution.event_shape)
                )
            )
    return tfd.JointDistributionNamed(surrogate_dict)
