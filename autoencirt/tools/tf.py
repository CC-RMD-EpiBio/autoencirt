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


tfd = tfp.distributions
tfb = tfp.bijectors


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

            if self.spike_epochs is not None and len(self.epoch) in self.spike_epochs:
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
