import tensorflow as tf
import tensorflow_probability as tfp


def clip_gradients(fn, clip_value):
    def wrapper(*args):
        @tf.custom_gradient
        def grad_wrapper(*flat_args):
            with tf.GradientTape() as tape:
                tape.watch(flat_args)
                ret = fn(*tf.nest.pack_sequence_as(args, flat_args))

            def grad_fn(*dy):
                flat_grads = tape.gradient(ret, flat_args, output_gradients=dy)
                flat_grads = tf.nest.map_structure(lambda g: tf.where(
                    tf.math.is_finite(g), g, tf.zeros_like(g)), flat_grads)
                return tf.clip_by_global_norm(flat_grads, clip_value)[0]
            return ret, grad_fn
        return grad_wrapper(*tf.nest.flatten(args))
    return wrapper


def clip_gradients_dict(fn, clip_value):
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
def graph_hmc(*args, **kwargs):
    """Compile static graph for tfp.mcmc.sample_chain.
    Since this is bulk of the computation, using @tf.function here
    signifcantly improves performance (empirically about ~5x).
    """
    return tfp.mcmc.sample_chain(*args, **kwargs)
