import tensorflow as tf

from autoencirt.irt.grm import GRModel
from autoencirt.nn import DenseHorseshoe


class AEGRModel(GRModel):

    def __init__(self,
                 auxiliary_parameterization=True,
                 xi_scale=1e-2,
                 eta_scale=1e-2,
                 kappa_scale=1e-2,
                 weight_exponent=1.0,
                 dim=2,
                 decay=0.25,
                 positive_discriminations=True,
                 hidden_layers=[100, 100],
                 num_items=1,
                 ):
        super(AEGRModel, self).__init__(
            auxiliary_parameterization=True,
            xi_scale=xi_scale,
            eta_scale=eta_scale,
            kappa_scale=kappa_scale,
            weight_exponent=weight_exponent,
            dim=dim,
            decay=decay,
            positive_discriminations=positive_discriminations
        )
        self.num_items = num_items,
        self.hidden_layers = hidden_layers
        self.grm_vars = self.var_list

    def initialize_nn(self, hidden_layers=None):
        if hidden_layers is not None:
            self.hidden_layers = hidden_layers
        else:
            hidden_layers = self.hidden_layers

        self.nn = DenseHorseshoe(
            self.num_items,
            hidden_layers + [self.dimensions],
            reparameterized=True)

        self.nn_var_list = self.nn.var_list

    def load_data(self, *args, **kwargs):
        super(AEGRModel, self).load_data(*args, **kwargs)
        self.initialize_nn()

    def joint_log_prob(self, **x):
        prior = self.joint_log_prior(**x)
        d0 = tf.concat(
            [x['difficulties0'], x['ddifficulties']],
            axis=-1)
        difficulties = tf.cumsum(
            d0, axis=-1)
        likelihood = tf.reduce_sum(
            self.log_likelihood(
                self.calibration_data,
                x['discriminations'],
                difficulties,
                x['abilities']
            ),
            axis=[-1, -2]
        )
        return prior + likelihood

    def joint_log_prior(
            self, **x):
        weight_tensors = {v: x[v] for v in self.nn.weight_var_list}
        abilities = self.nn.assemble_networks(weight_tensors)(self.calibration_data)
        grm_vars = {k: x[k] for k in self.grm_vars}
        grm_vars["abilities"] = abilities[..., tf.newaxis, tf.newaxis]
        grm_vars["responses"] = self.calibration_data
        nn_log_prior = self.nn.log_prob(weight_tensors)
        grm_log_prior = (
            super(
                AEGRModel, self
            ).joint_log_prob_auxiliary(**grm_vars) if self.auxiliary_parameterization
            else
            super(
                AEGRModel, self
            ).joint_log_prob(**grm_vars)
        )
        return nn_log_prior + grm_log_prior

    def sample(self, *args, **kwargs):
        nn_sample = self.nn.sample(*args, **kwargs)
        grm_sample = self.surrogate_posterior.sample(*args, **kwargs)
        return {**nn_sample, **grm_sample}


def main():
    from autoencirt.data.rwa import get_data
    aegrm = AEGRModel(hidden_layers=[20, 30])
    aegrm.load_data(get_data())
    aegrm.create_distributions()
    sample = aegrm.sample([2, 3])
    prob = aegrm.joint_log_prob(**sample)
    return


if __name__ == "__main__":
    main()
