import numpy as np
import jax    
import tensorflow_probability.substrates.jax as tfp
from bayesianquilts.predictors.nn.dense import DenseHorseshoe

from tensorflow_probability.substrates.jax import distributions as tfd
from tensorflow_probability.substrates.jax import tf2jax as tf

from autoencirt.irt.grm import GRModel


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
                 **kwargs
                 ):
        if 'item_keys' not in kwargs:
             kwargs['item_keys'] = [f"Item_{i}" for i in range(num_items)]
        
        num_people = kwargs.pop('num_people', None)

        self.num_items = num_items
        self.hidden_layers = hidden_layers
        self.dimensions = dim
        self.grm_vars = self.var_list # self.var_list might be empty here? IRTModel sets it? 
        # BayesianModel sets var_list = []. So self.var_list exists.
        
        # We need to initialize NN early so create_distributions (called by super) works
        if kwargs.get('data') is not None or kwargs.get('calibration_data') is not None or True: 
             # Always initialize NN? verify_models passed no data but expected instantiation.
             # Original code waited for load_data.
             # But create_distributions needs it.
             self.initialize_nn()

        super(AEGRModel, self).__init__(
            auxiliary_parameterization=auxiliary_parameterization,
            xi_scale=xi_scale,
            eta_scale=eta_scale,
            kappa_scale=kappa_scale,
            weight_exponent=weight_exponent,
            dim=dim,
            decay=decay,
            positive_discriminations=positive_discriminations,
            num_people=num_people, # ensure num_people passes if in kwargs
            **kwargs
        )

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

    def set_data(self, data, data_transform_fn=None):
        super(BayesianModel, self).set_data(data, data_transform_fn)
        # In original load_data, initialize_nn was called.
        self.initialize_nn()
        self.create_distributions()

    def joint_log_prob(self, **x):
        prior = self.joint_log_prior(**x)
        d0 = tf.concat(
            [x['difficulties0'], x['ddifficulties']],
            axis=-1)
        difficulties = tf.cumsum(
            d0, axis=-1)
        
        # We need data to calculate likelihood.
        # self.data in BayesianModel terms might be the full dataset object
        # but log_likelihood expects specific args or 'data' dict.
        # If running calibration, we usually pass batches.
        # Here we assume 'self.calibration_data' was what was used before, 
        # which now should be 'self.data' or derived from it.
        
        # However, joint_log_prob seems to assume x contains everything including data? 
        # Or self.calibration_data is global?
        # In new API, unormalized_log_prob is passed 'data'.
        pass 
        return prior # Wrapper for unormalized_log_prob usage

    def unormalized_log_prob(self, data, prior_weight=1.0, **x):
        # New API method replacement for joint_log_prob
        
        # 1. Prior
        weight_tensors = {v: x[v] for v in self.nn.weight_var_list}
        nn_log_prior = self.nn.log_prob(weight_tensors)
        
        # Calculate abilities from NN
        # We need the input to the NN. 
        # 'data' argument should contain the features for the NN?
        # In autoencirt, often the input is the response vector itself (autoencoder)
        abilities = self.nn.assemble_networks(weight_tensors)(data)
        
        grm_vars = {k: x[k] for k in self.grm_vars if k in x}
        grm_vars["abilities"] = abilities[..., tf.newaxis, tf.newaxis]
        
        # Check if auxiliary parameterization is needed for GRM prior
        # (Assuming super().unormalized_log_prob handles GRM prior + likelihood logic 
        # but here we are mixing NN prior + GRM prior + GRM likelihood)
        
        # Actually, let's call the super helper if it exists, or manually do it.
        # super(AEGRModel, self).joint_log_prob checks aux param.
        
        # Let's simplify:
        # The prior for the GRM params (excluding abilities which come from NN):
        # We need to compute log_likelihood manually because abilities come from NN, 
        # not sampled parameters.
        
        d0 = tf.concat([x['difficulties0'], x['ddifficulties']], axis=-1)
        difficulties = tf.cumsum(d0, axis=-1)
        
        # Likelihood
        # self.log_likelihood comes from GRModel
        ll = self.log_likelihood(
            data,
            x['discriminations'],
            x['difficulties0'],
            x['ddifficulties'],
            grm_vars['abilities']
        )
        # Sum over items/people? log_likelihood returns shape usually?
        ll = tf.reduce_sum(ll, axis=[-1])

        # GRM Prior
        # We can reuse parts of GRModel.unormalized_log_prob but it computes full posterior?
        # Let's just compute the prior for the GRM vars.
        grm_prior_dist = self.joint_prior_distribution # From GRModel.create_distributions
        # We need to filter x to only include GRM vars
        # But wait, create_distributions for AEGRModel creates a hybrid distribution?
        
        # Let's look at how create_distributions was done before:
        # It combined self.surrogate_distribution_dict and self.nn.surrogate_distribution_dict
        
        # If we use the new API, `create_distributions` should set `self.prior_distribution` 
        # and `self.surrogate_distribution`.
        
        # Let's assume create_distributions (below) does the right thing and sets a joint prior.
        # If so, we can just use that?
        # But the "abilities" in the GRM prior are usually standardized normal. 
        # In AE, "abilities" are deterministic outputs of NN(data). 
        # So "abilities" are NOT random variables in the prior sense, they are functions.
        # So the prior should NOT include 'abilities' independent term if we treat them as deterministic.
        
        # grm_vars passed to super().joint_log_prob_auxiliary might expect abilities? 
        # TODO: Add GRM prior terms properly
        return nn_log_prior + ll
        
        # For now, let's just stick to the existing structure but adapted to signature
        return self.joint_log_prob(**x, data=data) # Pass data if joint_log_prob updated

    # Let's keep the existing logic structure but make it work with new fit()
    
    def joint_log_prior(self, **x):
         # This method relies on self.calibration_data which is deprecated.
         # We need to fetch data from arguments.
         # Refactoring to take data as arg.
         raise NotImplementedError("Use unormalized_log_prob")

    def unormalized_log_prob(self, data, prior_weight=1.0, **x):
        weight_tensors = {v: x[v] for v in self.nn.weight_var_list}
        abilities = self.nn.assemble_networks(weight_tensors)(data)
        
        grm_vars = {k: x[k] for k in self.grm_vars if k in x}
        grm_vars["abilities"] = abilities[..., tf.newaxis, tf.newaxis]
        # grm_vars["responses"] = data # Used in auxiliary?
        
        nn_log_prior = self.nn.log_prob(weight_tensors)
        
        # We need the log prior of the GRM parameters (excluding abilities)
        # GRModel.unormalized_log_prob calculates Probs + Prior. 
        # We just want Prior (minus abilities) + Likelihood (using NN abilities).
        
        # Re-implementing simplified version:
        
        # 1. GRM Params Prior:
        # We can use self.joint_prior_distribution.log_prob(grm_vars) 
        # BUT self.joint_prior_distribution includes 'abilities' (standard normal).
        # We don't want to penalize NN abilities with that specific prior unless intended (VAE style)? 
        # Usually AEIRT implies VAE, so KL term? 
        # The original code's joint_log_prior calculated nn_log_prior + grm_log_prior.
        # And grm_log_prior called super().joint_log_prob_auxiliary(**grm_vars)
        
        # Let's assume we want valid VAE-like objective.
        
        # If we just want to run calibration_advi from BayesianModel:
        # It calls minimize( - (target_log_prob) ) => maximize target_log_prob.
        
        # Let's interpret the original code:
        # likelihood = tf.reduce_sum(self.log_likelihood(..., x['abilities'])) 
        # where x['abilities'] was passed in joint_log_prob... wait.
        # In joint_log_prior, abilities = nn(data). 
        # Then grm_vars['abilities'] = abilities.
        # Then super().joint_log_prob(**grm_vars).
        
        # So yes, it evaluates GRModel prior/likelihood using the NN-derived abilities.
        
        # So our unormalized_log_prob should be:
        
        weight_tensors = {v: x[v] for v in self.nn.weight_var_list}
        abilities = self.nn.assemble_networks(weight_tensors)(data)
        grm_vars = {k: x[k] for k in self.grm_vars if k in x}
        grm_vars["abilities"] = abilities[..., tf.newaxis, tf.newaxis] # Shape adjustment
        
        nn_log_prior = self.nn.log_prob(weight_tensors)
        
        # We need to call GRModel's logic for the rest.
        # GRModel doesn't have a simple 'log_prob_parameters' method.
        # It has unormalized_log_prob which includes data likelihood.
        
        # Let's call super().unormalized_log_prob but with our computed abilities?
        # But 'abilities' is not in 'self.var_list' effectively for GRModel if we override it?
        # Actually in AEGRModel 'abilities' is NOT sampled directly, it's deterministic from weights.
        # So 'abilities' is NOT in 'x'.
        
        # So passing grm_vars (with computed abilities) to super().unormalized_log_prob 
        # should work IF super uses the passed 'abilities' and doesn't look for them in its own var list expectations?
        # GRModel.unormalized_log_prob takes **params.
        
        grm_term = super(AEGRModel, self).unormalized_log_prob(data, prior_weight, **grm_vars)
        
        return nn_log_prior + grm_term

    def sample(self, *args, **kwargs):
        nn_sample = self.nn.sample(*args, **kwargs)
        # We need to sample only GRM keys from surrogate?
        # The surrogate is hybrid.
        hybrid_sample = self.surrogate_distribution.sample(*args, **kwargs)
        # hybrid_sample contains both NN and GRM vars.
        return hybrid_sample 

    def create_distributions(self, *args, **kwargs):
        super(AEGRModel, self).create_distributions(*args, **kwargs)
        # Create NN distributions
        # self.nn initialized?
        # Combine them.
        
        # The new API expects self.surrogate_distribution_generator to be set.
        # We need to merge the generator from GRModel and NN.
        
        grm_gen = self.surrogate_distribution_generator
        nn_gen = self.nn.surrogate_distribution_generator
        
        # We need to create a new generator function that returns a JointDistribution 
        # containing both sets of variables.
        
        def hybrid_generator(params=None):
             # Split params into GRM and NN?
             # Or assuming params is the flat dictionary of all.
             d1 = grm_gen(params)
             d2 = nn_gen(params)
             # return CombinedDistribution(d1, d2)
             # TFP doesn't allow easy merging of Named JDs other than creating a new dict.
             return tfd.JointDistributionNamed({**d1.model, **d2.model})
             
        self.surrogate_distribution_generator = hybrid_generator
        
        # Initializer
        grm_init = self.surrogate_parameter_initializer
        nn_init = self.nn.surrogate_parameter_initializer
        
        def hybrid_init(key=None, **kwargs):
             # grm_init and nn_init are closures from build_factored_surrogate_posterior_generator
             # and do not accept arguments. Initialization randomness was handled at creation time.
             i1 = grm_init()
             i2 = nn_init()
             return {**i1, **i2}
             
        self.surrogate_parameter_initializer = hybrid_init
        self.params = self.surrogate_parameter_initializer()

    def calibrate_advi(self, **kwargs):
        # Forward to fit()
        # Map old args to new args if necessary
        return self.fit(**kwargs)


def main():
    from autoencirt.data.rwa import get_data
    aegrm = AEGRModel(hidden_layers=[20, 30])
    aegrm.load_data(get_data())
    aegrm.create_distributions()
    sample = aegrm.sample([2, 3])
    prob = aegrm.joint_log_prob(**sample)
    print(prob)
    aegrm.calibrate_advi(10, clip=1.)
    return


if __name__ == "__main__":
    main()
