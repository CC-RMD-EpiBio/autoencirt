/*
GRM in Stan using long-format data, handling missingness (responses=0)
*/

data {
  int<lower=1> N_person;                 // number of people
  int<lower=1> N_item;                   // number of items
  int<lower=1> N_scales;                 // number of latent scales
  int<lower=1> N_responses;              // rows in long data
  int<lower=1> N_missing;                // number of missing responses
  int<lower=2, upper=5> K;               // number of categories
  array[N_responses] int responses;   // 0 = missing, 1..K observed
  array[N_responses] int<lower=1, upper=N_person> person;
  array[N_responses] int<lower=1, upper=N_item> item;
  array[N_missing] int<lower=1, upper=N_person> person_missing;
  array[N_missing] int<lower=1, upper=N_item> item_missing;
  array[N_item]      int<lower=1, upper=N_scales> scale; // scale membership per item
}

parameters {
  vector<lower=0>[N_item] lambda;        // item discriminations (slopes)
  matrix[N_person, N_scales] theta;      // person abilities per scale
  array[N_item] ordered[K-1] tau;        // item thresholds
  real mu_tau;                           // hyperpriors for thresholds
  real<lower=0> sigma_tau;
  real<lower=0> sigma;                   // sd for theta
}



model {
  // Priors
  lambda ~ cauchy(0, 5);
  to_vector(theta) ~ normal(0, sigma);
  sigma ~ cauchy(0, 1);

  for (i in 1:N_item) {
    for (k in 1:(K-1)) {
      target += normal_lpdf(tau[i, k] | mu_tau, sigma_tau);
    }
  }
  mu_tau ~ normal(0, 5);
  sigma_tau ~ cauchy(0, 5);

  // Likelihood (skip missing; that's equivalent to marginalizing over categories)
  for (r in 1:N_responses) {
    if (responses[r] >= 1 && responses[r] <= K)  {
      int i = item[r];
      int p = person[r];
      int s = scale[i];

      // GRM parameterization: P(Y>=k) = logistic( a * (theta - b_k) )
      // ordered_logistic uses location and cutpoints; multiply both by a = lambda[i]
      target += ordered_logistic_lpmf(
        responses[r] |
        lambda[i] * theta[p, s],
        lambda[i] * tau[i]
      );
    } 
  }
  /*
  for (r in 1:N_missing) {
    int i = item_missing[r];
    int p = person_missing[r];
    int s = scale[i];
    vector[K] logp;
    vector[K] pcat;

    // log p(Y = k) for all categories
    for (k in 1:K) {
        logp[k] = ordered_logistic_lpmf(
        k |
        lambda[i] * theta[p, s],
        lambda[i] * tau[i]
        );
    }
    pcat = softmax(logp); // normalize to probs

    // add expected log-likelihood
    target += dot_product(pcat, logp);
} */
}

generated quantities {
  // Per-response category probabilities and per-person log-lik
  matrix[N_responses, K] prob;    // probabilities p(Y = k)
  vector[N_person] log_lik = rep_vector(0, N_person);

  for (r in 1:N_responses) {
    int i = item[r];
    int p = person[r];
    int s = scale[i];
    vector[K] lpmf_r;

    // compute log pmf for all categories then exponentiate
    for (k in 1:K) {
      lpmf_r[k] = ordered_logistic_lpmf(
        k |
        lambda[i] * theta[p, s],
        lambda[i] * tau[i]
      );
    }
    for (k in 1:K) prob[r, k] = exp(lpmf_r[k]);

    if(responses[r] >= 1 && responses[r] <= K)  {
      log_lik[p] += ordered_logistic_lpmf(
        responses[r] |
        lambda[i] * theta[p, s],
        lambda[i] * tau[i]
      );
    }
  }
}
