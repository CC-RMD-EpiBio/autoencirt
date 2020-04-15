/*
GRM in stan using long-format data, to deal with missingness
*/

data {
    int<lower=1> N_person; // number of people
    int<lower=1> N_item; // number of items
    int<lower=1> N_responses; // each response is associated with a person and an item
    int<lower=2, upper = 6> K; // number of categories
    int<lower = 1, upper = K> responses[N_responses]; // response data
    int<lower = 1, upper = N_person> person[N_responses]; 
    int<lower = 1, upper = N_item> item[N_responses];
    item<lower = 1> N_dimensions;
}

parameters{
    real<lower=0> lambda[N_item];
    vector[N_person] theta; // domain score
    ordered[K-1] tau[N_item];
    real mu_tau; // hyperprior parameters
    real<lower=0> sigma_tau;
    real<lower=0> sigma;
}

model{
    lambda ~ cauchy(0,5);
    theta ~ normal(0,sigma);
    sigma ~ cauchy(0,1);
    for (i in 1:N_item){
        for(k in 1:(K-1)){
            target += normal_lpdf(tau[i,k] | mu_tau, sigma_tau);
        }
    }
    mu_tau ~ normal(0,5);
    sigma_tau ~ cauchy(0,5);
    for (i in 1:N_responses){
            target += ordered_logistic_lpmf(responses[i] | theta[person[i]] * lambda[item[i]],tau[item[i]]);
    }
}

generated quantities{
        vector[N_person] log_lik;
        for(i in 1:N_person){
          log_lik[i] = 0;
        }
        for(i in 1:N_responses){
            // log_lik[i] = ordered_logistic_log(responses[i],theta[person[i]]*lambda[item[i]], tau[item[i]]);
            log_lik[person[i]] += ordered_logistic_lpmf(responses[i] | theta[person[i]] * lambda[item[i]],tau[item[i]]);
        }
    }


