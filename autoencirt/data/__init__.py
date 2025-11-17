import importlib.resources

def get_grm_stan():
    return importlib.resources.read_text('autoencirt.data', 'grm.stan')

def get_grm_multi_stan():
    return importlib.resources.read_text('autoencirt.data', 'grm_multi.stan')

def get_grm_stan_path():
    return importlib.resources.files('autoencirt.data').joinpath('grm.stan')

def get_grm_multi_stan_path():
    return importlib.resources.files('autoencirt.data').joinpath('grm_multi.stan')