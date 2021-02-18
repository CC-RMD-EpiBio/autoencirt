#!/usr/bin/env python3

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="autoencirt",  # Replace with your own username
    version="0.0.5",
    author="Josh Chang",
    author_email="josh@mederrata.com",
    description="",
    long_description="Probabilistically-autoencoded horseshoe-disentangled multidomain item-response theory models",
    long_description_content_type="text/markdown",
    url="https://github.com/joshchang/autoencirt",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'dill>=0.3.1.1',
        'matplotlib>=3.1',
        'factor-analyzer',
        'arviz>=0.10.0',
        'numpy>=1.17',
        'pandas>=1.0.0, <1.2.0',
        'scipy>=1.4.1',
        'tensorflow>=2.4.0',
        'tensorflow-probability>=0.12.1',
        'tensorflow-addons>=0.12.0',
        'bayesianquilts@git+https://github.com/mederrata/bayesianquilts.git#egg=bayesianquilts-0.0.1'
    ],
    python_requires='>=3.6',
    scripts=[
        'autoencirt/scripts/rwas_test.py',
        'autoencirt/scripts/test_nn.py'
    ]
)
