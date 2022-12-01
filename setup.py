#!/usr/bin/env python3

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="autoencirt",  # Replace with your own username
    version="0.0.7",
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
        'bayesianquilts@git+https://github.com/mederrata/bayesianquilts.git'
    ],
    python_requires='>=3.8',
    scripts=[
        'autoencirt/scripts/rwas_test.py',
        'autoencirt/scripts/test_nn.py'
    ]
)
