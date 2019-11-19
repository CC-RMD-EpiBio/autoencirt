#!/usr/bin/env python3

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="joshchang", # Replace with your own username
    version="0.0.1",
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
    python_requires='>=3.7',
    entry_points="""
        [console_scripts]
        fitrwas=autoencoder_irt.tools.cli.rwas:main
    """
)