#!/usr/bin/env python3
import path
from os import system
import pandas as pd

if not path.exists('RWAS/data.csv'):
    system("wget https://openpsychometrics.org/_rawdata/RWAS.zip")
    system("unzip RWAS.zip")

