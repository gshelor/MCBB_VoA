##### Men's D1 College Basketball Vortex of Projection Version 0.1 #####
### this script takes VoA ratings from MCBB_VoA.py and uses them to create projections for future games over a 1-2 week period
##### importing libraries and setting up python environment and API connection #####
import cbbd
import polars as pl
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from dotenv import load_dotenv
from datetime import date, datetime
import cmdstanpy
import pymc as pm
import arviz as az
import preliz as pz
# import sportsdataverse
# import json

### loading environmental variables (API token, so it's not directly printed in the code for security)
load_dotenv()

### setting up API configuration
configuration = cbbd.Configuration(
    host="https://api.collegebasketballdata.com", 
    access_token = os.getenv("CBB_TOKEN"))
api_client = cbbd.ApiClient(configuration)

### creating an instance with the stats, games, and lines API endpoints
gamesapi_instance = cbbd.GamesApi(api_client)