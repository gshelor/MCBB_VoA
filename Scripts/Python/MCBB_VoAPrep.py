##### Men's D1 College Basketball Vortex of Accuracy Prep Script #####
### This script prepares the df that will be used to attach adjusted stats calculated in R with hoopR PBP data
### the dataframe output here will only contain team information, no stats yet, those will be compiled in MCBB_VoA.py and MCBB_VoA.R
### should only need to be run once, when the first ratings are produced at the start of each season
### After that a previous VoA ratings csv will be read in in the R script to obtain the relevant teams for opponent-adjusted stats
### other stats gotten from CBBD API in MCBB_VoA.py

##### importing libraries and setting up python environment and API connection #####
import cbbd
import polars as pl
import os
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sbn
from dotenv import load_dotenv
from datetime import date, datetime
# import sportsdataverse
# import json

### loading environmental variables (API token, so it's not directly printed in the code for security)
load_dotenv()

### storing values for season, month, day maybe, and which iteration of the VoA this is (for filenames, probably)
## cbb_season to be used as year/season input for team season stats grab from API
### Today's date
today_dt = datetime.now()
if today_dt.month >= 5:
    ### for preseason ratings and ratings from November-December of current season
    cbb_season = today_dt.year
    cbb_season_str = str(today_dt.year) + "/" + str(today_dt.year + 1)
else:
    ### for VoA ratings being made after January
    cbb_season = today_dt.year - 1
    cbb_season_str = str(today_dt.year - 1) + "/" + str(today_dt.year)
### identifying which orientation of the VoA this is
## may not need this for this script
voa_num = input("Which release of the VoA is this for the season? ")

### creating data directory
data_dir = os.path.join(os.getcwd(), "Data", "VoA" + str(cbb_season))
os.mkdir(data_dir)


### setting up API configuration
configuration = cbbd.Configuration(
    host="https://api.collegebasketballdata.com", 
    access_token = os.getenv("CBB_TOKEN"))
api_client = cbbd.ApiClient(configuration)

### creating an instance with the stats, games, and lines API endpoints
api_instance = cbbd.TeamsApi(api_client)

### getting team info
VoATeams = api_instance.get_teams(season = cbb_season)

VoATeams_df = pl.DataFrame(VoATeams)

VoATeams_df.write_csv(os.path.join(data_dir, "MCBBVoA" + str(cbb_season) + "Teams.csv"))
