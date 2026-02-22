##### D1 Men's College Basketball Vortex of Accuracy Model Accuracy #####
##### importing libraries and setting up python environment and API connection #####
import cbbd
import polars as pl
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from dotenv import load_dotenv
from datetime import date, datetime
# import cmdstanpy
# import pymc as pm
# import arviz as az
# import preliz as pz
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

voa_num = input("Which release of the VoA is being evaluated for the season? ")


### setting up API configuration
configuration = cbbd.Configuration(
    host="https://api.collegebasketballdata.com", 
    access_token = os.getenv("CBB_TOKEN"))
api_client = cbbd.ApiClient(configuration)

### creating an instance with the lines API endpoint
linesapi_instance = cbbd.LinesApi(api_client)

### getting games from relevant time period
if today_dt.month == 10 or today_dt.month == 11:
    Lines_json = linesapi_instance.get_lines(start_date_range = datetime(2025, 10, 1), end_date_range = datetime(2026, 2, 15))
else:
    Lines_json = linesapi_instance.get_lines(start_date_range = datetime(2025, 10, 1), end_date_range = datetime(2026, 2, 15))



Lines_df = pl.DataFrame(Lines_json, strict = False)

### the column that contains the actual lines for the games is a column of lists of "structs" so I take all the items out of the structs and make them their own column, then pivot so that games with multiple spreads will have multiple columns for each spread provider
Lines_df = Lines_df.explode('lines').unnest('lines')

Lines_df = Lines_df.pivot(
    values=["spread", "over_under"],
    index=["game_id", "season", "season_type", "start_date", "home_team_id", "home_team", "home_conference", "home_score", "away_team_id", "away_team", "away_conference", "away_score"], # game identifier
    on="provider"
).with_columns(
    mean_spread = pl.mean_horizontal(pl.selectors.starts_with("spread"), ignore_nulls = True)
)