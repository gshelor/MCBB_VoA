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
from datetime import date, datetime, timedelta
# import cmdstanpy
# import pymc as pm
# import arviz as az
# import preliz as pz
# import sportsdataverse
# import json

### loading environmental variables (API token, so it's not directly printed in the code for security)
load_dotenv()
### reading in script of functions for cleaning VoA data
exec(open(os.path.join(os.getcwd(), "Scripts", "Python", "MCBB_VoAFuncs.py")).read())

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

### storing which iteration of the VoA this is for the season
voa_num = input("Which release of the VoA most recently happened for this season? ")

### creating directory to store game projections if necessary
if int(voa_num) == 1:
    os.makedirs(os.path.join(os.getcwd(), 'Data', 'VoA' + str(cbb_season), 'Projections', 'CBBD'))
else:
    print('projections directory should already exist')

### setting up API configuration
configuration = cbbd.Configuration(
    host="https://api.collegebasketballdata.com", 
    access_token = os.getenv("CBB_TOKEN"))
api_client = cbbd.ApiClient(configuration)

### creating an instance with the stats, games, and lines API endpoints
gamesapi_instance = cbbd.GamesApi(api_client)


### loading games to be projected
if today_dt.month == 10:
    print("preseason stuff goes here")
    VoAVariables = os.path.join(os.getcwd(), "Data", "VoA" + str(cbb_season), "MCBBVoA" + str(cbb_season) + "VoA" + voa_num + ".csv")
else:
    ### loading upcoming week of games
    VoAVariables = os.path.join(os.getcwd(), "Data", "VoA" + str(cbb_season), "MCBBVoA" + str(cbb_season) + "VoA" + voa_num + ".csv")
    UpcomingGames = gamesapi_instance.get_games(start_date_range = today_dt, end_date_range = today_dt + timedelta(days = 7))


### converting API output to df
if today_dt.month == 10:
    print("preseason stuff")
else:
    ### converting API output to df
    UpcomingGames_df = pl.DataFrame(UpcomingGames).filter(
        pl.col('status') != 'final'
    ).filter(
        (pl.col("home_team").is_in(VoAVariables['team']) & pl.col("away_team").is_in(VoAVariables['team']))
    )
    VoARatings = VoAVariables.select(['team', 'OvrlVoA_MeanRating'])
    HomeVoARating = VoARatings.rename({'team': 'home_team', 'OvrlVoA_MeanRating': 'home_rating'})
    AwayVoARating = VoARatings.rename({'team': 'away_team', 'OvrlVoA_MeanRating': 'away_rating'})



### Assigning Ratings to the corresponding home and away teams
### using a home court advantage of 3 points until I can figure out something better
UpcomingGames_df = UpcomingGames_df.join(HomeVoARating, on = 'home_team', how = 'left')
UpcomingGames_df = UpcomingGames_df.join(AwayVoARating, on = 'away_team', how = 'left').with_columns(
    proj_margin = pl.when(pl.col('neutral_site') == False).then(
        pl.col('away_rating') - (pl.col('home_rating') + 3)
    ).otherwise(
        pl.col('away_rating') - pl.col('home_rating')
    )
).with_columns(
    proj_winner = pl.when(pl.col('proj_margin') < 0).then(pl.col('home_team'))
    .when(pl.col('proj_margin') > 0).then(pl.col('away_team'))
    .otherwise(pl.lit('TIE'))
).select(['id', 'source_id', 'season_label', 'season', 'season_type', 'start_date', 'start_time_tbd', 'neutral_site', 'conference_game', 'game_type', 'tournament', 'game_notes', 'status', 'home_team_id', 'home_team', 'home_conference_id', 'home_conference', 'home_seed', 'away_team_id', 'away_team', 'away_conference_id', 'away_conference', 'away_seed', 'venue_id', 'venue', 'city', 'state', 'home_rating', 'away_rating', 'proj_margin', 'proj_winner'])

### creating dataframe for cbbd submission
CBBDSubmission_df = UpcomingGames_df.select(['id', 'home_team', 'away_team', 'proj_margin']).rename(
    {'id': 'id', 
    'home_team': 'home',
    'away_team': 'away',
    'proj_margin': 'predicted'}
)


### creating string of date combined together so I can identify unique projections compiled during the course of a season
datestring = str(today_dt.year) + str(today_dt.month) + str(today_dt.day)

##### Saving Projections as csvs #####
UpcomingGames_df.write_csv(os.path.join(os.getcwd(), 'Data', 'VoA' + str(cbb_season), 'Projections', 'MCBBVoA' + datestring + 'GameProjections.csv'))
CBBDSubmission_df.write_csv(os.path.join(os.getcwd(), 'Data', 'VoA' + str(cbb_season), 'Projections', 'CBBD', 'MCBBVoA' + datestring + 'GameProjections.csv'))
