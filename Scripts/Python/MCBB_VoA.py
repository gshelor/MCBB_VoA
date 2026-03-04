##### Men's D1 College Basketball Vortex of Accuracy Version 0.1 #####
### This script reads in data from the College Basketball Data (CBBD) API and uses it to construct predictive team-strength ratings which are used to make projections for future games.
## possibly uses opponent-adjusted stats created in R using game-level data because the PBP data is too big
##### importing libraries and setting up python environment and API connection #####
import cbbd
import polars as pl
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from dotenv import load_dotenv
from datetime import date, datetime, timedelta
import cmdstanpy
import pymc as pm
import arviz as az
import preliz as pz
import statsmodels.formula.api as smf
import random
# import sportsdataverse
# import json

### loading environmental variables (API token, so it's not directly printed in the code for security)
load_dotenv()

### reading in script of functions for cleaning VoA data
exec(open(os.path.join(os.getcwd(), "Scripts", "Python", "MCBB_VoAFuncs.py")).read())

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
### storing which iteration of the VoA this is for the season
voa_num = input("Which release of the VoA is this for the season? ")


### setting up API configuration
configuration = cbbd.Configuration(
    host="https://api.collegebasketballdata.com", 
    access_token = os.getenv("CBB_TOKEN"))
api_client = cbbd.ApiClient(configuration)

### creating an instance with the stats, games, and lines API endpoints
statsapi_instance = cbbd.StatsApi(api_client)
gamesapi_instance = cbbd.GamesApi(api_client)

# Teams_json = teamsapi_instance.get_teams(season = 2025)
##### Pulling and cleaning Team stats #####
### output is basically a list of json stuff from what I can tell, not that I'm overly familiar with json
## next step after this is coercing it into a polars data frame
if today_dt.month == 10:
    ### getting stats information from CBB data API
    TeamSeasonStats_json_PY2 = statsapi_instance.get_team_season_stats(cbb_season - 2)
    TeamSeasonStats_json_PY1 = statsapi_instance.get_team_season_stats(cbb_season - 1)
elif today_dt.month == 11 and today_dt.day < 15:
    ### reading in PY1 stats saved from preseason
    TeamSeasonStats_json = statsapi_instance.get_team_season_stats(cbb_season)
else:
    ### only current season info
    TeamSeasonStats_json = statsapi_instance.get_team_season_stats(cbb_season)
    VoAVariables = clean_season_stats(TeamSeasonStats_json)
    ### getting games for opponent-adjusted stats
    ### in the future, I'll have a saved csv of games stats for a season so I don't have to hit the API for every game in the season
    Games_json = gamesapi_instance.get_game_teams(start_date_range = datetime(2025, 11, 1), end_date_range = datetime(2025, 11, 25, 23, 59, 59))
    Games2_json = gamesapi_instance.get_game_teams(start_date_range = datetime(2025, 11, 26), end_date_range = datetime(2025, 12, 25, 23, 59, 59))
    Games3_json = gamesapi_instance.get_game_teams(start_date_range = datetime(2025, 12, 26), end_date_range = datetime(2026, 1, 15, 23, 59, 59))
    Games4_json = gamesapi_instance.get_game_teams(start_date_range = datetime(2026, 1, 16), end_date_range = datetime(2026, 1, 31, 23, 59, 59))
    Games5_json = gamesapi_instance.get_game_teams(start_date_range = datetime(2026, 2, 1), end_date_range = datetime(2026, 3, 2, 23, 59, 59))
    # Games6_json = gamesapi_instance.get_game_teams(start_date_range = datetime(2026, 3, 3), end_date_range = datetime(today_dt.year, today_dt.month, today_dt.day - 1, 23, 59, 59))
    CleanGames_df = clean_team_game_stats(Games_json)
    CleanGames2_df = clean_team_game_stats(Games2_json)
    CleanGames3_df = clean_team_game_stats(Games3_json)
    CleanGames4_df = clean_team_game_stats(Games4_json)
    CleanGames5_df = clean_team_game_stats(Games5_json)
    ### binding cleaned games together
    ### polars gives a warning that the is_in() (in the filter) part of the code below is deprecated (as of March 2, 2026 I'm on polars version 1.37.1) but it works and does what I want it to and everybody in the github issue it links to (https://github.com/pola-rs/polars/issues/22149) is mad that it would be producing an issue at all, given that the below usage of is_in() is what they all think would be the most common usage would be (and I agree) so hopefully it stays intact
    AllGamesStatAdj = pl.concat(
        [CleanGames_df, CleanGames2_df, CleanGames3_df, CleanGames4_df, CleanGames5_df], how = "vertical_relaxed").filter(
            (pl.col("team").is_in(VoAVariables['team']) & pl.col("opponent").is_in(VoAVariables['team']))
        ).with_columns(
            pl.when(pl.col('neutral_site') == True).then(0)
            .when(pl.col('neutral_site') == False, pl.col('is_home') == True).then(1)
            .otherwise(-1)
            .alias("hfa")
        )
    AllGamesPaceAdj = AllGamesStatAdj.group_by("game_id").first()
    

### Once data is loaded, I should have: VoAVariables, which is the aggregated season stats, and AllGames, which is the combined dataframe of all games with their various stats columns unnested

### now onto using mixed-effects models to get opponent-adjusted stats
##### Opponent Adjusted Stats using game-level data #####


# Assuming 'teams_master_df' is your existing Polars dataframe with all teams
# teams_master_df = teams_master_df.join(
#     adj_pace_results, 
#     on="team", 
#     how="left"
# )

# # Sort by the fastest teams in the country
# teams_master_df = teams_master_df.sort("adj_pace", descending=True)












### Saving final VoA csv
# VoAVariables.write_csv(os.path.join(data_dir, "MCBBVoA" + str(cbb_season) + "VoA" + voa_num + ".csv"))

##### POOPYPANTS TESTING HERE #####
# VoAVariables.write_csv(os.path.join(os.getcwd(), "poopypants.csv"))
