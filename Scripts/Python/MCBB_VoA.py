##### Men's D1 College Basketball Vortex of Accuracy Version 0.1 #####
### This script reads in data from the College Basketball Data (CBBD) API and uses it to construct predictive team-strength ratings which are used to make projections for future games.
## possibly uses opponent-adjusted stats created in R using play-by-play data from hoopR because the cbbd API doesn't provide pbp in a useful way
## if I had unlimited API calls I guess it'd be worth it but I don't
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
###
voa_num = input("Which release of the VoA is this for the season? ")


### setting up API configuration
configuration = cbbd.Configuration(
    host="https://api.collegebasketballdata.com", 
    access_token = os.getenv("CBB_TOKEN"))
api_client = cbbd.ApiClient(configuration)

### creating an instance with the stats, games, and lines API endpoints
api_instance = cbbd.StatsApi(api_client)
# playsapi_instance = cbbd.PlaysApi(api_client)
# gamesapi_instance = cbbd.GamesApi(api_client)

# Teams_json = teamsapi_instance.get_teams(season = 2025)
# Plays_json = playsapi_instance.get_plays_by_date(datetime(2025, 12, 2))
##### Initial Request for team season stats #####
### output is basically a list of json stuff from what I can tell, not that I'm overly familiar with json
## next step after this is coercing it into a polars data frame
if today_dt.month == 10:
    ### getting stats information from CBB data API
    TeamSeasonStats_json_PY2 = api_instance.get_team_season_stats(cbb_season - 2)
    TeamSeasonStats_json_PY1 = api_instance.get_team_season_stats(cbb_season - 1)
    ### getting games for first 
    # Games_json = gamesapi_instance.get_games(datetime(today_dt.year, 10, 1), datetime(2026, 2, 15))
elif today_dt.month == 11:
    TeamSeasonStats_json = api_instance.get_team_season_stats(cbb_season)




### coercing list output with data in json format to polars dataframe
TeamSeasonStats_df = pl.DataFrame(TeamSeasonStats_json)
# Games_df = pl.DataFrame(Games_json)
# Teams_df = pl.DataFrame(Teams_json)

### unnesting season stats columns for each team and their respective opponents' averages (against them)
TeamStatsCols = TeamSeasonStats_df['team_stats'].struct.unnest()
OppTeamStatsCols = TeamSeasonStats_df['opponent_stats'].struct.unnest()
### more unnesting necessary, and also setting column names so each column is unique
for i in np.arange(0, len(TeamStatsCols.columns)):
    if i == 0:
        if TeamStatsCols[TeamStatsCols.columns[i]].dtype == pl.Struct:
            TeamStats_df = TeamStatsCols[TeamStatsCols.columns[i]].struct.unnest()
            col_prefix = str(TeamStatsCols.columns[i])
            TeamStats_df = TeamStats_df.rename(lambda col_name: col_prefix + "_" + col_name)
    else:
        if TeamStatsCols[TeamStatsCols.columns[i]].dtype == pl.Struct:
            temp_df = TeamStatsCols[TeamStatsCols.columns[i]].struct.unnest()
            col_prefix = str(TeamStatsCols.columns[i])
            temp_df = temp_df.rename(lambda col_name: col_prefix + "_" + col_name)
            TeamStats_df = pl.concat([TeamStats_df, temp_df], how = "horizontal")

### unnesting opposition team stats info
for i in np.arange(0, len(OppTeamStatsCols.columns)):
    if i == 0:
        if OppTeamStatsCols[OppTeamStatsCols.columns[i]].dtype == pl.Struct:
            OppTeamStats_df = OppTeamStatsCols[OppTeamStatsCols.columns[i]].struct.unnest()
            col_prefix = str(OppTeamStatsCols.columns[i])
            OppTeamStats_df = OppTeamStats_df.rename(lambda col_name: "opp_" + col_prefix + "_" + col_name)
    else:
        if OppTeamStatsCols[OppTeamStatsCols.columns[i]].dtype == pl.Struct:
            temp_df = OppTeamStatsCols[OppTeamStatsCols.columns[i]].struct.unnest()
            col_prefix = str(OppTeamStatsCols.columns[i])
            temp_df = temp_df.rename(lambda col_name: "opp_" + col_prefix + "_" + col_name)
            OppTeamStats_df = pl.concat([OppTeamStats_df, temp_df], how = "horizontal")

### taking nested columns out of original team season stats data frame
## binding them to this data frame
VoAVariables = TeamSeasonStats_df.select(
    pl.selectors.by_dtype(pl.Int64, pl.String, pl.Float64)
)
### binding unnested team stats and opposition stats (for each team's opponents) to original team variables df
VoAVariables = pl.concat(items = [VoAVariables, TeamStats_df, OppTeamStats_df], how = "horizontal")

### filtering out non-D1 teams
## doesn't seem to be a great way to do this since there's no Division/classification field output by the CBBD API unless I wanted to do it manually (i do not), but conference luckily seems to serve as a perfectly fine proxy for the same thing since teams not in D1 are just not listed as having a conference
VoAVariables = VoAVariables.filter(
    ~pl.col("conference").is_null()
)


### Saving final VoA csv
# VoAVariables_df.write_csv(os.path.join(data_dir, "MCBBVoA" + str(cbb_season) + "VoA" + voa_num + ".csv"))

##### POOPYPANTS TESTING HERE #####
# VoAVariables.write_csv(os.path.join(os.getcwd(), "poopypants.csv"))
