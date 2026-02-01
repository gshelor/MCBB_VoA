##### Working Script for MCBB VoA #####
##### importing modules #####
import cbbd
import polars as pl
import os
import numpy as np
import seaborn as sbn
from dotenv import load_dotenv
from datetime import datetime
# import sportsdataverse
# import json

### loading environmental variables (API token, so it's not directly printed in the code for security)
load_dotenv()

### setting up API configuration
configuration = cbbd.Configuration(
    host="https://api.collegebasketballdata.com", 
    access_token = os.getenv("CBB_TOKEN"))
api_client = cbbd.ApiClient(configuration)

### creating an instance with the stats API
api_instance = cbbd.StatsApi(api_client)
playsapi_instance = cbbd.PlaysApi(api_client)

### getting stats information from CBB data API
TeamSeasonStats_json = api_instance.get_team_season_stats(2025)
# PoopypantsPlays = playsapi_instance.get_dfays_by_date(datetime(2025, 12, 1))

### coercing list output with data in json format to polars dataframe
TeamSeasonStats_df = pl.DataFrame(TeamSeasonStats_json)



TeamStatsCols_json = TeamSeasonStats_df['team_stats'].struct.unnest()
OppTeamStatsCols_json = TeamSeasonStats_df['opponent_stats'].struct.unnest()

for i in np.arange(0, len(TeamStatsCols_json.columns)):
    if i == 0:
        if TeamStatsCols_json[TeamStatsCols_json.columns[i]].dtype == pl.Struct:
            TeamStats_df = TeamStatsCols_json[TeamStatsCols_json.columns[i]].struct.unnest()
            col_prefix = str(TeamStatsCols_json.columns[i])
            TeamStats_df = TeamStats_df.rename(lambda col_name: col_prefix + "_" + col_name)
    else:
        if TeamStatsCols_json[TeamStatsCols_json.columns[i]].dtype == pl.Struct:
            temp_df = TeamStatsCols_json[TeamStatsCols_json.columns[i]].struct.unnest()
            col_prefix = str(TeamStatsCols_json.columns[i])
            temp_df = temp_df.rename(lambda col_name: col_prefix + "_" + col_name)
            TeamStats_df = pl.concat([TeamStats_df, temp_df], how = "horizontal")

### unnesting opposition team stats info
for i in np.arange(0, len(OppTeamStatsCols_json.columns)):
    if i == 0:
        if OppTeamStatsCols_json[OppTeamStatsCols_json.columns[i]].dtype == pl.Struct:
            OppTeamStats = OppTeamStatsCols_json[OppTeamStatsCols_json.columns[i]].struct.unnest()
            col_prefix = str(OppTeamStatsCols_json.columns[i])
            OppTeamStats = OppTeamStats.rename(lambda col_name: "opp_" + col_prefix + "_" + col_name)
    else:
        if OppTeamStatsCols_json[OppTeamStatsCols_json.columns[i]].dtype == pl.Struct:
            temp_df = OppTeamStatsCols_json[OppTeamStatsCols_json.columns[i]].struct.unnest()
            col_prefix = str(OppTeamStatsCols_json.columns[i])
            temp_df = temp_df.rename(lambda col_name: "opp_" + col_prefix + "_" + col_name)
            OppTeamStats = pl.concat([OppTeamStats, temp_df], how = "horizontal")

### taking nested columns out of original team season stats data frame
## binding them to this data frame
VoAVariables = TeamSeasonStats_df.select(
    pl.selectors.by_dtype(pl.Int64, pl.String, pl.Float64)
)
### binding unnested team stats and opposition stats (for each team's opponents) to original team variables df
VoAVariables = pl.concat(items = [VoAVariables, TeamStats_df, OppTeamStats], how = "horizontal")

##### POOPYPANTS TESTING HERE #####

