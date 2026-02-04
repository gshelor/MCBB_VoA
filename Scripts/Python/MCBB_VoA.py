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
# playsapi_instance = cbbd.PlaysApi(api_client)
# teamsapi_instance = cbbd.TeamsApi(api_client)

### getting stats information from CBB data API
TeamSeasonStats_json = api_instance.get_team_season_stats(2025)
# Teams_json = teamsapi_instance.get_teams(season = 2025)
# Plays_json = playsapi_instance.get_dfays_by_date(datetime(2025, 12, 1))

### coercing list output with data in json format to polars dataframe
TeamSeasonStats_df = pl.DataFrame(TeamSeasonStats_json)
# Teams_df = pl.DataFrame(Teams_json)


### unnesting 
TeamStatsCols = TeamSeasonStats_df['team_stats'].struct.unnest()
OppTeamStatsCols = TeamSeasonStats_df['opponent_stats'].struct.unnest()

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

##### POOPYPANTS TESTING HERE #####
VoAVariables.write_csv(os.path.join(os.getcwd(), "poopypants.csv"))
