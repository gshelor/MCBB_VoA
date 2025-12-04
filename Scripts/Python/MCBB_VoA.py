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

load_dotenv()


configuration = cbbd.Configuration(host="https://api.collegebasketballdata.com", access_token = os.getenv("CBB_TOKEN"))
api_client = cbbd.ApiClient(configuration)

api_instance = cbbd.StatsApi(api_client)
playsapi_instance = cbbd.PlaysApi(api_client)

TeamSeasonStats_json = api_instance.get_team_season_stats(2025)
PoopypantsPlays = playsapi_instance.get_plays_by_date(datetime(2025, 12, 1))


TeamSeasonStats_pl = pl.DataFrame(TeamSeasonStats_json)



TeamStatsCols_json = TeamSeasonStats_pl['team_stats'].struct.unnest()
OppTeamStatsCols_json = TeamSeasonStats_pl['opponent_stats'].struct.unnest()

for i in np.arange(0, len(TeamStatsCols_json.columns)):
    if i == 0:
        TeamStats = TeamStatsCols_json[TeamStatsCols_json.columns[i]].struct.unnest()
    else:
        temp_df = TeamStatsCols_json[TeamStatsCols_json.columns[i]].struct.unnest()
        TeamStats = pl.concat([TeamStats, temp_df], how = "horizontal")


TeamSeasonStats_pl2 = pl.concat(items = [TeamSeasonStats_pl, TeamStatsCols_json, OppTeamStatsCols_json], how = "horizontal")



