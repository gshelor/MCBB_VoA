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
import multiprocessing
import pymc as pm
import arviz as az
import preliz as pz
import statsmodels.formula.api as smf
import random
from great_tables import GT
import selenium
# import bambi as bmb
from pymer4.models import lmer
# import sportsdataverse
# import json

### loading environmental variables (API token, so it's not directly printed in the code for security)
load_dotenv()

### reading in script of functions for cleaning VoA data and occasional ad-hoc game projections
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

### if this is the first VoA, all of my file pathways where I save various files and such will be created here
## it should create any nested pathways as well, meaning any folders in the pathway that don't exist yet will also be created
## inshallah motherfuckers
if int(voa_num) == 1:
    accuracy_metrics_path = os.path.join(os.getcwd(), "Data", "VoA" + str(cbb_season), "AccuracyMetrics")
    game_stats_path = os.path.join(os.getcwd(), "Data", "VoA" + str(cbb_season), "GameStats")
    projections_cbbd_path = os.path.join(os.getcwd(), "Data", "VoA" + str(cbb_season), "Projections", "CBBD")
    fitted_models_path = os.path.join(os.getcwd(), "Data", "FittedModels")
    py_data_path = os.path.join(os.getcwd(), "Data", "VoA" + str(cbb_season), "PYData")
    outputs_path = os.path.join(os.getcwd(), "Outputs", "VoA" + str(cbb_season))
    new_filepaths = [accuracy_metrics_path, game_stats_path, projections_cbbd_path, fitted_models_path, py_data_path, outputs_path]
    [os.makedirs(path, exist_ok = True) for path in new_filepaths]
else:
    print("file pathways should exist already")

### setting up API configuration
configuration = cbbd.Configuration(
    host="https://api.collegebasketballdata.com", 
    access_token = os.getenv("CBB_TOKEN"))
api_client = cbbd.ApiClient(configuration)

### creating an instance with the stats, games, and lines API endpoints
statsapi_instance = cbbd.StatsApi(api_client)
gamesapi_instance = cbbd.GamesApi(api_client)
teamsapi_instance = cbbd.TeamsApi(api_client)
# recruitapi_instance = cbbd.RecruitingApi(api_client)
# poopypants = pl.DataFrame(recruitapi_instance.get_portal_transfers(2026))

##### Pulling and cleaning Team and Game stats #####
### output is basically a list of json stuff from what I can tell, not that I'm overly familiar with json
## next step after this is coercing it into a polars data frame
if today_dt.month == 10:
    ### getting stats information from CBB data API
    TeamSeasonStats_json_PY3 = statsapi_instance.get_team_season_stats(cbb_season - 3)
    VoATrain_PY3 = clean_season_stats(TeamSeasonStats_json_PY3)
    TeamSeasonStats_json_PY2 = statsapi_instance.get_team_season_stats(cbb_season - 2)
    VoATrain_PY2 = clean_season_stats(TeamSeasonStats_json_PY2)
    TeamSeasonStats_json_PY1 = statsapi_instance.get_team_season_stats(cbb_season - 1)
    VoATrain_PY1 = clean_season_stats(TeamSeasonStats_json_PY1)
    ### getting team info to make sure I have all D1 teams for the current season that have previous stats
    Teams_json = teamsapi_instance.get_teams(season = cbb_season)
    Teams_df = pl.DataFrame(Teams_json).rename({"school": "team"})

    ### the logos column coerces to a list type to I'm moving each unique logo URL to its own column
    ### then, filtering the df of current D1 teams to only include teams with 3 years of PY data, so that I can 
    max_width = Teams_df['logos'].list.len().max()
    VoATeams = Teams_df.with_columns(
        pl.col('logos').list.get(x, null_on_oob=True).alias(f'logo_{x}') for x in range(max_width)).filter(
            pl.col('team').is_in(VoATrain_PY1['team'].implode()) & pl.col('team').is_in(VoATrain_PY2['team'].implode()) & pl.col('team').is_in(VoATrain_PY3['team'].implode())
        )

    ### converting train season stats dfs to VoAStats dfs so they can be left joined
    VoAStats_PY3 = voa_train_to_stats("_PY3", VoATrain_PY3)
    VoAStats_PY2 = voa_train_to_stats("_PY2", VoATrain_PY2)
    VoAStats_PY1 = voa_train_to_stats("_PY1", VoATrain_PY1)
    ### combining PY stats to one df for eventual calculation of weighted stats for model inference
    VoAStats = VoATeams.join(
        VoAStats_PY3, on = "team", how = "left"
    ).join(
        VoAStats_PY2, on = "team", how = "left"
    ).join(
        VoAStats_PY1, on = "team", how = "left"
    )
    ### pulling historical game info from previous seasons to use to calculate adjusted stats
    Games_PY3 = get_PY_game_data(cbb_season - 3)
    Games_PY2 = get_PY_game_data(cbb_season - 2)
    Games_PY1 = get_PY_game_data(cbb_season - 1)
    ### creating games dfs for adjusted pace modeling
    GamesPaceAdj_PY3 = Games_PY3.group_by("game_id").first()
    GamesPaceAdj_PY2 = Games_PY2.group_by("game_id").first()
    GamesPaceAdj_PY1 = Games_PY1.group_by("game_id").first()
elif today_dt.month == 11 and today_dt.day < 15:
    ### reading in PY1 and PY2 stats saved from preseason VoA
    PYData = pl.read_parquet()
    TeamSeasonStats_json = statsapi_instance.get_team_season_stats(cbb_season)
else:
    ### only current season info
    TeamSeasonStats_json = statsapi_instance.get_team_season_stats(cbb_season)
    VoAStats = clean_season_stats(TeamSeasonStats_json)
    ### getting games for opponent-adjusted stats
    ### reading in df of previous games so I don't have to pull in old games every time I rerun the VoA
    AllGamesStatAdj = pl.read_parquet(os.path.join(os.getcwd(), "Data", "VoA" + str(cbb_season), "GameStats", "MCBBD1VoA" + str(cbb_season) + "Games.parquet"))
    Games_json = gamesapi_instance.get_game_teams(start_date_range = max(AllGamesStatAdj['start_date']), end_date_range = datetime(today_dt.year, today_dt.month, today_dt.day - 1, 23, 59, 59))
    # Games2_json = gamesapi_instance.get_game_teams(start_date_range = datetime(2025, 11, 26), end_date_range = datetime(2025, 12, 25, 23, 59, 59))
    # Games3_json = gamesapi_instance.get_game_teams(start_date_range = datetime(2025, 12, 26), end_date_range = datetime(2026, 1, 15, 23, 59, 59))
    # Games4_json = gamesapi_instance.get_game_teams(start_date_range = datetime(2026, 1, 16), end_date_range = datetime(2026, 1, 31, 23, 59, 59))
    # Games5_json = gamesapi_instance.get_game_teams(start_date_range = datetime(2026, 2, 1), end_date_range = datetime(2026, 3, 2, 23, 59, 59))
    # Games6_json = gamesapi_instance.get_game_teams(start_date_range = datetime(2026, 3, 3), end_date_range = datetime(today_dt.year, today_dt.month, today_dt.day - 1, 23, 59, 59))
    CleanGames_df = clean_team_game_stats(Games_json)
    # CleanGames2_df = clean_team_game_stats(Games2_json)
    # CleanGames3_df = clean_team_game_stats(Games3_json)
    # CleanGames4_df = clean_team_game_stats(Games4_json)
    # CleanGames5_df = clean_team_game_stats(Games5_json)
    # CleanGames6_df = clean_team_game_stats(Games6_json)
    ### binding cleaned games together
    ### polars gives a warning that the is_in() in the filter part of the code below is deprecated (as of March 2, 2026 I'm on polars version 1.37.1) but it works and does what I want it to and everybody in the github issue it links to (https://github.com/pola-rs/polars/issues/22149) is mad that it would be producing an issue at all, given that the below usage of is_in() is what they all think would be the most common usage would be (and I agree) so hopefully it stays intact
    AllGamesStatAdj = pl.concat(
        [AllGamesStatAdj, CleanGames_df], how = "vertical_relaxed").filter(
            (pl.col("team").is_in(VoAStats['team'].implode()) & pl.col("opponent").is_in(VoAStats['team'].implode()))
        )
    ### saving AllGamesStatAdj so I don't have to hit the API for every game every time I run the script
    AllGamesStatAdj.write_parquet(os.path.join(os.getcwd(), "Data", "VoA" + str(cbb_season), "GameStats", "MCBBD1VoA" + str(cbb_season) + "Games.parquet"))
    ### Adding new columns now that I've saved it so new games can be added to the df
    AllGamesStatAdj = AllGamesStatAdj.with_columns(
            pl.when(pl.col('neutral_site') == True).then(0)
            .when(pl.col('neutral_site') == False, pl.col('is_home') == True).then(1)
            .otherwise(-1)
            .alias("hfa"),
            points_per_poss = pl.col("points_total") / pl.col("possessions"),
            opp_points_per_poss = pl.col("opp_points_total") / pl.col("opp_possessions")
        )
    AllGamesPaceAdj = AllGamesStatAdj.group_by("game_id").first()
    

### Once data is loaded, I should have: VoAStats, which is the aggregated season stats, and AllGames, which is the combined dataframe of all games with their various stats columns unnested

### now onto using mixed-effects models to get opponent-adjusted stats
##### Opponent Adjusted Stats using game-level data #####
if today_dt.month == 10:
    ### calling function to calculate opponent-adjusted stats for each PY and return them all in one polars dataframe
    ## this function is also in the VoAFuncs.py script
    # function call: opponent_adjustments(pace_df, full_df)
    ### performing opponent adjustments for each year, joining them to their appropriate VoATrain df, then renaming the columns so they can be joined to VoAStats so weighted stats can be calculated for Stan-based inference later when modeling team ratings
    ## PY3
    AdjStats_PY3 = opponent_adjustments(GamesPaceAdj_PY3, Games_PY3)
    VoATrain_PY3 = VoATrain_PY3.join(AdjStats_PY3, on = "team", how = "left")
    AdjStats_PY3 = AdjStats_PY3.rename(lambda col_name: col_name + "_PY3").rename({"team_PY3": "team"})
    ## PY2
    AdjStats_PY2 = opponent_adjustments(GamesPaceAdj_PY2, Games_PY2)
    VoATrain_PY2 = VoATrain_PY2.join(AdjStats_PY2, on = "team", how = "left")
    AdjStats_PY2 = AdjStats_PY2.rename(lambda col_name: col_name + "_PY2").rename({"team_PY2": "team"})
    ## PY1
    AdjStats_PY1 = opponent_adjustments(GamesPaceAdj_PY1, Games_PY1)
    VoATrain_PY1 = VoATrain_PY1.join(AdjStats_PY1, on = "team", how = "left")
    AdjStats_PY1 = AdjStats_PY1.rename(lambda col_name: col_name + "_PY1").rename({"team_PY1": "team"})
    ### concatenating VoATrain dfs so they can be used in Stan model below
    VoATrain = pl.concat([VoATrain_PY1, VoATrain_PY2, VoATrain_PY3], how = "vertical_relaxed").with_columns(
        adjoff_ppg = pl.col("adjoff_points_per_poss") * pl.col("adj_pace").mean(),
        adjdef_ppg = pl.col("adjdef_opp_points_per_poss") * pl.col("adj_pace").mean()
    )
    ### joining opponent-adjusted stats to main VoAStats df 
    VoAStats = VoAStats.join(AdjStats_PY3, on = "team").with_columns(
        adjoff_ppg_PY3 = pl.col("adjoff_points_per_poss_PY3") * pl.col("adj_pace_PY3").mean(),
        adjdef_ppg_PY3 = pl.col("adjdef_opp_points_per_poss_PY3") * pl.col("adj_pace_PY3").mean()
    ).join(AdjStats_PY2, on = "team").with_columns(
        adjoff_ppg_PY2 = pl.col("adjoff_points_per_poss_PY2") * pl.col("adj_pace_PY2").mean(),
        adjdef_ppg_PY2 = pl.col("adjdef_opp_points_per_poss_PY2") * pl.col("adj_pace_PY2").mean()
    ).join(AdjStats_PY1, on = "team").with_columns(
        adjoff_ppg_PY1 = pl.col("adjoff_points_per_poss_PY1") * pl.col("adj_pace_PY1").mean(),
        adjdef_ppg_PY1 = pl.col("adjdef_opp_points_per_poss_PY1") * pl.col("adj_pace_PY1").mean()
    )
elif today_dt.month == 11:
    print("might do something here too")
else:
    ### calling function to calculate opponent-adjusted stats and return them all in one polars dataframe
    ## this function is also in the VoAFuncs.py script
    # function call: opponent_adjustments(pace_df, full_df)
    AdjStats = opponent_adjustments(AllGamesPaceAdj, AllGamesStatAdj)
    VoAStats = VoAStats.join(AdjStats, on = "team").with_columns(
        adjoff_ppg = pl.col("adjoff_points_per_poss") * pl.col("adj_pace").mean(),
        adjdef_ppg = pl.col("adjdef_opp_points_per_poss") * pl.col("adj_pace").mean()
    )


##### Calculating Weighted Variables of Stats to be used in Stan Models #####
### setting PY Weights based on date
## as we move further into the season, weights of less recent seasons are decreased and/or removed
if today_dt.month == 10:
    PY3_weight = 0.05
    PY2_weight = 0.2
    PY1_weight = 0.75
elif today_dt.month == 11 and today_dt.day <= 15:
    PY2_weight = 0.05
    PY1_weight = 0.85
    CY_weight = 0.1
elif today_dt.month == 11 and today_dt.day > 15:
    PY1_weight = 0.8
    CY_weight = 0.2
elif today_dt.month == 12 and today_dt.day <= 8:
    PY1_weight = 0.6
    CY_weight = 0.4
elif today_dt.month == 12 and today_dt.day <= 15:
    PY1_weight = 0.5
    CY_weight = 0.5
elif today_dt.month == 12 and today_dt.day <= 22:
    PY1_weight = 0.3
    CY_weight = 0.7
elif today_dt.month == 12 and today_dt.day <= 31:
    PY1_weight = 0.1
    CY_weight = 0.9
else:
    print("no weighting necessary, only current season data being used")

### actually calculating weighted variables now, using weights established above
### only going to weight the variables actually used in the model formulas because I'm lazy and also if I don't have an average stat ranking like the VoA Output in my football models, what's the point of overdoing it
if today_dt.month == 10:
    ##### preseason VoA weighted variables #####
    VoAStats = VoAStats.with_columns(
        ### offensive stats
        weighted_adjoff_ppg = (pl.col("adjoff_ppg_PY3") * PY3_weight) + (pl.col("adjoff_ppg_PY2") * PY2_weight) + (pl.col("adjoff_ppg_PY1") * PY1_weight),
        weighted_adjoff_points_per_poss = (pl.col("adjoff_points_per_poss_PY3") * PY3_weight) + (pl.col("adjoff_points_per_poss_PY2") * PY2_weight) + (pl.col("adjoff_points_per_poss_PY1") * PY1_weight),
        weighted_adjoff_assists = (pl.col("adjoff_assists_PY3") * PY3_weight) + (pl.col("adjoff_assists_PY2") * PY2_weight) + (pl.col("adjoff_assists_PY1") * PY1_weight),
        weighted_adjoff_true_shooting = (pl.col("adjoff_true_shooting_PY3") * PY3_weight) + (pl.col("adjoff_true_shooting_PY2") * PY2_weight) + (pl.col("adjoff_points_per_poss_PY1") * PY1_weight),
        weighted_adjoff_opp_blocks = (pl.col("adjoff_opp_blocks_PY3") * PY3_weight) + (pl.col("adjoff_opp_blocks_PY2") * PY2_weight) + (pl.col("adjoff_opp_blocks_PY1") * PY1_weight),
        weighted_adjoff_opp_steals = (pl.col("adjoff_opp_steals_PY3") * PY3_weight) + (pl.col("adjoff_opp_steals_PY2") * PY2_weight) + (pl.col("adjoff_opp_steals_PY1") * PY1_weight),
        weighted_adjoff_field_goals_pct = (pl.col("adjoff_field_goals_pct_PY3") * PY3_weight) + (pl.col("adjoff_field_goals_pct_PY2") * PY2_weight) + (pl.col("adjoff_field_goals_pct_PY1") * PY1_weight),
        weighted_adjoff_two_point_field_goals_pct = (pl.col("adjoff_two_point_field_goals_pct_PY3") * PY3_weight) + (pl.col("adjoff_two_point_field_goals_pct_PY2") * PY2_weight) + (pl.col("adjoff_two_point_field_goals_pct_PY1") * PY1_weight),
        weighted_adjoff_three_point_field_goals_pct = (pl.col("adjoff_three_point_field_goals_pct_PY3") * PY3_weight) + (pl.col("adjoff_three_point_field_goals_pct_PY2") * PY2_weight) + (pl.col("adjoff_three_point_field_goals_pct_PY1") * PY1_weight),
        weighted_adjoff_free_throws_pct = (pl.col("adjoff_free_throws_pct_PY3") * PY3_weight) + (pl.col("adjoff_free_throws_pct_PY2") * PY2_weight) + (pl.col("adjoff_free_throws_pct_PY1") * PY1_weight),
        weighted_adjoff_rebounds_offensive = (pl.col("adjoff_rebounds_offensive_PY3") * PY3_weight) + (pl.col("adjoff_rebounds_offensive_PY2") * PY2_weight) + (pl.col("adjoff_rebounds_offensive_PY1") * PY1_weight),
        weighted_adjoff_turnovers_total = (pl.col("adjoff_turnovers_total_PY3") * PY3_weight) + (pl.col("adjoff_turnovers_total_PY2") * PY2_weight) + (pl.col("adjoff_turnovers_total_PY1") * PY1_weight),
        ### defensive stats
        weighted_adjdef_ppg = (pl.col("adjdef_ppg_PY3") * PY3_weight) + (pl.col("adjdef_ppg_PY2") * PY2_weight) + (pl.col("adjdef_ppg_PY1") * PY1_weight),
        weighted_adjdef_opp_points_per_poss = (pl.col("adjdef_opp_points_per_poss_PY3") * PY3_weight) + (pl.col("adjdef_opp_points_per_poss_PY2") * PY2_weight) + (pl.col("adjdef_opp_points_per_poss_PY1") * PY1_weight),
        weighted_adjdef_opp_assists = (pl.col("adjdef_opp_assists_PY3") * PY3_weight) + (pl.col("adjdef_opp_assists_PY2") * PY2_weight) + (pl.col("adjdef_opp_assists_PY1") * PY1_weight),
        weighted_adjdef_opp_true_shooting = (pl.col("adjdef_opp_true_shooting_PY3") * PY3_weight) + (pl.col("adjdef_opp_true_shooting_PY2") * PY2_weight) + (pl.col("adjdef_opp_points_per_poss_PY1") * PY1_weight),
        weighted_adjdef_blocks = (pl.col("adjdef_blocks_PY3") * PY3_weight) + (pl.col("adjdef_blocks_PY2") * PY2_weight) + (pl.col("adjdef_blocks_PY1") * PY1_weight),
        weighted_adjdef_steals = (pl.col("adjdef_steals_PY3") * PY3_weight) + (pl.col("adjdef_steals_PY2") * PY2_weight) + (pl.col("adjdef_steals_PY1") * PY1_weight),
        weighted_adjdef_opp_field_goals_pct = (pl.col("adjdef_opp_field_goals_pct_PY3") * PY3_weight) + (pl.col("adjdef_opp_field_goals_pct_PY2") * PY2_weight) + (pl.col("adjdef_opp_field_goals_pct_PY1") * PY1_weight),
        weighted_adjdef_opp_two_point_field_goals_pct = (pl.col("adjdef_opp_two_point_field_goals_pct_PY3") * PY3_weight) + (pl.col("adjdef_opp_two_point_field_goals_pct_PY2") * PY2_weight) + (pl.col("adjdef_opp_two_point_field_goals_pct_PY1") * PY1_weight),
        weighted_adjdef_opp_three_point_field_goals_pct = (pl.col("adjdef_opp_three_point_field_goals_pct_PY3") * PY3_weight) + (pl.col("adjdef_opp_three_point_field_goals_pct_PY2") * PY2_weight) + (pl.col("adjdef_opp_three_point_field_goals_pct_PY1") * PY1_weight),
        weighted_adjdef_opp_free_throws_pct = (pl.col("adjdef_opp_free_throws_pct_PY3") * PY3_weight) + (pl.col("adjdef_opp_free_throws_pct_PY2") * PY2_weight) + (pl.col("adjdef_opp_free_throws_pct_PY1") * PY1_weight),
        weighted_adjdef_rebounds_defensive = (pl.col("adjdef_rebounds_defensive_PY3") * PY3_weight) + (pl.col("adjdef_rebounds_defensive_PY2") * PY2_weight) + (pl.col("adjdef_rebounds_defensive_PY1") * PY1_weight),
        weighted_adjdef_opp_turnovers_total = (pl.col("adjdef_opp_turnovers_total_PY3") * PY3_weight) + (pl.col("adjdef_opp_turnovers_total_PY2") * PY2_weight) + (pl.col("adjdef_opp_turnovers_total_PY1") * PY1_weight)
    )
elif today_dt.month == 11 and today_dt.day <= 15:
    ##### pre-Nov 16 VoA weighted variables #####
    ### PY2 and PY1 included still
    VoAStats = VoAStats.with_columns(
        ### offensive stats
        weighted_adjoff_ppg = (pl.col("adjoff_ppg") * CY_weight) + (pl.col("adjoff_ppg_PY2") * PY2_weight) + (pl.col("adjoff_ppg_PY1") * PY1_weight),
        weighted_adjoff_points_per_poss = (pl.col("adjoff_points_per_poss") * CY_weight) + (pl.col("adjoff_points_per_poss_PY2") * PY2_weight) + (pl.col("adjoff_points_per_poss_PY1") * PY1_weight),
        weighted_adjoff_assists = (pl.col("adjoff_assists") * CY_weight) + (pl.col("adjoff_assists_PY2") * PY2_weight) + (pl.col("adjoff_assists_PY1") * PY1_weight),
        weighted_adjoff_true_shooting = (pl.col("adjoff_true_shooting") * CY_weight) + (pl.col("adjoff_true_shooting_PY2") * PY2_weight) + (pl.col("adjoff_points_per_poss_PY1") * PY1_weight),
        weighted_adjoff_opp_blocks = (pl.col("adjoff_opp_blocks") * CY_weight) + (pl.col("adjoff_opp_blocks_PY2") * PY2_weight) + (pl.col("adjoff_opp_blocks_PY1") * PY1_weight),
        weighted_adjoff_opp_steals = (pl.col("adjoff_opp_steals") * CY_weight) + (pl.col("adjoff_opp_steals_PY2") * PY2_weight) + (pl.col("adjoff_opp_steals_PY1") * PY1_weight),
        weighted_adjoff_field_goals_pct = (pl.col("adjoff_field_goals_pct") * CY_weight) + (pl.col("adjoff_field_goals_pct_PY2") * PY2_weight) + (pl.col("adjoff_field_goals_pct_PY1") * PY1_weight),
        weighted_adjoff_two_point_field_goals_pct = (pl.col("adjoff_two_point_field_goals_pct") * CY_weight) + (pl.col("adjoff_two_point_field_goals_pct_PY2") * PY2_weight) + (pl.col("adjoff_two_point_field_goals_pct_PY1") * PY1_weight),
        weighted_adjoff_three_point_field_goals_pct = (pl.col("adjoff_three_point_field_goals_pct") * CY_weight) + (pl.col("adjoff_three_point_field_goals_pct_PY2") * PY2_weight) + (pl.col("adjoff_three_point_field_goals_pct_PY1") * PY1_weight),
        weighted_adjoff_free_throws_pct = (pl.col("adjoff_free_throws_pct") * CY_weight) + (pl.col("adjoff_free_throws_pct_PY2") * PY2_weight) + (pl.col("adjoff_free_throws_pct_PY1") * PY1_weight),
        weighted_adjoff_rebounds_offensive = (pl.col("adjoff_rebounds_offensive") * CY_weight) + (pl.col("adjoff_rebounds_offensive_PY2") * PY2_weight) + (pl.col("adjoff_rebounds_offensive_PY1") * PY1_weight),
        weighted_adjoff_turnovers_total = (pl.col("adjoff_turnovers_total") * CY_weight) + (pl.col("adjoff_turnovers_total_PY2") * PY2_weight) + (pl.col("adjoff_turnovers_total_PY1") * PY1_weight),
        ### defensive stats
        weighted_adjdef_ppg = (pl.col("adjdef_ppg") * CY_weight) + (pl.col("adjdef_ppg_PY2") * PY2_weight) + (pl.col("adjdef_ppg_PY1") * PY1_weight),
        weighted_adjdef_opp_points_per_poss = (pl.col("adjdef_opp_points_per_poss") * CY_weight) + (pl.col("adjdef_opp_points_per_poss_PY2") * PY2_weight) + (pl.col("adjdef_opp_points_per_poss_PY1") * PY1_weight),
        weighted_adjdef_opp_assists = (pl.col("adjdef_opp_assists") * CY_weight) + (pl.col("adjdef_opp_assists_PY2") * PY2_weight) + (pl.col("adjdef_opp_assists_PY1") * PY1_weight),
        weighted_adjdef_opp_true_shooting = (pl.col("adjdef_opp_true_shooting") * CY_weight) + (pl.col("adjdef_opp_true_shooting_PY2") * PY2_weight) + (pl.col("adjdef_opp_points_per_poss_PY1") * PY1_weight),
        weighted_adjdef_blocks = (pl.col("adjdef_blocks") * CY_weight) + (pl.col("adjdef_blocks_PY2") * PY2_weight) + (pl.col("adjdef_blocks_PY1") * PY1_weight),
        weighted_adjdef_steals = (pl.col("adjdef_steals") * CY_weight) + (pl.col("adjdef_steals_PY2") * PY2_weight) + (pl.col("adjdef_steals_PY1") * PY1_weight),
        weighted_adjdef_opp_field_goals_pct = (pl.col("adjdef_opp_field_goals_pct") * CY_weight) + (pl.col("adjdef_opp_field_goals_pct_PY2") * PY2_weight) + (pl.col("adjdef_opp_field_goals_pct_PY1") * PY1_weight),
        weighted_adjdef_opp_two_point_field_goals_pct = (pl.col("adjdef_opp_two_point_field_goals_pct") * CY_weight) + (pl.col("adjdef_opp_two_point_field_goals_pct_PY2") * PY2_weight) + (pl.col("adjdef_opp_two_point_field_goals_pct_PY1") * PY1_weight),
        weighted_adjdef_opp_three_point_field_goals_pct = (pl.col("adjdef_opp_three_point_field_goals_pct") * CY_weight) + (pl.col("adjdef_opp_three_point_field_goals_pct_PY2") * PY2_weight) + (pl.col("adjdef_opp_three_point_field_goals_pct_PY1") * PY1_weight),
        weighted_adjdef_opp_free_throws_pct = (pl.col("adjdef_opp_free_throws_pct") * CY_weight) + (pl.col("adjdef_opp_free_throws_pct_PY2") * PY2_weight) + (pl.col("adjdef_opp_free_throws_pct_PY1") * PY1_weight),
        weighted_adjdef_rebounds_defensive = (pl.col("adjdef_rebounds_defensive") * CY_weight) + (pl.col("adjdef_rebounds_defensive_PY2") * PY2_weight) + (pl.col("adjdef_rebounds_defensive_PY1") * PY1_weight),
        weighted_adjdef_opp_turnovers_total = (pl.col("adjdef_opp_turnovers_total") * CY_weight) + (pl.col("adjdef_opp_turnovers_total_PY2") * PY2_weight) + (pl.col("adjdef_opp_turnovers_total_PY1") * PY1_weight)
    )
elif (today_dt.month == 11 and today_dt.day > 15) or (today_dt == 12):
    ##### pre-January VoA weighted variables #####
    ### PY1 included still
    VoAStats = VoAStats.with_columns(
        ### offensive stats
        weighted_adjoff_ppg = (pl.col("adjoff_ppg") * CY_weight) + (pl.col("adjoff_ppg_PY1") * PY1_weight),
        weighted_adjoff_points_per_poss = (pl.col("adjoff_points_per_poss") * CY_weight) + (pl.col("adjoff_points_per_poss_PY1") * PY1_weight),
        weighted_adjoff_assists = (pl.col("adjoff_assists") * CY_weight) + (pl.col("adjoff_assists_PY1") * PY1_weight),
        weighted_adjoff_true_shooting = (pl.col("adjoff_true_shooting") * CY_weight) + (pl.col("adjoff_points_per_poss_PY1") * PY1_weight),
        weighted_adjoff_opp_blocks = (pl.col("adjoff_opp_blocks") * CY_weight) + (pl.col("adjoff_opp_blocks_PY1") * PY1_weight),
        weighted_adjoff_opp_steals = (pl.col("adjoff_opp_steals") * CY_weight) + (pl.col("adjoff_opp_steals_PY1") * PY1_weight),
        weighted_adjoff_field_goals_pct = (pl.col("adjoff_field_goals_pct") * CY_weight) + (pl.col("adjoff_field_goals_pct_PY1") * PY1_weight),
        weighted_adjoff_two_point_field_goals_pct = (pl.col("adjoff_two_point_field_goals_pct") * CY_weight) + (pl.col("adjoff_two_point_field_goals_pct_PY1") * PY1_weight),
        weighted_adjoff_three_point_field_goals_pct = (pl.col("adjoff_three_point_field_goals_pct") * CY_weight) + (pl.col("adjoff_three_point_field_goals_pct_PY1") * PY1_weight),
        weighted_adjoff_free_throws_pct = (pl.col("adjoff_free_throws_pct") * CY_weight) + (pl.col("adjoff_free_throws_pct_PY1") * PY1_weight),
        weighted_adjoff_rebounds_offensive = (pl.col("adjoff_rebounds_offensive") * CY_weight) + (pl.col("adjoff_rebounds_offensive_PY1") * PY1_weight),
        weighted_adjoff_turnovers_total = (pl.col("adjoff_turnovers_total") * CY_weight) + (pl.col("adjoff_turnovers_total_PY1") * PY1_weight),
        ### defensive stats
        weighted_adjdef_ppg = (pl.col("adjdef_ppg") * CY_weight) + (pl.col("adjdef_ppg_PY1") * PY1_weight),
        weighted_adjdef_opp_points_per_poss = (pl.col("adjdef_opp_points_per_poss") * CY_weight) + (pl.col("adjdef_opp_points_per_poss_PY1") * PY1_weight),
        weighted_adjdef_opp_assists = (pl.col("adjdef_opp_assists") * CY_weight) + (pl.col("adjdef_opp_assists_PY1") * PY1_weight),
        weighted_adjdef_opp_true_shooting = (pl.col("adjdef_opp_true_shooting") * CY_weight) + (pl.col("adjdef_opp_points_per_poss_PY1") * PY1_weight),
        weighted_adjdef_blocks = (pl.col("adjdef_blocks") * CY_weight) + (pl.col("adjdef_blocks_PY1") * PY1_weight),
        weighted_adjdef_steals = (pl.col("adjdef_steals") * CY_weight) + (pl.col("adjdef_steals_PY1") * PY1_weight),
        weighted_adjdef_opp_field_goals_pct = (pl.col("adjdef_opp_field_goals_pct") * CY_weight) + (pl.col("adjdef_opp_field_goals_pct_PY1") * PY1_weight),
        weighted_adjdef_opp_two_point_field_goals_pct = (pl.col("adjdef_opp_two_point_field_goals_pct") * CY_weight) + (pl.col("adjdef_opp_two_point_field_goals_pct_PY1") * PY1_weight),
        weighted_adjdef_opp_three_point_field_goals_pct = (pl.col("adjdef_opp_three_point_field_goals_pct") * CY_weight) + (pl.col("adjdef_opp_three_point_field_goals_pct_PY1") * PY1_weight),
        weighted_adjdef_opp_free_throws_pct = (pl.col("adjdef_opp_free_throws_pct") * CY_weight) + (pl.col("adjdef_opp_free_throws_pct_PY1") * PY1_weight),
        weighted_adjdef_rebounds_defensive = (pl.col("adjdef_rebounds_defensive") * CY_weight) + (pl.col("adjdef_rebounds_defensive_PY1") * PY1_weight),
        weighted_adjdef_opp_turnovers_total = (pl.col("adjdef_opp_turnovers_total") * CY_weight) + (pl.col("adjdef_opp_turnovers_total_PY1") * PY1_weight)
    )
else:
    print("no weighting necessary, only current season data being used")



##### Fitting model to create opponent-adjusted team strength ratings for offense and defense #####
### Offensive Model
## offensive stats that received opponent adjustments: ["assists", "true_shooting", "opp_blocks", "opp_steals", "field_goals_pct", "two_point_field_goals_pct", "three_point_field_goals_pct", "free_throws_pct", "rebounds_offensive", "turnovers_total", "points_fast_break", "points_off_turnovers", "points_in_paint", "points_per_poss"]
modelstarttime = datetime.now()
# with pm.Model() as offensive_model:
#     ### Priors for unknown model parameters
#     ### using normal priors because I don't know how to use something else (more accurately, I don't know how to fix something else if it goes wrong or doesn't work)
#     ## b0 is the intercept
#     b0 = pm.Normal("b0", mu = 50, sigma = 5)
#     beta_off_ppp = pm.Normal("beta_off_adjoffppp", mu = 2, sigma = 10)
#     beta_off_assists = pm.Normal("beta_off_assists", mu = 1, sigma = 10)
#     beta_off_trueshooting = pm.Normal("beta_off_trueshooting", mu = 2.5, sigma = 10)
#     beta_off_oppblocks = pm.Normal("beta_off_oppblocks", mu = -2, sigma = 5)
#     beta_off_oppsteals = pm.Normal("beta_off_oppsteals", mu = -2, sigma = 5)
#     beta_off_fgpct = pm.Normal("beta_off_fgpct", mu = 2, sigma = 20)
#     beta_off_2ptfgpct = pm.Normal("beta_off_twoptfgpct", mu = 1, sigma = 15)
#     beta_off_3ptfgpct = pm.Normal("beta_off_threeptfgpct", mu = 1, sigma = 15)
#     beta_off_ftpct = pm.Normal("beta_off_ftpct", mu = 1, sigma = 15)
#     beta_off_rebounds = pm.Normal("beta_off_rebounds", mu = 2, sigma = 15)
#     beta_off_turnovers = pm.Normal("beta_off_turnovers", mu = -1, sigma = 5)
#     ### variance prior
#     sigma = pm.HalfNormal("sigma", sigma=10)

#     # Expected value of outcome (The Linear Predictor)
#     mu = (
#         b0 +
#         beta_off_ppp * VoAStats['adjoff_points_per_poss'].to_numpy() +
#         beta_off_assists * VoAStats['adjoff_assists'].to_numpy() +
#         beta_off_trueshooting * VoAStats['adjoff_true_shooting'].to_numpy() +
#         beta_off_oppblocks * VoAStats['adjoff_opp_blocks'].to_numpy() +
#         beta_off_oppsteals * VoAStats['adjoff_opp_steals'].to_numpy() +
#         beta_off_fgpct * VoAStats['adjoff_field_goals_pct'].to_numpy() +
#         beta_off_2ptfgpct * VoAStats['adjoff_two_point_field_goals_pct'].to_numpy() +
#         beta_off_3ptfgpct * VoAStats['adjoff_three_point_field_goals_pct'].to_numpy() +
#         beta_off_ftpct * VoAStats['adjoff_free_throws_pct'].to_numpy() +
#         beta_off_rebounds * VoAStats['adjoff_rebounds_offensive'].to_numpy() +
#         beta_off_turnovers * VoAStats['adjoff_turnovers_total'].to_numpy()
#     )

#     ### Likelihood (Sampling distribution of the data)
#     Y_obs = pm.Normal(
#         "Y_obs", 
#         mu = mu, 
#         sigma = sigma, 
#         observed = VoAStats["adjoff_ppg"].to_numpy()
#     )

#     ### Fit the Model (MCMC Sampling)
#     idata = pm.sample(
#         draws = 7500, 
#         tune = 2500, 
#         chains = 3, 
#         random_seed = 802,
#         cores = os.cpu_count() // 2
#     )

#     ### Generate Posterior Predictive Samples
#     pm.sample_posterior_predictive(idata, extend_inferencedata = True)

# ### Extract posterior predictions using ArviZ
# posterior_preds = idata.posterior_predictive["Y_obs"].stack(sample=("chain", "draw")).values

###  Add mean, median, percentiles back to Polars as new columns
# VoAStats = VoAStats.with_columns(
#     OffVoA_MeanRating = posterior_preds.mean(axis=1),
#     OffVoA_MedRating = np.median(posterior_preds, axis=1),
#     OffVoA_975PctRating = np.percentile(posterior_preds, 97.5, axis=1),
#     OffVoA_025PctRating = np.percentile(posterior_preds, 2.5, axis=1),
#     OffVoA_SD = np.std(posterior_preds, axis = 1))

### what if stan, instead of pymc
if int(voa_num) == 1:
    ### Prepare Data Dictionary
    off_stan_datalist = {
        "N": VoATrain.height,
        "adjoff_ppg": VoATrain["adjoff_ppg"].to_numpy(),
        "ppp": VoATrain["adjoff_points_per_poss"].to_numpy(),
        "assists": VoATrain["adjoff_assists"].to_numpy(),
        "trueshooting": VoATrain["adjoff_true_shooting"].to_numpy(),
        "oppblocks": VoATrain["adjoff_opp_blocks"].to_numpy(),
        "oppsteals": VoATrain["adjoff_opp_steals"].to_numpy(),
        "fgpct": VoATrain["adjoff_field_goals_pct"].to_numpy(),
        "twoptfgpct": VoATrain["adjoff_two_point_field_goals_pct"].to_numpy(),
        "threeptfgpct": VoATrain["adjoff_three_point_field_goals_pct"].to_numpy(),
        "ftpct": VoATrain["adjoff_free_throws_pct"].to_numpy(),
        "rebounds": VoATrain["adjoff_rebounds_offensive"].to_numpy(),
        "turnovers": VoATrain["adjoff_turnovers_total"].to_numpy()
    }

    # cores = max(1, multiprocessing.cpu_count() // 2)

    ### Compile and Sample
    OffStan_file = os.path.join(os.getcwd(), "Scripts", "Stan", "MCBB_OffVoA.stan")
    OffStan_model = cmdstanpy.CmdStanModel(stan_file = OffStan_file)
    OffStan_fit = OffStan_model.sample(
        data = off_stan_datalist,
        iter_sampling = 10000,
        iter_warmup = 5000,
        chains = 3,
        seed = 802,
        parallel_chains = os.cpu_count() // 2
    )

    ### summarizing and diagnosing model
    print(OffStan_fit.summary())
    print(OffStan_fit.diagnose())

    ### Extract Posterior Draws
    OffVoA_pars = pl.from_pandas(OffStan_fit.draws_pd(vars=[
        "b0", "beta_off_ppp", "beta_off_assists", "beta_off_trueshooting", "beta_off_oppblocks", "beta_off_oppsteals", "beta_off_fgpct", "beta_off_2ptfgpct", "beta_off_3ptfgpct", "beta_off_ftpct", "beta_off_rebounds", "beta_off_turnovers", "sigma"
    ]))

    ### saving parameters as a parquet file so I don't have to re-fit the model each time I run the VoA
    OffVoA_pars.write_parquet(os.path.join(
        os.getcwd(),
        "Data",
        "FittedModels",
        "OffVoA_pars.parquet"
    ))

     ### using matrix math to turn posterior samples from Stan model into win probability projections for each game
    OffVoA_design_matrix = np.column_stack(
        [
            np.ones(VoAStats.height),
            VoAStats["weighted_adjoff_ppg"].to_numpy(),
            VoAStats["weighted_adjoff_points_per_poss"].to_numpy(),
            VoAStats["weighted_adjoff_assists"].to_numpy(),
            VoAStats["weighted_adjoff_true_shooting"].to_numpy(),
            VoAStats["weighted_adjoff_opp_blocks"].to_numpy(),
            VoAStats["weighted_adjoff_opp_steals"].to_numpy(),
            VoAStats["weighted_adjoff_field_goals_pct"].to_numpy(),
            VoAStats["weighted_adjoff_two_point_field_goals_pct"].to_numpy(),
            VoAStats["weighted_adjoff_three_point_field_goals_pct"].to_numpy(),
            VoAStats["weighted_adjoff_free_throws_pct"].to_numpy(),
            VoAStats["weighted_adjoff_rebounds_offensive"].to_numpy(),
            VoAStats["weighted_adjoff_turnovers_total"].to_numpy()
        ]
    )

    ### Extract Parameter Matrix: (N_draws x 2)
    OffVoA_pars_matrix = OffVoA_pars.select(["b0", "beta_off_ppp", "beta_off_assists", "beta_off_trueshooting", "beta_off_oppblocks", "beta_off_oppsteals", "beta_off_fgpct", "beta_off_2ptfgpct", "beta_off_3ptfgpct", "beta_off_ftpct", "beta_off_rebounds", "beta_off_turnovers", "sigma"]).to_numpy()

    ### Matrix Multiplication -> Linear Predictors (N_draws x N_games)
    OffVoA_means_matrix = OffVoA_pars_matrix @ OffVoA_design_matrix.T

    ### applying sigma parameter to posterior samples of adjusted off ppg to generate Off VoA Ratings
    random.seed(802)
    sigma_noise = np.random.normal(loc=0.0, scale=OffVoA_pars["sigma"].to_numpy()[:, np.newaxis], size=(OffVoA_pars.height, VoAStats.height))
    OffVoA_Ratings = OffVoA_means_matrix + sigma_noise

    ### creating final ratings columns
    OffVoA_MeanRatings = np.mean(OffVoA_Ratings, axis = 0)
    OffVoA_MedianRatings = np.median(OffVoA_Ratings, axis = 0)
    OffVoA_upper = np.percentile(OffVoA_Ratings, 97.5, axis = 0)
    OffVoA_lower = np.percentile(OffVoA_Ratings, 2.5, axis = 0)


# ### Vectorized Posterior Predictive Calculation
# predictors = np.vstack([
#     np.ones(VoAStats.height), # Intercept
#     VoAStats["adjoff_points_per_poss"].to_numpy(),
#     VoAStats["adjoff_assists"].to_numpy(),
#     VoAStats["adjoff_true_shooting"].to_numpy(),
#     VoAStats["adjoff_opp_blocks"].to_numpy(),
#     VoAStats["adjoff_opp_steals"].to_numpy(),
#     VoAStats["adjoff_field_goals_pct"].to_numpy(),
#     VoAStats["adjoff_two_point_field_goals_pct"].to_numpy(),
#     VoAStats["adjoff_three_point_field_goals_pct"].to_numpy(),
#     VoAStats["adjoff_free_throws_pct"].to_numpy(),
#     VoAStats["adjoff_rebounds_offensive"].to_numpy(),
#     VoAStats["adjoff_turnovers_total"].to_numpy()
# ])

# # Matrix of coefficients: (Samples x Features)
# betas = pars.drop(columns=["sigma"]).values 

# # Calculate mu (Samples x Teams)
# mu = np.dot(betas, predictors)

# # Add process uncertainty (sigma)
# rng = np.random.default_rng(802)
# sigma_vec = pars["sigma"].values[:, np.newaxis]
# posterior_preds = mu + rng.normal(0, 1, size=mu.shape) * sigma_vec

# # 5. Summary Statistics back to Polars
# VoAStatsStan = VoAStats.with_columns([
#     pl.Series("OffVoA_MeanRating_stan", posterior_preds.mean(axis=0)),
#     pl.Series("OffVoA_MedRating_stan", np.median(posterior_preds, axis=0)),
#     pl.Series("OffVoA_95PctRating_stan", np.percentile(posterior_preds, 97.5, axis=0)),
#     pl.Series("OffVoA_05PctRating_stan", np.percentile(posterior_preds, 2.5, axis=0)),
#     pl.Series("OffVoA_SD_stan", np.std(posterior_preds, axis = 0))
# ])

# # Sort by the fastest teams in the country
# teams_master_df = teams_master_df.sort("adj_pace", descending=True)


with pm.Model() as defensive_model:
    ### Priors for unknown model parameters
    ### using normal priors because I don't know how to use something else (more accurately, I don't know how to fix something else if it goes wrong or doesn't work)
    ## b0 is the intercept
    b0 = pm.Normal("b0", mu = 50, sigma = 5)
    beta_def_oppppp = pm.Normal("beta_def_adjdefppp", mu = 2, sigma = 10)
    beta_def_oppassists = pm.Normal("beta_def_assists", mu = 1, sigma = 10)
    beta_def_opptrueshooting = pm.Normal("beta_def_trueshooting", mu = 2.5, sigma = 10)
    beta_def_blocks = pm.Normal("beta_def_oppblocks", mu = -2, sigma = 5)
    beta_def_steals = pm.Normal("beta_def_oppsteals", mu = -2, sigma = 5)
    beta_def_oppfgpct = pm.Normal("beta_def_fgpct", mu = 2, sigma = 20)
    beta_def_opp2ptfgpct = pm.Normal("beta_def_twoptfgpct", mu = 1, sigma = 15)
    beta_def_opp3ptfgpct = pm.Normal("beta_def_threeptfgpct", mu = 1, sigma = 15)
    beta_def_oppftpct = pm.Normal("beta_def_ftpct", mu = 1, sigma = 15)
    beta_def_rebounds = pm.Normal("beta_def_rebounds", mu = 2, sigma = 15)
    beta_def_oppturnovers = pm.Normal("beta_def_turnovers", mu = -1, sigma = 5)
    ### variance prior
    sigma = pm.HalfNormal("sigma", sigma=10)

    # Expected value of outcome (The Linear Predictor)
    mu = (
        b0 +
        beta_def_oppppp * VoAStats['adjdef_opp_points_per_poss'].to_numpy() +
        beta_def_oppassists * VoAStats['adjdef_opp_assists'].to_numpy() +
        beta_def_opptrueshooting * VoAStats['adjdef_opp_true_shooting'].to_numpy() +
        beta_def_blocks * VoAStats['adjdef_blocks'].to_numpy() +
        beta_def_steals * VoAStats['adjdef_steals'].to_numpy() +
        beta_def_oppfgpct * VoAStats['adjdef_opp_field_goals_pct'].to_numpy() +
        beta_def_opp2ptfgpct * VoAStats['adjdef_opp_two_point_field_goals_pct'].to_numpy() +
        beta_def_opp3ptfgpct * VoAStats['adjdef_opp_three_point_field_goals_pct'].to_numpy() +
        beta_def_oppftpct * VoAStats['adjdef_opp_free_throws_pct'].to_numpy() +
        beta_def_rebounds * VoAStats['adjdef_rebounds_defensive'].to_numpy() +
        beta_def_oppturnovers * VoAStats['adjdef_opp_turnovers_total'].to_numpy()
    )

    ### Likelihood (Sampling distribution of the data)
    Y_obs = pm.Normal(
        "Y_obs", 
        mu = mu, 
        sigma = sigma, 
        observed = VoAStats["adjdef_ppg"].to_numpy()
    )

    ### Fit the Model (MCMC Sampling)
    idata = pm.sample(
        draws = 7500, 
        tune = 2500, 
        chains = 3, 
        random_seed = 802,
        cores = os.cpu_count() // 2
    )

    ### Generate Posterior Predictive Samples
    pm.sample_posterior_predictive(idata, extend_inferencedata = True)

### Extract posterior predictions using ArviZ
posterior_preds = idata.posterior_predictive["Y_obs"].stack(sample=("chain", "draw")).values

###  Add mean, median, percentiles back to Polars as new columns
VoAStats = VoAStats.with_columns(
    DefVoA_MeanRating = posterior_preds.mean(axis=1),
    DefVoA_MedRating = np.median(posterior_preds, axis=1),
    DefVoA_975PctRating = np.percentile(posterior_preds, 97.5, axis=1),
    DefVoA_025PctRating = np.percentile(posterior_preds, 2.5, axis=1),
    DefVoA_SD = np.std(posterior_preds, axis = 1))

### Calculating Overall Rating by subtracting defensive rating from overall rating
## accidentally mislabled
VoAStats = VoAStats.with_columns(
    OvrlVoA_MeanRating = pl.col('OffVoA_MeanRating') - pl.col('DefVoA_MeanRating'),
    OvrlVoA_975pctRating = pl.col('OffVoA_975PctRating') - pl.col('DefVoA_975PctRating'),
    OvrlVoA_025pctRating = pl.col('OffVoA_025PctRating') - pl.col('DefVoA_025PctRating')
)

### Creating Ranking Columns for the overall, offensive, and defensive mean ratings
VoAStats = VoAStats.with_columns(
    OvrlVoARanking = pl.col('OvrlVoA_MeanRating').rank(method = "dense", descending = True),
    OffVoARanking = pl.col('OffVoA_MeanRating').rank(method = "dense", descending = True),
    DefVoARanking = pl.col('DefVoA_MeanRating').rank(method = "dense", descending = False),
)

modelendtime = datetime.now()


##### Saving VoAStats and creating output tables #####
### Saving final VoA csv
VoAStats.write_csv(os.path.join(os.getcwd(), "Data", "VoA" + str(cbb_season), "MCBBVoA" + str(cbb_season) + "VoA" + voa_num + ".csv"))

### preparing VoAStats for making the gt table
VoATable_df = VoAStats.select(['team', 'OvrlVoA_MeanRating', 'OvrlVoARanking', 'OffVoA_MeanRating', 'OffVoARanking', 'DefVoA_MeanRating', 'DefVoARanking']).sort('OvrlVoARanking', descending = False)

VoATable_Top25df = VoATable_df.filter(pl.col('OvrlVoARanking') <= 25)

### creating table using gt
Top25_gt = (
    GT(VoATable_Top25df)
    .tab_header(
        title = cbb_season_str + " MCBB D1 Vortex of Accuracy Top 25 Ratings",
        subtitle = "Supremely Excellent Yet Salaciously Godlike And Infallibly Magnificent Vortex of Accuracy, " + str(today_dt.date())
    )
    # Formatting numbers (grouped by decimal count for efficiency)
    .fmt_number(
        columns = ["OvrlVoA_MeanRating", "OffVoA_MeanRating", "DefVoA_MeanRating"],
        decimals = 3
    )
    .fmt_number(
        columns=["OvrlVoARanking"],
        decimals=0
    )
    .data_color(
        columns = ["OvrlVoA_MeanRating", "OffVoA_MeanRating"],
        palette = "RdYlGn",
        na_color = "white",
        ### this autocolor line was added because there's apparently a bug in great-tables?
        ## stack overflow also suggested upgrading but I did that and nothing happened (I went from 0.18.0 to 0.21.0, same key error occured "'font_color_row_striping_background_color'")
        # autocolor_text = False
    )
    .data_color(
        columns = ["DefVoA_MeanRating"],
        palette = "RdYlGn",
        reverse = True,
        na_color = "white",
        ### turning off autocolor_text for same reason described above
        # autocolor_text = False
    )
    # Column Labels
    .cols_label(
        OvrlVoA_MeanRating = "Overall VoA Rating",
        OvrlVoARanking = "VoA Ranking",
        OffVoA_MeanRating = "Off VoA Rating",
        OffVoARanking = "Off Ranking",
        DefVoA_MeanRating = "Def VoA Rating",
        DefVoARanking = "Def Ranking"
    )
    # Hide columns
    # .cols_hide(columns=["week", "VoA_Output"])
    # Add Footnote
    .tab_source_note(
        source_note = "Table by @gshelor, Data from CBBD API"
    )
)

### saving Top 25 table
Top25_gt.save(file = os.path.join(os.getcwd(), "Outputs", "VoA" + str(cbb_season), "MCBBVoA" + str(cbb_season) + "VoATop25Table.png"))

### creating Full table using gt
FullTable_gt = (
    GT(VoATable_df)
    .tab_header(
        title = cbb_season_str + " MCBB D1 Vortex of Accuracy Ratings",
        subtitle = "Supremely Excellent Yet Salaciously Godlike And Infallibly Magnificent Vortex of Accuracy, " + str(today_dt.date())
    )
    # Formatting numbers (grouped by decimal count for efficiency)
    .fmt_number(
        columns = ["OvrlVoA_MeanRating", "OffVoA_MeanRating", "DefVoA_MeanRating"],
        decimals = 3
    )
    .fmt_number(
        columns=["OvrlVoARanking"],
        decimals=0
    )
    .data_color(
        columns = ["OvrlVoA_MeanRating", "OffVoA_MeanRating"],
        palette = "RdYlGn",
        na_color = "white",
        ### this autocolor line was added because there's apparently a bug in great-tables?
        ## stack overflow also suggested upgrading but I did that and nothing happened (I went from 0.18.0 to 0.21.0, same key error occured "'font_color_row_striping_background_color'")
        # autocolor_text = False
    )
    .data_color(
        columns = ["DefVoA_MeanRating"],
        palette = "RdYlGn",
        reverse = True,
        na_color = "white",
        ### turning off autocolor_text for same reason described above
        # autocolor_text = False
    )
    # Column Labels
    .cols_label(
        OvrlVoA_MeanRating = "Overall VoA Rating",
        OvrlVoARanking = "VoA Ranking",
        OffVoA_MeanRating = "Off VoA Rating",
        OffVoARanking = "Off Ranking",
        DefVoA_MeanRating = "Def VoA Rating",
        DefVoARanking = "Def Ranking"
    )
    # Hide columns
    # .cols_hide(columns=["week", "VoA_Output"])
    # Add Footnote
    .tab_source_note(
        source_note = "Table by @gshelor, Data from CBBD API"
    )
)

FullTable_gt.show()
### Display and Save
FullTable_gt.save(file = os.path.join(os.getcwd(), "Outputs", "VoA" + str(cbb_season), "MCBBVoA" + str(cbb_season) + "VoAFullTable.png"))

##### POOPYPANTS TESTING HERE #####
# VoAStats.write_csv(os.path.join(os.getcwd(), "poopypants.csv"))

# poopypants = VoAStats.select(['team', 'OvrlVoA_MeanRating', 'OffVoA_MeanRating', 'DefVoA_MeanRating']).sort('OvrlVoA_MeanRating', descending = True)

# poopypants.tail(10)
# poopypants.head(10)

# poopypants = VoAStats.with_columns(
#     pts_per_100_poss = pl.col('adjoff_points_per_poss') * 100,
#     opppts_per_100_poss = pl.col('adjdef_opp_points_per_poss') * 100
# ).with_columns(
#     adj_ppg_diff = pl.col('adjoff_ppg') - pl.col('adjdef_ppg'),
#     pts_per_100_poss_diff = pl.col('pts_per_100_poss') - pl.col('opppts_per_100_poss')
# ).select(['team', 'OvrlVoARanking', 'OvrlVoA_MeanRating', 'adj_ppg_diff', 'pts_per_100_poss_diff', 'OffVoA_MeanRating', 'pts_per_100_poss', 'adjoff_ppg', 'DefVoA_MeanRating', 'opppts_per_100_poss', 'adjdef_ppg'])

print(modelendtime - modelstarttime)
