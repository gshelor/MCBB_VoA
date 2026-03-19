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
from great_tables import GT
import selenium
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
    ### polars gives a warning that the is_in() (in the filter) part of the code below is deprecated (as of March 2, 2026 I'm on polars version 1.37.1) but it works and does what I want it to and everybody in the github issue it links to (https://github.com/pola-rs/polars/issues/22149) is mad that it would be producing an issue at all, given that the below usage of is_in() is what they all think would be the most common usage would be (and I agree) so hopefully it stays intact
    AllGamesStatAdj = pl.concat(
        [AllGamesStatAdj, CleanGames_df], how = "vertical_relaxed").filter(
            (pl.col("team").is_in(VoAVariables['team']) & pl.col("opponent").is_in(VoAVariables['team']))
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
    

### Once data is loaded, I should have: VoAVariables, which is the aggregated season stats, and AllGames, which is the combined dataframe of all games with their various stats columns unnested

### now onto using mixed-effects models to get opponent-adjusted stats
##### Opponent Adjusted Stats using game-level data #####
if today_dt.month == 10:
    print("put preseason stuff here")
elif today_dt.month == 11:
    print("might do something here too")
else:
    ### calling function to calculate opponent-adjusted stats and return them all in one polars dataframe
    ## this function is also in the VoAFuncs.py script
    # function call: opponent_adjustments(pace_df, full_df)
    AdjStats = opponent_adjustments(AllGamesPaceAdj, AllGamesStatAdj)
    VoAVariables = VoAVariables.join(AdjStats, on = "team").with_columns(
        adjoff_ppg = pl.col("adjoff_points_per_poss") * pl.col("adj_pace"),
        adjdef_ppg = pl.col("adjdef_opp_points_per_poss") * pl.col("adj_pace")
    )


##### Fitting model to create opponent-adjusted team strength ratings for offense and defense #####
### Offensive Model
## offensive stats that received opponent adjustments: ["assists", "true_shooting", "opp_blocks", "opp_steals", "field_goals_pct", "two_point_field_goals_pct", "three_point_field_goals_pct", "free_throws_pct", "rebounds_offensive", "turnovers_total", "points_fast_break", "points_off_turnovers", "points_in_paint", "points_per_poss"]
modelstarttime = datetime.now()
with pm.Model() as offensive_model:
    ### Priors for unknown model parameters
    ### using normal priors because I don't know how to use something else (more accurately, I don't know how to fix something else if it goes wrong or doesn't work)
    ## b0 is the intercept
    b0 = pm.Normal("b0", mu = 50, sigma = 5)
    beta_off_ppp = pm.Normal("beta_off_adjoffppp", mu = 2, sigma = 10)
    beta_off_assists = pm.Normal("beta_off_assists", mu = 1, sigma = 10)
    beta_off_trueshooting = pm.Normal("beta_off_trueshooting", mu = 2.5, sigma = 10)
    beta_off_oppblocks = pm.Normal("beta_off_oppblocks", mu = -2, sigma = 5)
    beta_off_oppsteals = pm.Normal("beta_off_oppsteals", mu = -2, sigma = 5)
    beta_off_fgpct = pm.Normal("beta_off_fgpct", mu = 2, sigma = 20)
    beta_off_2ptfgpct = pm.Normal("beta_off_twoptfgpct", mu = 1, sigma = 15)
    beta_off_3ptfgpct = pm.Normal("beta_off_threeptfgpct", mu = 1, sigma = 15)
    beta_off_ftpct = pm.Normal("beta_off_ftpct", mu = 1, sigma = 15)
    beta_off_rebounds = pm.Normal("beta_off_rebounds", mu = 2, sigma = 15)
    beta_off_turnovers = pm.Normal("beta_off_turnovers", mu = -1, sigma = 5)
    ### variance prior
    sigma = pm.HalfNormal("sigma", sigma=10)

    # Expected value of outcome (The Linear Predictor)
    mu = (
        b0 +
        beta_off_ppp * VoAVariables['adjoff_points_per_poss'].to_numpy() +
        beta_off_assists * VoAVariables['adjoff_assists'].to_numpy() +
        beta_off_trueshooting * VoAVariables['adjoff_true_shooting'].to_numpy() +
        beta_off_oppblocks * VoAVariables['adjoff_opp_blocks'].to_numpy() +
        beta_off_oppsteals * VoAVariables['adjoff_opp_steals'].to_numpy() +
        beta_off_fgpct * VoAVariables['adjoff_field_goals_pct'].to_numpy() +
        beta_off_2ptfgpct * VoAVariables['adjoff_two_point_field_goals_pct'].to_numpy() +
        beta_off_3ptfgpct * VoAVariables['adjoff_three_point_field_goals_pct'].to_numpy() +
        beta_off_ftpct * VoAVariables['adjoff_free_throws_pct'].to_numpy() +
        beta_off_rebounds * VoAVariables['adjoff_rebounds_offensive'].to_numpy() +
        beta_off_turnovers * VoAVariables['adjoff_turnovers_total'].to_numpy()
    )

    ### Likelihood (Sampling distribution of the data)
    Y_obs = pm.Normal(
        "Y_obs", 
        mu = mu, 
        sigma = sigma, 
        observed = VoAVariables["adjoff_ppg"].to_numpy()
    )

    ### Fit the Model (MCMC Sampling)
    idata = pm.sample(
        draws = 10000, 
        tune = 3000, 
        chains = 3, 
        random_seed = 802,
        cores = os.cpu_count() // 2
    )

    ### Generate Posterior Predictive Samples
    pm.sample_posterior_predictive(idata, extend_inferencedata = True)

### Extract posterior predictions using ArviZ
posterior_preds = idata.posterior_predictive["Y_obs"].stack(sample=("chain", "draw")).values

###  Add mean, median, percentiles back to Polars as new columns
VoAVariables = VoAVariables.with_columns(
    OffVoA_MeanRating = posterior_preds.mean(axis=1),
    OffVoA_MedRating = np.median(posterior_preds, axis=1),
    OffVoA_975PctRating = np.percentile(posterior_preds, 97.5, axis=1),
    OffVoA_025PctRating = np.percentile(posterior_preds, 2.5, axis=1),
    OffVoA_SD = np.std(posterior_preds, axis = 1))

### what if stan, instead of pymc
### Prepare Data Dictionary
# stan_data = {
#     "N": VoAVariables.height,
#     "adjoff_ppg": VoAVariables["adjoff_ppg"].to_numpy(),
#     "ppp": VoAVariables["adjoff_points_per_poss"].to_numpy(),
#     "assists": VoAVariables["adjoff_assists"].to_numpy(),
#     "trueshooting": VoAVariables["adjoff_true_shooting"].to_numpy(),
#     "oppblocks": VoAVariables["adjoff_opp_blocks"].to_numpy(),
#     "oppsteals": VoAVariables["adjoff_opp_steals"].to_numpy(),
#     "fgpct": VoAVariables["adjoff_field_goals_pct"].to_numpy(),
#     "twoptfgpct": VoAVariables["adjoff_two_point_field_goals_pct"].to_numpy(),
#     "threeptfgpct": VoAVariables["adjoff_three_point_field_goals_pct"].to_numpy(),
#     "ftpct": VoAVariables["adjoff_free_throws_pct"].to_numpy(),
#     "rebounds": VoAVariables["adjoff_rebounds_offensive"].to_numpy(),
#     "turnovers": VoAVariables["adjoff_turnovers_total"].to_numpy(),
# }

# ### Compile and Sample
# stanfile = os.path.join(os.getcwd(), "Scripts", "Stan", "MCBB_OffVoA.stan")
# model = cmdstanpy.CmdStanModel(stan_file = stanfile)
# fit = model.sample(
#     data=stan_data,
#     iter_sampling=10000,
#     iter_warmup=3000,
#     chains=3,
#     seed=802,
#     parallel_chains = os.cpu_count() // 2
# )

# ### Extract Posterior Draws
# pars = fit.draws_pd(vars=[
#     "b0", "beta_off_ppp", "beta_off_assists", "beta_off_trueshooting",
#     "beta_off_oppblocks", "beta_off_oppsteals", "beta_off_fgpct",
#     "beta_off_2ptfgpct", "beta_off_3ptfgpct", "beta_off_ftpct",
#     "beta_off_rebounds", "beta_off_turnovers", "sigma"
# ])

# ### Vectorized Posterior Predictive Calculation
# predictors = np.vstack([
#     np.ones(VoAVariables.height), # Intercept
#     VoAVariables["adjoff_points_per_poss"].to_numpy(),
#     VoAVariables["adjoff_assists"].to_numpy(),
#     VoAVariables["adjoff_true_shooting"].to_numpy(),
#     VoAVariables["adjoff_opp_blocks"].to_numpy(),
#     VoAVariables["adjoff_opp_steals"].to_numpy(),
#     VoAVariables["adjoff_field_goals_pct"].to_numpy(),
#     VoAVariables["adjoff_two_point_field_goals_pct"].to_numpy(),
#     VoAVariables["adjoff_three_point_field_goals_pct"].to_numpy(),
#     VoAVariables["adjoff_free_throws_pct"].to_numpy(),
#     VoAVariables["adjoff_rebounds_offensive"].to_numpy(),
#     VoAVariables["adjoff_turnovers_total"].to_numpy()
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
# VoAVariablesStan = VoAVariables.with_columns([
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
        beta_def_oppppp * VoAVariables['adjdef_opp_points_per_poss'].to_numpy() +
        beta_def_oppassists * VoAVariables['adjdef_opp_assists'].to_numpy() +
        beta_def_opptrueshooting * VoAVariables['adjdef_opp_true_shooting'].to_numpy() +
        beta_def_blocks * VoAVariables['adjdef_blocks'].to_numpy() +
        beta_def_steals * VoAVariables['adjdef_steals'].to_numpy() +
        beta_def_oppfgpct * VoAVariables['adjdef_opp_field_goals_pct'].to_numpy() +
        beta_def_opp2ptfgpct * VoAVariables['adjdef_opp_two_point_field_goals_pct'].to_numpy() +
        beta_def_opp3ptfgpct * VoAVariables['adjdef_opp_three_point_field_goals_pct'].to_numpy() +
        beta_def_oppftpct * VoAVariables['adjdef_opp_free_throws_pct'].to_numpy() +
        beta_def_rebounds * VoAVariables['adjdef_rebounds_defensive'].to_numpy() +
        beta_def_oppturnovers * VoAVariables['adjdef_opp_turnovers_total'].to_numpy()
    )

    ### Likelihood (Sampling distribution of the data)
    Y_obs = pm.Normal(
        "Y_obs", 
        mu = mu, 
        sigma = sigma, 
        observed = VoAVariables["adjdef_ppg"].to_numpy()
    )

    ### Fit the Model (MCMC Sampling)
    idata = pm.sample(
        draws = 10000, 
        tune = 3000, 
        chains = 3, 
        random_seed = 802,
        cores = os.cpu_count() // 2
    )

    ### Generate Posterior Predictive Samples
    pm.sample_posterior_predictive(idata, extend_inferencedata = True)

### Extract posterior predictions using ArviZ
posterior_preds = idata.posterior_predictive["Y_obs"].stack(sample=("chain", "draw")).values

###  Add mean, median, percentiles back to Polars as new columns
VoAVariables = VoAVariables.with_columns(
    DefVoA_MeanRating = posterior_preds.mean(axis=1),
    DefVoA_MedRating = np.median(posterior_preds, axis=1),
    DefVoA_975PctRating = np.percentile(posterior_preds, 97.5, axis=1),
    DefVoA_025PctRating = np.percentile(posterior_preds, 2.5, axis=1),
    DefVoA_SD = np.std(posterior_preds, axis = 1))

### Calculating Overall Rating by subtracting defensive rating from overall rating
## accidentally mislabled
VoAVariables = VoAVariables.with_columns(
    OvrlVoA_MeanRating = pl.col('OffVoA_MeanRating') - pl.col('DefVoA_MeanRating'),
    OvrlVoA_975pctRating = pl.col('OffVoA_975PctRating') - pl.col('DefVoA_975PctRating'),
    OvrlVoA_025pctRating = pl.col('OffVoA_025PctRating') - pl.col('DefVoA_025PctRating')
)

### Creating Ranking Columns for the overall, offensive, and defensive mean ratings
VoAVariables = VoAVariables.with_columns(
    OvrlVoARanking = pl.col('OvrlVoA_MeanRating').rank(method = "dense", descending = True),
    OffVoARanking = pl.col('OffVoA_MeanRating').rank(method = "dense", descending = True),
    DefVoARanking = pl.col('DefVoA_MeanRating').rank(method = "dense", descending = False),
)

modelendtime = datetime.now()


##### Saving VoAVariables and creating output tables #####
### Saving final VoA csv
VoAVariables.write_csv(os.path.join(os.getcwd(), "Data", "VoA" + str(cbb_season), "MCBBVoA" + str(cbb_season) + "VoA" + voa_num + ".csv"))

### preparing VoAVariables for making the gt table
VoATable_df = VoAVariables.select(['team', 'OvrlVoA_MeanRating', 'OvrlVoARanking', 'OffVoA_MeanRating', 'OffVoARanking', 'DefVoA_MeanRating', 'DefVoARanking']).sort('OvrlVoARanking', descending = False)

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
Top25_gt.save(file = os.path.join(os.getcwd(), "Outputs", "VoA" + str(cbb_season), "MCBBVoA" + str(cbb_season) + "VoA" + voa_num + "Top25Table.png"))

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
FullTable_gt.save(file = os.path.join(os.getcwd(), "Outputs", "VoA" + str(cbb_season), "MCBBVoA" + str(cbb_season) + "VoA" + voa_num + "FullTable.png"))

##### POOPYPANTS TESTING HERE #####
# VoAVariables.write_csv(os.path.join(os.getcwd(), "poopypants.csv"))

# poopypants = VoAVariables.select(['team', 'OvrlVoA_MeanRating', 'OffVoA_MeanRating', 'DefVoA_MeanRating']).sort('OvrlVoA_MeanRating', descending = True)

# poopypants.tail(10)
# poopypants.head(10)

# poopypants = VoAVariables.with_columns(
#     pts_per_100_poss = pl.col('adjoff_points_per_poss') * 100,
#     opppts_per_100_poss = pl.col('adjdef_opp_points_per_poss') * 100
# ).with_columns(
#     adj_ppg_diff = pl.col('adjoff_ppg') - pl.col('adjdef_ppg'),
#     pts_per_100_poss_diff = pl.col('pts_per_100_poss') - pl.col('opppts_per_100_poss')
# ).select(['team', 'OvrlVoARanking', 'OvrlVoA_MeanRating', 'adj_ppg_diff', 'pts_per_100_poss_diff', 'OffVoA_MeanRating', 'pts_per_100_poss', 'adjoff_ppg', 'DefVoA_MeanRating', 'opppts_per_100_poss', 'adjdef_ppg'])

print(modelendtime - modelstarttime)
