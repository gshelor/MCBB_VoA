##### Men's D1 College Basketball Vortex of Projection Version 0.1 #####
### this script takes VoA ratings from MCBB_VoA.py and uses them to create projections for future games over a 1-2 week period
##### importing libraries and setting up python environment and API connection #####
import great_tables
import cbbd
import polars as pl
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from dotenv import load_dotenv
from datetime import date, datetime, timedelta
from great_tables import GT
# import great_tables
import cmdstanpy
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
### creating string of date combined together so I can identify unique projections compiled during the course of a season
if today_dt.month >= 10:
    datestring = str(today_dt.year) + str(today_dt.year + 1)
else:
    datestring = str(today_dt.year - 1 ) + str(today_dt.year) #+ str(today_dt.month) + str(today_dt.day)

### storing which iteration of the VoA this is for the season
voa_num = input("Which release of the VoA most recently happened for this season? ")
# vop_check = input("Is this the first time running the VoP script? (y/n) ")

### creating directory to store game projections if necessary
if today_dt.month == 10:
    os.makedirs(os.path.join(os.getcwd(), 'Data', 'VoA' + str(cbb_season), 'Projections', 'CBBD'), exist_ok = True)
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
    VoAStats = pl.read_parquet(os.path.join(os.getcwd(), "Data", "VoA" + str(cbb_season), "MCBBVoA" + str(cbb_season) + "VoA" + voa_num + ".parquet"))
    ### reading in games from at least the first month of the season
    UpcomingGames = gamesapi_instance.get_games(start_date_range = today_dt, end_date_range = today_dt + timedelta(days = 50))
    ### converting API output to df
    UpcomingGames_df = pl.DataFrame(UpcomingGames, infer_schema_length = None).filter(
        pl.col('status') != 'final'
    ).filter(
        (pl.col("home_team").is_in(VoAStats['team'].implode()) & pl.col("away_team").is_in(VoAStats['team'].implode()))
    )
    ### selecting ratings column and team so I can do a left join to attach appropriate ratings to matching home and away teams
    VoARatings = VoAStats.select(['team', 'OvrlVoA_MeanRating'])
    HomeVoARating = VoARatings.rename({'team': 'home_team', 'OvrlVoA_MeanRating': 'home_rating'})
    AwayVoARating = VoARatings.rename({'team': 'away_team', 'OvrlVoA_MeanRating': 'away_rating'})

    ### reading in old projections to use as inputs for a win probability model built with Stan and cmdstanpy
    PrevVoAPreds_PY1 = pl.read_csv(os.path.join(os.getcwd(), "Data", "VoA" + str(int(cbb_season) - 1), "AccuracyMetrics", "VoACompletedGames" + str(int(cbb_season) - 1) + "AccuracyMetrics.csv")).select(["proj_margin", "VoA_correct_winner"])
    # PrevVoAPreds_PY2 = pl.read_csv(os.path.join(os.getcwd(), "Data", "VoA" + str(int(cfb_year) - 2), "AccuracyMetrics", "Games", "VoA" + str(int(cfb_year) - 2) + "Week1Week17GameAccuracyMetrics.csv")).select(["proj_margin", "straight_up_win"])
    # PrevVoAPreds = pl.concat([PrevVoAPreds_PY1, PrevVoAPreds_PY2], how = "vertical")
    PrevVoAPreds = PrevVoAPreds_PY1
else:
    ### loading upcoming week of games
    VoAStats = pl.read_parquet(os.path.join(
        os.getcwd(), 
        "Data", 
        "VoA" + str(cbb_season), 
        "MCBBVoA" + str(cbb_season) + "VoA" + voa_num + ".parquet"
        ))
    UpcomingGames = gamesapi_instance.get_games(start_date_range = today_dt, end_date_range = today_dt + timedelta(days = 7))
    ### converting API output to df
    UpcomingGames_df = pl.DataFrame(UpcomingGames).filter(
        pl.col('status') != 'final'
    ).filter(
        (pl.col("home_team").is_in(VoAStats['team'].implode()) & pl.col("away_team").is_in(VoAStats['team'].implode()))
    )
    VoARatings = VoAStats.select(['team', 'OvrlVoA_MeanRating'])
    HomeVoARating = VoARatings.rename({'team': 'home_team', 'OvrlVoA_MeanRating': 'home_rating'})
    AwayVoARating = VoARatings.rename({'team': 'away_team', 'OvrlVoA_MeanRating': 'away_rating'})
    ### reading in csv of prior predictions to be bound to upcoming games df so I can save all predictions together in one csv
    PriorPreds = pl.read_csv(os.path.join(os.getcwd(), "Data", "VoA" + str(cbb_season), "Projections", "MCBBVoA" + datestring + "GameProjections.csv"), schema = {
        'id': pl.Int64,
        'source_id': pl.String,
        'season_label': pl.String,
        'season': pl.Int64,
        'season_type': pl.String,
        'start_date': pl.Datetime(time_unit='us', time_zone=None),
        'start_time_tbd': pl.Boolean,
        'neutral_site': pl.Boolean,
        'conference_game': pl.Boolean,
        'game_type': pl.String,
        'tournament': pl.String,
        'game_notes': pl.String,
        'status': pl.String,
        'home_team_id': pl.Int64,
        'home_team': pl.String,
        'home_conference_id': pl.Int64,
        'home_conference': pl.String,
        'home_seed': pl.Int64,
        'away_team_id': pl.Int64,
        'away_team': pl.String,
        'away_conference_id': pl.Int64,
        'away_conference': pl.String,
        'away_seed': pl.Int64,
        'venue_id': pl.Int64,
        'venue': pl.String,
        'city': pl.String,
        'state': pl.String,
        'home_rating': pl.Float64,
        'away_rating': pl.Float64,
        'proj_margin': pl.Float64,
        'proj_winner': pl.String})

    ### reading in old projections to use as inputs for a win probability model built with Stan and cmdstanpy
    ## PY2 commented out for now since I only first started doing a MCBB VoA in spring 2026, so there's not 2 seasons of projections yet
    PrevVoAPreds_PY1 = pl.read_csv(os.path.join(os.getcwd(), "Data", "VoA" + str(int(cbb_season) - 1), "AccuracyMetrics", "VoACompletedGames" + str(int(cbb_season) - 1) + "AccuracyMetrics.csv")).select(["proj_margin", "VoA_correct_winner"])
    # PrevVoAPreds_PY2 = pl.read_csv(os.path.join(os.getcwd(), "Data", "VoA" + str(int(cfb_year) - 2), "AccuracyMetrics", "Games", "VoA" + str(int(cfb_year) - 2) + "Week1Week17GameAccuracyMetrics.csv")).select(["proj_margin", "straight_up_win"])
    PrevVoAPreds = pl.read_csv(os.path.join(os.getcwd(), "Data", "VoA" + str(int(cbb_season)), "AccuracyMetrics", "VoACompletedGames" + str(int(cbb_season)) + "AccuracyMetrics.csv")).select(["proj_margin", "VoA_correct_winner"])
    PrevVoAPreds = pl.concat([PrevVoAPreds_PY1, PrevVoAPreds], how = "vertical")
    


##### Fitting a logistic regression model using old VoA projections #####
##### Fitting Stan Model to make win probability projections #####
### creating list of data to be fed into model
VoP_WP_datalist = {
    "N": PrevVoAPreds.height,
    "win_loss": PrevVoAPreds["VoA_correct_winner"].to_numpy(),
    "win_margin": PrevVoAPreds["proj_margin"].abs().to_numpy()
}

# cores = max(1, multiprocessing.cpu_count() // 2)

WP_VoP_model = cmdstanpy.CmdStanModel(
    stan_file= os.path.join(
        os.getcwd(),
        "Scripts",
        "Stan",
        "MCBBVoP_WinProb.stan"))

### sampling from posterior distributions
WP_VoP_fit = WP_VoP_model.sample(
    data = VoP_WP_datalist,
    chains=3,
    iter_sampling = 10000,
    iter_warmup = 5000,
    seed = 802,
    parallel_chains = os.cpu_count() // 2
)

### summarizing and diagnosing model
print(WP_VoP_fit.summary())
print(WP_VoP_fit.diagnose())

### saving model fit
# WP_VoP_fit.save_csvfiles(os.path.join(
#     os.getcwd(),
#     "Data",
#     "FittedModels",
#     "CFBVoP_WPModel"
# ))

### extracting parameters from Stan VoP Win Probability model
WP_VoP_pars = pl.from_pandas(WP_VoP_fit.draws_pd(vars=["alpha", "beta_score"]))

### Assigning Ratings to the corresponding home and away teams, adding win probability
### using a home court advantage of 3 points until I can figure out something better
UpcomingGames_df = UpcomingGames_df.join(HomeVoARating, on = 'home_team', how = 'left')
UpcomingGames_df = UpcomingGames_df.join(AwayVoARating, on = 'away_team', how = 'left').with_columns(
    proj_margin = pl.when(pl.col('neutral_site') == False).then(
        pl.col('away_rating') - (pl.col('home_rating') + 3)
    ).otherwise(
        pl.col('away_rating') - pl.col('home_rating')
    )
)

### using Stan model fit above to calculate win probability values for each game based on projected margin
### using matrix math to turn posterior samples from Stan model into win probability projections for each game
WP_design_matrix = np.column_stack(
    [
        np.ones(UpcomingGames_df.height),
        UpcomingGames_df["proj_margin"].abs().to_numpy(),
    ]
)

### Extract Parameter Matrix: (N_draws x 2)
WP_VoP_pars_matrix = WP_VoP_pars.select(["alpha", "beta_score"]).to_numpy()

### Matrix Multiplication -> Linear Predictors (N_draws x N_games)
WP_means_matrix = WP_VoP_pars_matrix @ WP_design_matrix.T

### Inverse logit transformation to win probabilities
VoP_win_probs = 1 / (1 + np.exp(-WP_means_matrix))

### Calculate median win probability across draws/ for each game (axis=0 operates across draws)
VoP_median_win_probs = np.median(VoP_win_probs, axis=0)


UpcomingGames_df = UpcomingGames_df.with_columns(
    proj_winner = pl.when(pl.col('proj_margin') < 0).then(pl.col('home_team'))
    .when(pl.col('proj_margin') > 0).then(pl.col('away_team'))
    .otherwise(pl.lit('TIE')),
    win_prob = pl.Series(VoP_median_win_probs)
).select(['id', 'source_id', 'season_label', 'season', 'season_type', 'start_date', 'start_time_tbd', 'neutral_site', 'conference_game', 'game_type', 'tournament', 'game_notes', 'status', 'home_team_id', 'home_team', 'home_conference_id', 'home_conference', 'home_seed', 'away_team_id', 'away_team', 'away_conference_id', 'away_conference', 'away_seed', 'venue_id', 'venue', 'city', 'state', 'home_rating', 'away_rating', 'proj_margin', 'proj_winner', 'win_prob'])


### creating dataframe for cbbd submission
CBBDSubmission_df = UpcomingGames_df.select(['id', 'home_team', 'away_team', 'proj_margin']).rename(
    {'id': 'id', 
    'home_team': 'home',
    'away_team': 'away',
    'proj_margin': 'predicted'}
)

##### Saving Projections #####
### the upcoming games_df will contain multiple background info columns in case I want to go back and know more about the games later, the cbbd df only contains the columns necessary for aligning with the submission protocol for the CBBD pickem contest
### binding upcoming games to prior preds before saving, taking most recent projection in case of duplicate game entries
if int(voa_num) > 0:
    UpcomingGames_df = pl.concat([PriorPreds, UpcomingGames_df], how = 'diagonal').group_by("id").last()
else:
    print("first projections, so no need to bind newest game projections to older projections")

### saving updated game projections
UpcomingGames_df.write_parquet(os.path.join(os.getcwd(), 'Data', 'VoA' + str(cbb_season), 'Projections', 'MCBBVoA' + datestring + 'GameProjections.parquet'))
CBBDSubmission_df.write_csv(os.path.join(os.getcwd(), 'Data', 'VoA' + str(cbb_season), 'Projections', 'CBBD', 'MCBBVoACBBDGameProjections.csv'))


##### Making table showing games and projected margins #####
### selecting columns to be shown in gt table
UpcomingGamesTable_df = UpcomingGames_df.filter(
    (pl.col('start_date') > today_dt)).with_columns(
            proj_margin_abs = pl.col('proj_margin').abs()
            ).select(
        ['tournament', 'start_date', 'home_team', 'home_rating', 'away_team', 'away_rating', 'proj_margin', 'proj_winner', 'proj_margin_abs', 'win_prob']
        )
UpcomingGamesTable_df_sorted = UpcomingGamesTable_df.sort('proj_margin_abs', descending = True)
### creating games table using gt
GamesTable_gt = (
    GT(UpcomingGamesTable_df)
    .tab_header(
        title = cbb_season_str + " MCBB D1 Vortex of Accuracy Upcoming Games",
        subtitle = "The Unquestionably Puzzling Yet Impeccibly Perceptive Vortex of Projection, " + str(today_dt.date()) + ' - ' + str(UpcomingGamesTable_df['start_date'].max().date())
    )
    # Formatting numbers (grouped by decimal count for efficiency)
    .fmt_number(
        columns = ["home_rating", "away_rating", "win_prob"],
        decimals = 3
    )
    .fmt_number(
        columns=["proj_margin_abs"],
        decimals = 1
    )
    .data_color(
        columns = ['proj_margin_abs'],
        palette = "RdBu",
        na_color = "white",
        ### this autocolor line was added because there's apparently a bug in great-tables?
        ## stack overflow also suggested upgrading but I did that and nothing happened (I went from 0.18.0 to 0.21.0, same key error occured "'font_color_row_striping_background_color'")
        # autocolor_text = False
    )
    .data_color(
        columns = ["win_prob"],
        palette = "RdYlGn",
        reverse = False,
        na_color = "white",
        ### turning off autocolor_text for same reason described above
        # autocolor_text = False
    )
    # Column Labels
    .cols_label(
        tournament = "Tournament",
        home_team = "Home Team",
        home_rating = "Home VoA Rating",
        away_team = "Away Team",
        away_rating = "Away VoA Rating",
        proj_margin_abs = "Projected Margin",
        proj_winner = "Projected Winner",
        win_prob = "Win Probability"
    )
    # Hide columns
    .cols_hide(columns=['proj_margin', 'tournament', 'start_date'])
    # Add Footnote
    .tab_source_note(
        source_note = "Table by @gshelor, Data from CBBD API"
    )
)

### creating games table sorted by projected margin using gt
### creating games table using gt
GamesTable_sorted_gt = (
    GT(UpcomingGamesTable_df_sorted)
    .tab_header(
        title = cbb_season_str + " MCBB D1 Vortex of Accuracy Upcoming Games",
        subtitle = "The Unquestionably Puzzling Yet Impeccibly Perceptive Vortex of Projection, " + str(today_dt.date()) + ' - ' + str(UpcomingGamesTable_df['start_date'].max().date())
    )
    # Formatting numbers (grouped by decimal count for efficiency)
    .fmt_number(
        columns = ["home_rating", "away_rating", "win_prob"],
        decimals = 3
    )
    .fmt_number(
        columns=["proj_margin_abs"],
        decimals = 1
    )
    .data_color(
        columns = ['proj_margin_abs'],
        palette = "RdBu",
        na_color = "white",
        ### this autocolor line was added because there's apparently a bug in great-tables?
        ## stack overflow also suggested upgrading but I did that and nothing happened (I went from 0.18.0 to 0.21.0, same key error occured "'font_color_row_striping_background_color'")
        # autocolor_text = False
    )
    .data_color(
        columns = ["win_prob"],
        palette = "RdYlGn",
        reverse = False,
        na_color = "white",
        ### turning off autocolor_text for same reason described above
        # autocolor_text = False
    )
    # Column Labels
    .cols_label(
        tournament = "Tournament",
        home_team = "Home Team",
        home_rating = "Home VoA Rating",
        away_team = "Away Team",
        away_rating = "Away VoA Rating",
        proj_margin_abs = "Projected Margin",
        proj_winner = "Projected Winner",
        win_prob = "Win Probability"
    )
    # Hide columns
    .cols_hide(columns=['proj_margin', 'tournament', 'start_date'])
    # Add Footnote
    .tab_source_note(
        source_note = "Table by @gshelor, Data from CBBD API"
    )
)

### displaying tables
GamesTable_gt.show()
GamesTable_sorted_gt.show()
GamesTable_gt.gtsave(file = os.path.join(os.getcwd(), "Outputs", "VoA" + str(cbb_season), "MCBBVoACurrentProjections.png"))



