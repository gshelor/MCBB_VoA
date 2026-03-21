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
from great_tables import GT
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
    VoAVariables = pl.read_csv(os.path.join(os.getcwd(), "Data", "VoA" + str(cbb_season), "MCBBVoA" + str(cbb_season) + "VoA" + voa_num + ".csv"))
else:
    ### loading upcoming week of games
    VoAVariables = pl.read_csv(os.path.join(os.getcwd(), "Data", "VoA" + str(cbb_season), "MCBBVoA" + str(cbb_season) + "VoA" + voa_num + ".csv"))
    UpcomingGames = gamesapi_instance.get_games(start_date_range = today_dt, end_date_range = today_dt + timedelta(days = 7))
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

##### Saving Projections as csvs #####
### the upcoming games_df will contain multiple background info columns in case I want to go back and know more about the games later, the cbbd df only contains the columns necessary for aligning with the submission protocol for the CBBD pickem contest
### binding upcoming games to prior preds before saving, taking most recent projection in case of duplicate game entries
UpcomingGames_df = pl.concat([PriorPreds, UpcomingGames_df], how = 'diagonal').group_by("id").last()
UpcomingGames_df.write_csv(os.path.join(os.getcwd(), 'Data', 'VoA' + str(cbb_season), 'Projections', 'MCBBVoA' + datestring + 'GameProjections.csv'))
CBBDSubmission_df.write_csv(os.path.join(os.getcwd(), 'Data', 'VoA' + str(cbb_season), 'Projections', 'CBBD', 'MCBBVoACBBDGameProjections.csv'))


##### Making table showing games and projected margins #####
### selecting columns to be shown in gt table
UpcomingGamesTable_df = UpcomingGames_df.filter(
    (pl.col('start_date') > today_dt)).select(
        ['tournament', 'home_team', 'home_rating', 'away_team', 'away_rating', 'proj_margin', 'proj_winner']).with_columns(
            proj_margin_abs = pl.col('proj_margin').abs())
UpcomingGamesTable_df_sorted = UpcomingGamesTable_df.sort('proj_margin_abs', descending = True)
### creating games table using gt
GamesTable_gt = (
    GT(UpcomingGamesTable_df)
    .tab_header(
        title = cbb_season_str + " MCBB D1 Vortex of Accuracy Upcoming Games",
        subtitle = "The Unquestionably Puzzling Yet Impeccibly Perceptive Vortex of Projection, " + str(today_dt.date()) + ' - ' + str((today_dt + timedelta(days = 7)).date())
    )
    # Formatting numbers (grouped by decimal count for efficiency)
    .fmt_number(
        columns = ["home_rating", "away_rating"],
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
    # .data_color(
    #     columns = ["DefVoA_MeanRating"],
    #     palette = "RdYlGn",
    #     reverse = True,
    #     na_color = "white",
    #     ### turning off autocolor_text for same reason described above
    #     # autocolor_text = False
    # )
    # Column Labels
    .cols_label(
        tournament = "Tournament",
        home_team = "Home Team",
        home_rating = "Home VoA Rating",
        away_team = "Away Team",
        away_rating = "Away VoA Rating",
        proj_margin_abs = "Projected Margin",
        proj_winner = "Projected Winner"
    )
    # Hide columns
    .cols_hide(columns=['proj_margin'])
    # Add Footnote
    .tab_source_note(
        source_note = "Table by @gshelor, Data from CBBD API"
    )
)

### creating games table sorted by projected margin using gt
GamesTable_sorted_gt = (
    GT(UpcomingGamesTable_df_sorted)
    .tab_header(
        title = cbb_season_str + " MCBB D1 Vortex of Accuracy Upcoming Games",
        subtitle = "The Unquestionably Puzzling Yet Impeccibly Perceptive Vortex of Projection, " + str(today_dt.date()) + ' - ' + str((today_dt + timedelta(days = 7)).date())
    )
    # Formatting numbers (grouped by decimal count for efficiency)
    .fmt_number(
        columns = ["home_rating", "away_rating"],
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
    # .data_color(
    #     columns = ["DefVoA_MeanRating"],
    #     palette = "RdYlGn",
    #     reverse = True,
    #     na_color = "white",
    #     ### turning off autocolor_text for same reason described above
    #     # autocolor_text = False
    # )
    # Column Labels
    .cols_label(
        tournament = "Tournament",
        home_team = "Home Team",
        home_rating = "Home VoA Rating",
        away_team = "Away Team",
        away_rating = "Away VoA Rating",
        proj_margin_abs = "Projected Margin",
        proj_winner = "Projected Winner"
    )
    # Hide columns
    .cols_hide(columns=['proj_margin'])
    # Add Footnote
    .tab_source_note(
        source_note = "Table by @gshelor, Data from CBBD API"
    )
)

### displaying tables
GamesTable_gt.show()
GamesTable_sorted_gt.show()
GamesTable_gt.save(file = os.path.join(os.getcwd(), "Outputs", "VoA" + str(cbb_season), "MCBBVoACurrentProjections.png"))