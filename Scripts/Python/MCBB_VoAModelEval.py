##### D1 Men's College Basketball Vortex of Accuracy Model Accuracy #####
##### importing libraries and setting up python environment and API connection #####
import cbbd
import polars as pl
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from dotenv import load_dotenv
from datetime import date, datetime
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
### creating string of date combined together so I can identify unique projections compiled during the course of a season
if today_dt.month >= 10:
    datestring = str(today_dt.year) + str(today_dt.year + 1)
else:
    datestring = str(today_dt.year - 1 ) + str(today_dt.year) #+ str(today_dt.month) + str(today_dt.day)

### setting VoA number so the script knows which VoA csv to read in
eval_check = input("Is this the first model eval of the season? (y/n) ")

### setting up a for loop so I can add a break statement at the end of the if statement
for i in eval_check:
    if eval_check == 'y':
        ### reading in csv of prior predictions to be bound to upcoming games df so I can save all predictions together in one csv
        GamePreds = pl.read_csv(os.path.join(os.getcwd(), "Data", "VoA" + str(cbb_season), "Projections", "MCBBVoA" + datestring + "GameProjections.csv"), schema = {
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
    elif eval_check == 'n':
        ### previously saved df with games, margins, and accuracy metrics for each game will be read in here
        ## not summary csv with just accuracy metrics
        ### reading in a csv of games with accuracy metrics added, calling it VoAGames
        print("we'll add in the code here later")
        # VoAGames = pl.read_csv()
    else:
        print("only input 'y' or 'n', no other characters, try again")
        break


### setting up API configuration
configuration = cbbd.Configuration(
    host = "https://api.collegebasketballdata.com", 
    access_token = os.getenv("CBB_TOKEN"))
api_client = cbbd.ApiClient(configuration)

### creating an instance with the lines API endpoint
linesapi_instance = cbbd.LinesApi(api_client)

### getting games from relevant time period
if eval_check == 'y':
    Lines_json = linesapi_instance.get_lines(start_date_range = GamePreds['start_date'].min(), end_date_range = GamePreds['start_date'].max())
    ### cleaning API output to turn it into something useful
    CompletedGames = get_clean_lines(lines_json = Lines_json, games_df = GamePreds).with_columns(
        VoA_AE = (pl.col('actual_margin') - pl.col('proj_margin')).abs(),
        vegas_AE = (pl.col('actual_margin') - pl.col('mean_spread')).abs(),
        VoA_SE = (pl.col('actual_margin') - pl.col('proj_margin')).pow(2),
        vegas_SE = (pl.col('actual_margin') - pl.col('mean_spread')).pow(2),
        VoA_correct_winner = pl.when(
            ((pl.col('proj_margin') < 0) & (pl.col('actual_margin') < 0)) | 
            ((pl.col('proj_margin') > 0) & (pl.col('actual_margin') > 0))).then(1)
            .otherwise(0),
        vegas_correct_winner = pl.when(
            ((pl.col('actual_margin') < 0) & (pl.col('mean_spread') < 0)) | 
            ((pl.col('actual_margin') > 0) & (pl.col('mean_spread') > 0))).then(1)
            .otherwise(0),
        VoA_ATS_winner = pl.when(
            ((pl.col('proj_margin') < pl.col('mean_spread')) & (pl.col('actual_margin') < pl.col('mean_spread'))) |
            ((pl.col('proj_margin') > pl.col('mean_spread')) & (pl.col('actual_margin') > pl.col('mean_spread')))).then(1)
        .otherwise(0)
    ).with_columns(
        VoA_AEATS_winner = pl.when(pl.col('VoA_AE') < pl.col('vegas_AE')).then(1).otherwise(0)
    )
else:
    Lines_json = linesapi_instance.get_lines(start_date_range = VoAGames['start_date'].max(), end_date_range = GamePreds['start_date'].max())
    ### cleaning API output to turn it into something useful
    NewCompletedGames = get_clean_lines(lines_json = Lines_json, games_df = GamePreds).with_columns(
        VoA_AE = (pl.col('actual_margin') - pl.col('proj_margin')).abs(),
        vegas_AE = (pl.col('actual_margin') - pl.col('mean_spread')).abs(),
        VoA_SE = (pl.col('actual_margin') - pl.col('proj_margin')).pow(2),
        vegas_SE = (pl.col('actual_margin') - pl.col('mean_spread')).pow(2),
        VoA_correct_winner = pl.when(
            ((pl.col('proj_margin') < 0) & (pl.col('actual_margin') < 0)) | 
            ((pl.col('proj_margin') > 0) & (pl.col('actual_margin') > 0))).then(1)
            .otherwise(0),
        vegas_correct_winner = pl.when(
            ((pl.col('actual_margin') < 0) & (pl.col('mean_spread') < 0)) | 
            ((pl.col('actual_margin') > 0) & (pl.col('mean_spread') > 0))).then(1)
            .otherwise(0),
        VoA_ATS_winner = pl.when(
            ((pl.col('proj_margin') < pl.col('mean_spread')) & (pl.col('actual_margin') < pl.col('mean_spread'))) |
            ((pl.col('proj_margin') > pl.col('mean_spread')) & (pl.col('actual_margin') > pl.col('mean_spread')))).then(1)
        .otherwise(0)
    ).with_columns(
        VoA_AEATS_winner = pl.when(pl.col('VoA_AE') < pl.col('vegas_AE')).then(1).otherwise(0)
    )
    # CompletedGames = VoAGames.concat()


SeasonAccuracy = pl.DataFrame({
    'season' : datestring, 
    'games' : CompletedGames.height,
    'VoA_MAE' : CompletedGames['VoA_AE'].mean(),
    'vegas_MAE' : CompletedGames['vegas_AE'].mean(),
    'VoA_MSE' : CompletedGames['VoA_SE'].mean(),
    'vegas_MSE' : CompletedGames['vegas_SE'].mean(),
    'VoA_RMSE' : np.sqrt(CompletedGames['VoA_SE'].mean()),
    'vegas_RMSE' : np.sqrt(CompletedGames['vegas_SE'].mean()),
    'VoA_win_pct' : CompletedGames['VoA_correct_winner'].mean(),
    'vegas_win_pct' : CompletedGames['vegas_correct_winner'].mean(),
    'VoA_ATS_win_pct' : CompletedGames['VoA_ATS_winner'].mean(),
    'VoA_AEATS_win_pct' : CompletedGames['VoA_AEATS_winner'].mean()
}
)








##### POOPYPANTS TESTING, PLEASE IGNORE #####
