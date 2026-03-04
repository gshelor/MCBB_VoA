##### Vortex of Accuracy Functions #####
### this script contains functions used, mostly for cleaning data, in the Men's Division 1 College Basketball Vortex of Accuracy
### the CBBD api outputs stuff in basically lists/dictionaries of data that can sort of be easily coerced into a useful dataframe format except for some columns. So there's extra steps that I would need to constantly repeat every time I run the VoA, and this script will hopefully contain functions that prove useful
### this script is intended to be run within MCBB_VoA.py as a way of reading in functions, so nothing should actually be executed here, only defining of functions

### importing some modules
## these should honestly already be imported based on how I intend to use this script but here they are for redundancy or whatever I guess
import cbbd
import polars as pl
import numpy as np
import random
import statsmodels.formula.api as smf

##### Function for cleaning season team stats API output #####
def clean_season_stats(json_obj):
    ### coercing list output with data in json format to polars dataframe
    TeamSeasonStats_df = pl.DataFrame(json_obj)
    ### taking nested columns out of original team season stats data frame
    ## binding them to this data frame
    VoATeamStats = TeamSeasonStats_df.select(
        pl.selectors.by_dtype(pl.Int64, pl.String, pl.Float64)
    )
    ### unnesting struct columns so that each individual stat can be its own column
    TeamStatsCols = TeamSeasonStats_df['team_stats'].struct.unnest()
    OppTeamStatsCols = TeamSeasonStats_df['opponent_stats'].struct.unnest()
    ### more unnesting necessary, and also setting column names so each column is unique
    for i in np.arange(0, len(TeamStatsCols.columns)):
        if TeamStatsCols[TeamStatsCols.columns[i]].dtype == pl.Struct:
            temp_df = TeamStatsCols[TeamStatsCols.columns[i]].struct.unnest()
            col_prefix = str(TeamStatsCols.columns[i])
            temp_df = temp_df.rename(lambda col_name: col_prefix + "_" + col_name)
            VoATeamStats = pl.concat([VoATeamStats, temp_df], how = "horizontal")

    ### unnesting opposition team stats info, binding them to VoATeamStats
    for i in np.arange(0, len(OppTeamStatsCols.columns)):
        if OppTeamStatsCols[OppTeamStatsCols.columns[i]].dtype == pl.Struct:
            temp_df = OppTeamStatsCols[OppTeamStatsCols.columns[i]].struct.unnest()
            col_prefix = str(OppTeamStatsCols.columns[i])
            temp_df = temp_df.rename(lambda col_name: "opp_" + col_prefix + "_" + col_name)
            VoATeamStats = pl.concat([VoATeamStats, temp_df], how = "horizontal")
    ### binding unnested team stats and opposition stats (for each team's opponents) to original team variables df
    # VoATeamStats = pl.concat(items = [VoATeamStats, TeamStats_df, OppTeamStats_df], how = "horizontal")
    ### filtering out non-D1 teams
    ## doesn't seem to be a great way to do this since there's no Division/classification field output by the CBBD API unless I wanted to do it manually (i do not), but conference luckily seems to serve as a perfectly fine proxy for the same thing since teams not in D1 are just not listed as having a conference
    VoATeamStats = VoATeamStats.filter(
        ~pl.col("conference").is_null()
    )
    return VoATeamStats


##### Function for cleaning Games dataframe to extract team stats for each game #####
### data will be used for opponent adjustments
def clean_team_game_stats(json_obj):
    ### converting API output to a polars dataframe, and removing duplicate game ids since each team in a game gets its own row for some reason
    Games_df = pl.DataFrame(json_obj, infer_schema_length = None)
    ### unnesting stats columns for each team and their respective opponents' averages (against them)
    GamesStatsCols = Games_df['team_stats'].struct.unnest()
    GamesOppStatsCols = Games_df['opponent_stats'].struct.unnest()
    ### some columns are not structs after the initial unnesting of the 'team_stats' struct column
    ### since I want to keep those columns, they are selected out so the remaining structs can be further unnested
    GamesStatsCols_NoStructs = GamesStatsCols.select(pl.selectors.by_dtype(pl.Int64, pl.String, pl.Float64))
    GamesOppStatsCols_NoStructs = GamesOppStatsCols.select(pl.selectors.by_dtype(pl.Int64, pl.String, pl.Float64)).rename(lambda col_name: "opp_" + col_name)
    GamesStats_df = pl.concat([GamesStatsCols_NoStructs, GamesOppStatsCols_NoStructs], how = "horizontal")
    ### Unnesting team and opponent stats for games
    ## will not be bound to the season team stats df, only used for opponent adjustments
    for i in np.arange(0, len(GamesStatsCols.columns)):
        if GamesStatsCols[GamesStatsCols.columns[i]].dtype == pl.Struct:
            temp_df = GamesStatsCols[GamesStatsCols.columns[i]].struct.unnest()
            col_prefix = str(GamesStatsCols.columns[i])
            temp_df = temp_df.rename(lambda col_name: col_prefix + "_" + col_name)
            GamesStats_df = pl.concat([GamesStats_df, temp_df], how = "horizontal")
    ### unnesting opposition game stats
    for i in np.arange(0, len(GamesOppStatsCols.columns)):
        if GamesOppStatsCols[GamesOppStatsCols.columns[i]].dtype == pl.Struct:
            temp_df = GamesOppStatsCols[GamesOppStatsCols.columns[i]].struct.unnest()
            col_prefix = str(GamesOppStatsCols.columns[i])
            temp_df = temp_df.rename(lambda col_name: "opp_" + col_prefix + "_" + col_name)
            GamesStats_df = pl.concat([GamesStats_df, temp_df], how = "horizontal")
    ### taking nested columns out of original team season stats data frame
    ## binding them to this data frame
    ### basically I want everything that isn't a struct to stay, since the structs have been unnested and will be bound back as many different columns of types listed here
    FinalGames_df = pl.concat([
        Games_df.select(
            pl.selectors.by_dtype(pl.Int64, pl.String, pl.Float64, pl.Datetime, pl.Boolean, pl.Null)
    ), GamesStats_df], how = "horizontal").drop_nulls(
        subset = ["game_id", "season", "season_type", "team", "conference", "opponent", "opponent_conference", "neutral_site", "pace", "assists", "blocks", "steals", "possessions", "rating", "true_shooting", "game_score", pl.selectors.starts_with("opp_"), pl.selectors.starts_with("field_goals"), pl.selectors.starts_with("two_point_"), pl.selectors.starts_with("three_point_"), pl.selectors.starts_with("free_throws_"), pl.selectors.starts_with("rebounds_"), pl.selectors.starts_with("turnovers_"), pl.selectors.starts_with("fouls_"), pl.selectors.starts_with("points_"), pl.selectors.starts_with("four_factors_")]
    )

    ### returning FinalGames_df
    return FinalGames_df


def opponent_adjustments(pace_df, full_df):
    ### just gonna stick all the opponent-adjustments here
    ### Pace opponent adjustments
    random.seed(802)
    ### specifying formula, group effects, variance components
    pace_oppadj_model = smf.mixedlm(formula = "pace ~ hfa", data = pace_df.to_pandas(), groups = pace_df['team'], vc_formula={"opponent": "1 + C(opponent)"})
    ### fitting model
    ## For future: weight games by their time since current date, so more recent games are weighted more heavily than others
    pace_oppadj_fit = pace_oppadj_model.fit()
    pace_oppadj_fit.summary()
    ### Extract adjustments and join to VoAVariables using polars and pandas (ugh)
    ### extracting intercept
    pace_intercept = pace_oppadj_fit.params['Intercept']

    ### Extract adjusted coefficients and apply them to intercept to get adjusted metric
    team_pace_coefs = [
        {"team": team_name, "adjpace_coef": float(effect.iloc[0])} 
        for team_name, effect in pace_oppadj_fit.random_effects.items()
    ]
    ### calculating opponent-adjusted pace
    paceadj_df = pl.DataFrame(team_pace_coefs).with_columns(
        adj_pace = pl.col('adjpace_coef') + pace_intercept
    ).select(['team', 'adj_pace'])