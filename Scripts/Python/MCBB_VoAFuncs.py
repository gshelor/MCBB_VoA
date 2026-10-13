##### Vortex of Accuracy Functions #####
### this script contains functions used, mostly for cleaning data, in the Men's Division 1 College Basketball Vortex of Accuracy
### the CBBD api outputs stuff in basically lists/dictionaries of data that can sort of be easily coerced into a useful dataframe format except for some columns. So there's extra steps that I would need to constantly repeat every time I run the VoA, and this script will hopefully contain functions that prove useful
### this script is intended to be run within MCBB_VoA.py as a way of reading in functions, so nothing should actually be executed here, only defining of functions

### importing some modules
## these should honestly already be imported based on how I intend to use this script but here they are for redundancy or whatever I guess
from pytensor.graph.basic import op_as_string
import cbbd
import polars as pl
import pandas as pd
import numpy as np
import random
import statsmodels.formula.api as smf
# import bambi as bmb
from pymer4.models import lmer

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

##### Function for calculating opponent and conference-adjusted stats #####
def opponent_adjustments(pace_df, full_df):
    ### just gonna stick all the opponent-adjustments here
    ### Pace opponent adjustments
    random.seed(802)
    ### specifying formula, group effects, variance components
    pace_oppadj_model = smf.mixedlm(formula = "pace ~ hfa", data = pace_df.to_pandas(), groups = pace_df['team'], vc_formula={"opponent": "1 + C(opponent)"})
    ### fitting model
    ## For future: weight games by their time since current date, so more recent games are weighted more heavily than others
    pace_oppadj_fit = pace_oppadj_model.fit()
    # pace_oppadj_fit.summary()
    ### Extract adjustments and join to VoAStats using polars and pandas (ugh)
    ### extracting intercept
    pace_intercept = pace_oppadj_fit.params['Intercept']

    ### Extract adjusted coefficients and apply them to intercept to get adjusted metric
    team_pace_coefs = [
        {"team": team_name, "adjpace_coef": float(effect.iloc[0])} 
        for team_name, effect in pace_oppadj_fit.random_effects.items()
    ]
    ### calculating opponent-adjusted pace
    ### since pace is the first stat adjusted, this dataframe with the team and their corresponding adjusted value will be the dataframe that all the other adjusted stats dfs are joined to
    AdjStats_df = pl.DataFrame(team_pace_coefs).with_columns(
        adj_pace = pl.col('adjpace_coef') + pace_intercept
    ).select(['team', 'adj_pace'])

    ### performing opponent adjustments for select offensive stats
    ## offensive stats: assists, true_shooting, opp_blocks, opp_steals, field_goals_pct, two_point_field_goals_pct, three_point_field_goals_pct, free_throws_pct, rebounds_offensive, turnovers_total, points_fast_break, points_off_turnovers, points_in_paint
    ## defensive stats: opp_assists, opp_true_shooting, blocks, steals, opp_field_goals_pct, opp_two_point_field_goals_pct, opp_three_point_field_goals_pct, opp_free_throws_pct, rebounds_defensive, opp_turnovers_total, opp_points_fast_break, opp_points_off_turnovers, opp_points_in_paint

    ### fitting opponent-adjusted models for offensive stats now
    ### storing output dataframes in a list, will store the output dfs from the defensive models in this list too
    adj_dflist = []
    for i in ["assists", "true_shooting", "opp_blocks", "opp_steals", "field_goals_pct", "two_point_field_goals_pct", "three_point_field_goals_pct", "free_throws_pct", "rebounds_offensive", "turnovers_total", "points_fast_break", "points_off_turnovers", "points_in_paint", "points_per_poss"]:
        ### setting name of new column where opponent-adjusted stats will be stored
        new_colname = "adjoff_" + i
        ### setting random seed
        random.seed(802)
        ### specifying formula, group effects, variance components
        oppadj_model = lmer(formula = f"{i} ~ hfa + (1 | conference/team) + (1 | opponent_conference/opponent)", data = full_df)
        ### fitting model
        ## For future: weight games by their time since current date, so more recent games are weighted more heavily than others
        oppadj_model.fit()
        # oppadj_fit.summary()
        ### Extract adjustments and join to VoAStats using polars and pandas (ugh)
        ### making predictions (predictions are the adjusted value, will be aggregated to get a mean over the course of a season)
        adj_preds = oppadj_model.predict(full_df)
        adj_df = full_df.with_columns(
            adj_stat = adj_preds
        ).group_by("team").agg(
            pl.col("adj_stat").mean().alias(new_colname)
        )

        ### appending df of team names and opp-adjusted stats to the list above
        adj_dflist.append(adj_df)

    ### fitting opponent-adjusted models for defensive stats now
    ### storing output dataframes in the same list as above
    for i in ["opp_assists", "opp_true_shooting", "blocks", "steals", "opp_field_goals_pct", "opp_two_point_field_goals_pct", "opp_three_point_field_goals_pct", "opp_free_throws_pct", "rebounds_defensive", "opp_turnovers_total", "opp_points_fast_break", "opp_points_off_turnovers", "opp_points_in_paint", "opp_points_per_poss"]:
        ### setting name of new column where opponent-adjusted stats will be stored
        new_colname = "adjdef_" + i
        ### setting random seed
        random.seed(802)
        ### specifying formula, group effects, variance components
        oppadj_model = lmer(formula = f"{i} ~ hfa + (1 | conference/team) + (1 | opponent_conference/opponent)", data = full_df)
        ### fitting model
        ## For future: weight games by their time since current date, so more recent games are weighted more heavily than others
        oppadj_model.fit()
        # oppadj_fit.summary()
        ### Extract adjustments and join to VoAStats using polars and pandas (ugh)
        ### making predictions (predictions are the adjusted value, will be aggregated to get a mean over the course of a season)
        adj_preds = oppadj_model.predict(full_df)
        adj_df = full_df.with_columns(
            adj_stat = adj_preds
        ).group_by("team").agg(
            pl.col("adj_stat").mean().alias(new_colname)
        )

        ### appending df of team names and opp-adjusted stats to the list above
        adj_dflist.append(adj_df)

    ### binding dataframes of adjusted stats together
    for x in np.arange(0, len(adj_dflist)):
        AdjStats_df = AdjStats_df.join(adj_dflist[x], on = "team")

    ### returning full df of adjusted stats and team, which will be bound to VoAStats in the VoA script
    return AdjStats_df

### making a function for ad-hoc game projections (like when I'm going through the bracket seeing who VoA thinks would win but can't conventiently get game IDs from CBBD API)
def game_projections(home_team, away_team, neutral):
    voa_ratings = VoAStats['OvrlVoA_MeanRating'].to_numpy()
    voa_teams = VoAStats['team'].to_numpy()
    home_rating = voa_ratings[np.where(voa_teams == home_team)]
    away_rating = voa_ratings[np.where(voa_teams == away_team)]
    if neutral == True:
        proj_margin = away_rating - home_rating
    else:
        proj_margin = away_rating - (home_rating + 3)
    return proj_margin.item()


def get_clean_lines(lines_json, games_df):
    ### converting lines object to a polars dataframe
    Lines_df = pl.DataFrame(lines_json, strict = False)
    ### the column that contains the actual lines for the games is a column of lists of "structs" so I take all the items out of the structs and make them their own column, then pivot so that games with multiple spreads will have multiple columns for each spread provider
    Lines_df = Lines_df.explode('lines').unnest('lines')
    ### pivoting so that each spread provider gets their own column, which I will then average since I don't gamble and thus have no clue whether one spread is going to be more reliable than another
    Lines_df = Lines_df.pivot(
        values=["spread", "over_under"],
        index=["game_id", "season", "season_type", "start_date", "home_team_id", "home_team", "home_conference", "home_score", "away_team_id", "away_team", "away_conference", "away_score"], # game identifier
        on="provider"
    ).with_columns(
        mean_spread = pl.mean_horizontal(pl.selectors.starts_with("spread"), ignore_nulls = True),
        ### calculating actual margin, to be compared to mean_spread and VoA projection
        ## away score minus home score because of how the CBBD API takes projections, so that's how I set up the VoA projections too
        actual_margin = pl.col('away_score') - pl.col('home_score')
    ).select(
        ['game_id', 'home_score', 'away_score', 'mean_spread', 'actual_margin']
    ).rename({'game_id': 'id'})

    games_lines = games_df.join(Lines_df, on = 'id', how = 'left')

    return games_lines


##### function to get game data from full previous seasons #####
def get_PY_game_data(season_num):
    Games_json1 = gamesapi_instance.get_game_teams(start_date_range = datetime(season_num, 10, 1), end_date_range = datetime(season_num, 11, 30, 23, 59, 59))
    Games_json2 = gamesapi_instance.get_game_teams(start_date_range = datetime(season_num, 12, 1), end_date_range = datetime(season_num, 12, 31, 23, 59, 59))
    Games_json3 = gamesapi_instance.get_game_teams(start_date_range = datetime(season_num + 1, 1, 1), end_date_range = datetime(season_num + 1, 1, 24, 23, 59, 59))
    Games_json4 = gamesapi_instance.get_game_teams(start_date_range = datetime(season_num + 1, 1, 25), end_date_range = datetime(season_num + 1, 2, 15, 23, 59, 59))
    Games_json5 = gamesapi_instance.get_game_teams(start_date_range = datetime(season_num + 1, 2, 16), end_date_range = datetime(season_num + 1, 5, 1, 23, 59, 59))
    Games_df1 = clean_team_game_stats(Games_json1).with_columns(
            pl.when(pl.col('neutral_site') == True).then(0)
            .when(pl.col('neutral_site') == False, pl.col('is_home') == True).then(1)
            .otherwise(-1)
            .alias("hfa"),
            points_per_poss = pl.col("points_total") / pl.col("possessions"),
            opp_points_per_poss = pl.col("opp_points_total") / pl.col("opp_possessions")
        )
    Games_df2 = clean_team_game_stats(Games_json2).with_columns(
            pl.when(pl.col('neutral_site') == True).then(0)
            .when(pl.col('neutral_site') == False, pl.col('is_home') == True).then(1)
            .otherwise(-1)
            .alias("hfa"),
            points_per_poss = pl.col("points_total") / pl.col("possessions"),
            opp_points_per_poss = pl.col("opp_points_total") / pl.col("opp_possessions")
        )
    Games_df3 = clean_team_game_stats(Games_json3).with_columns(
            pl.when(pl.col('neutral_site') == True).then(0)
            .when(pl.col('neutral_site') == False, pl.col('is_home') == True).then(1)
            .otherwise(-1)
            .alias("hfa"),
            points_per_poss = pl.col("points_total") / pl.col("possessions"),
            opp_points_per_poss = pl.col("opp_points_total") / pl.col("opp_possessions")
        )
    Games_df4 = clean_team_game_stats(Games_json4).with_columns(
            pl.when(pl.col('neutral_site') == True).then(0)
            .when(pl.col('neutral_site') == False, pl.col('is_home') == True).then(1)
            .otherwise(-1)
            .alias("hfa"),
            points_per_poss = pl.col("points_total") / pl.col("possessions"),
            opp_points_per_poss = pl.col("opp_points_total") / pl.col("opp_possessions")
        )
    Games_df5 = clean_team_game_stats(Games_json5).with_columns(
            pl.when(pl.col('neutral_site') == True).then(0)
            .when(pl.col('neutral_site') == False, pl.col('is_home') == True).then(1)
            .otherwise(-1)
            .alias("hfa"),
            points_per_poss = pl.col("points_total") / pl.col("possessions"),
            opp_points_per_poss = pl.col("opp_points_total") / pl.col("opp_possessions")
        )

    ### binding games together
    Full_PY_Games = pl.concat([Games_df1, Games_df2, Games_df3, Games_df4, Games_df5], how = 'vertical_relaxed')

    return Full_PY_Games


##### function for converting initial VoATrain dataframes to dataframes that can be left joined after columns are renamed based on what previous season they are from #####
def voa_train_to_stats(year_string, train_df):
    VoAStats_df = train_df.rename(lambda col_name: col_name + year_string).rename(
        {
            "season" + year_string: "season",
            "season_label" + year_string: "season_label",
            "team_id" + year_string: "team_id",
            "team" + year_string: "team",
            "conference" + year_string: "conference"
        }
    ).select(["team", pl.selectors.ends_with(year_string)])

    return VoAStats_df

