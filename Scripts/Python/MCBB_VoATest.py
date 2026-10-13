##### This file is for just testing stuff out outside of the if/else confines of the main VoA/VoP/etc scripts #####
### no code here is meant to be operationalized or whatever, it's just me trying stuff and seeing what happens

TeamSeasonStats_json_PY1 = statsapi_instance.get_team_season_stats(cbb_season - 1)
VoATrain_PY1 = clean_season_stats(TeamSeasonStats_json_PY1)
Games_json1 = gamesapi_instance.get_game_teams(start_date_range = datetime(2025, 10, 1), end_date_range = datetime(2025, 11, 30, 23, 59, 59))
Games_json2 = gamesapi_instance.get_game_teams(start_date_range = datetime(2025, 12, 1), end_date_range = datetime(2025, 12, 31, 23, 59, 59))
Games_json3 = gamesapi_instance.get_game_teams(start_date_range = datetime(2026, 1, 1), end_date_range = datetime(2026, 1, 24, 23, 59, 59))
Games_json4 = gamesapi_instance.get_game_teams(start_date_range = datetime(2026, 1, 25), end_date_range = datetime(2026, 2, 15, 23, 59, 59))
Games_json5 = gamesapi_instance.get_game_teams(start_date_range = datetime(2026, 2, 16), end_date_range = datetime(2026, 5, 1, 23, 59, 59))
Games_df1 = clean_team_game_stats(Games_json1)
Games_df2 = clean_team_game_stats(Games_json2)
Games_df3 = clean_team_game_stats(Games_json3)
Games_df4 = clean_team_game_stats(Games_json4)
Games_df5 = clean_team_game_stats(Games_json5)


Games_df = pl.concat([Games_df1, Games_df2, Games_df3, Games_df4, Games_df5], how = "vertical_relaxed")

Games_df = Games_df.with_columns(
    pl.when(pl.col('neutral_site') == True).then(0)
    .when(pl.col('neutral_site') == False, pl.col('is_home') == True).then(1)
    .otherwise(-1)
    .alias("hfa"),
    points_per_poss = pl.col("points_total") / pl.col("possessions"),
    opp_points_per_poss = pl.col("opp_points_total") / pl.col("opp_possessions")
    ).drop_nulls(["conference", "opponent_conference"])


test_col = "adjoff_field_goals_pct"

### using bambi to fit a bayesian mixed effects model to account for opponent adjustments
model = bmb.Model(
    "field_goals_pct ~ hfa + (1 | conference / team) + (1 | opponent_conference / opponent)",
    data=Games_df.to_pandas(),
)

model = bmb.Model(
    "field_goals_pct ~ hfa + (1 | conference) + (1 | team) + (1 | opponent_conference) + (1 | opponent)",
    data=Games_df.to_pandas(),
)

i = "blocks"
model = pymer4.models.lmer(
    f"{i} ~ hfa + (1 | conference/team) + (1 | opponent_conference/opponent)",
    data=Games_df
)

model.fit()

# model_fit = model.fit()

preds = model.predict(Games_df)


### setting random seed
random.seed(802)
### specifying formula, group effects, variance components
oppadj_model = smf.mixedlm(formula = "field_goals_pct ~ hfa", data = Games_df.to_pandas(), groups = Games_df['team'], vc_formula={
    "opponent": "1 + C(opponent)",
    "conference": "1 + C(conference)",
    "opponent_conference": "1 + C(opponent_conference)"})

smf_fit = oppadj_model.fit()
smf_preds = pl.Series(smf_fit.predict(Games_df))

test_df = Games_df.with_columns(
    lme_fg_pct = preds,
    smf_fg_pct = smf_preds
).group_by("team").agg(
    mean_lme_fg_pct = pl.col("lme_fg_pct").mean(),
    mean_smf_fg_pct = pl.col("smf_fg_pct").mean()
)


# poopypants2 = voa_train_to_stats("_PY1", VoATrain_PY1)
