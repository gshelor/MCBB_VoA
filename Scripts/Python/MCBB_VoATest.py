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

### making sure I can load the csv files from cmdstanpy's save_csvfiles function
poopypants = cmdstanpy.from_csv(os.path.join(
        os.getcwd(),
        "Data",
        "FittedModels",
        "OffVoAStanFit.csv"
    ))

poopypants.diagnose()
poopypants.summary()
OffVoA_draws_dict = poopypants.stan_variables()


    ### saving parameters as a parquet file so I don't have to re-fit the model each time I run the VoA
    # OffVoA_pars.write_parquet(os.path.join(
    #     os.getcwd(),
    #     "Data",
    #     "FittedModels",
    #     "OffVoA_pars.parquet"
    # ))

     ### using matrix math to turn posterior samples from Stan model into win probability projections for each game
OffVoA_design_matrix = np.column_stack(
    [
        np.ones(VoAStats.height),
        # VoAStats["weighted_adjoff_ppg"].to_numpy(),
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

    ### list of parameter names (sigma to be factored in later)
OffVoA_param_names = ["b0", "beta_off_ppp", "beta_off_assists", "beta_off_trueshooting", "beta_off_oppblocks", "beta_off_oppsteals", "beta_off_fgpct", "beta_off_2ptfgpct", "beta_off_3ptfgpct", "beta_off_ftpct", "beta_off_rebounds", "beta_off_turnovers"]

    ### Extract Parameter Matrix: (N_draws x 2)
    # OffVoA_pars_matrix = OffVoA_pars.select(["b0", "beta_off_ppp", "beta_off_assists", "beta_off_trueshooting", "beta_off_oppblocks", "beta_off_oppsteals", "beta_off_fgpct", "beta_off_2ptfgpct", "beta_off_3ptfgpct", "beta_off_ftpct", "beta_off_rebounds", "beta_off_turnovers"]).to_numpy()
OffVoA_pars_matrix = np.column_stack([OffVoA_draws_dict[name] for name in OffVoA_param_names])

    ### extracting sigma to include error
OffVoA_sigma = OffVoA_draws_dict["sigma"]

    ### Matrix Multiplication -> Linear Predictors (N_draws x N_games)
OffVoA_means_matrix = OffVoA_pars_matrix @ OffVoA_design_matrix.T

    ### applying sigma parameter to posterior samples of adjusted off ppg to generate Off VoA Ratings
P = len(OffVoA_draws_dict['b0'])
T_num = VoAStats.height
np.random.seed(802)

noise = np.random.normal(loc=0.0, scale = OffVoA_sigma[:, np.newaxis], size=(P, T_num))
OffVoARatings = OffVoA_means_matrix + noise

    ### creating final ratings columns
OffVoA_MeanRatings = np.mean(OffVoARatings, axis = 0)
OffVoA_MedianRatings = np.median(OffVoARatings, axis = 0)
OffVoA_upper = np.percentile(OffVoARatings, 97.5, axis = 0)
OffVoA_lower = np.percentile(OffVoARatings, 2.5, axis = 0)
OffVoA_StDev = np.std(OffVoARatings, axis = 0)
