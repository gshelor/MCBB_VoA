##### Men's College Basketball Vortex of Accuracy Opponent Adjustments #####
### not the operational script what with the creation of the cbbd data api
### may be turned into script for producing plots and/or gt tables
### may also try to use hoopR for play by play data?
### any time you see "fmt: skip" commented out it's because I don't want Air to format it
## Air seems mostly fine/tolerable-to-good but it does some weird-ass shit on fcase and occasionally some other dplyr-style functions
### this script likely not going to be actually used, I can't do ridge regression on the PBP data without running out of memory, so I'm giving up on it for now I guess
##### Loading Packages #####
library(pacman)
# fmt: skip
p_load(hoopR, tidyverse, gtExtras, ggimage, fastDummies, here, cfbfastR, glmnet, data.table)
### PBP load keeps running into timeout issues, hopefully this fixes it
options(timeout = 180)

### setting season variable based on what month it is
## adding one to the current year if it's October-December, using current year if it's January or later
## this is due to expected inputs of functions that load pbp and schedule data below
cbb_season <- ifelse(
  month(Sys.Date()) >= 10,
  year(Sys.Date()) + 1,
  year(Sys.Date())
)
### as far as I know this function still does not exist natively in R
## I think it might, I feel like I saw that somewhere, or may be part of some common package like a tidyverse one or even data.table, but I don't remember and don't feel like looking it up and I've been using this forever now so, as long as it works, I guess
### anyway I might use this somewhere
## data.table has %notin% which seems to work same way like you'd expect it to
# `%nin%` = Negate(`%in%`)

##### Reading in Data #####
### Reading in team info saved in MCBB_VoAPrep.py
### taking out St Francis PA because they're D3 now and I forgot to do it in the python script and I don't feel like going back and re-running it
VoATeams_df <- fread(here(
  "Data",
  paste0("VoA", cbb_season - 1),
  paste0("MCBBVoA", cbb_season - 1, "Teams.csv")
))[school != "St. Francis (PA)"]

##### Loading Schedule and PBP Data #####
if (month(Sys.Date()) == 10) {
  ##### Preseason Data Load #####
  ### loading schedules to get neutral site info
  AllGames_PY2 <- as.data.table(load_mbb_schedule(cbb_season - 2))[, .(
    # fmt: skip
    home_team := fcase(
      home_location == "Maryland-Eastern Shore", "Maryland Eastern Shore",
      home_location == "Appalachian State", "App State",
      home_location == "Texas A&M-Commerce", "East Texas A&M",
      home_location == "IUPUI", "IU Indianapolis",
      default = home_location
    ),
    # fmt: skip
    away_team := fcase(
      away_location == "Maryland-Eastern Shore", "Maryland Eastern Shore",
      away_location == "Appalachian State", "App State",
      away_location == "Texas A&M-Commerce", "East Texas A&M",
      away_location == "IUPUI", "IU Indianapolis",
      default = away_location
    )
  )][
    home_team %in%
      VoATeams_df$school &
      away_team %in% VoATeams_df$school
  ]
  for (x in VoATeams_df$school) {
    if (x %notin% AllGames_PY2$home_team & x %notin% AllGames_PY2$away_team) {
      print(x)
    }
  }
  ### loading Previous season's schedule
  AllGames_PY1 <- as.data.table(load_mbb_schedule(cbb_season - 1))[, .(
    # fmt: skip
    home_team := fcase(
      home_location == "Maryland-Eastern Shore", "Maryland Eastern Shore",
      home_location == "Appalachian State", "App State",
      home_location == "Texas A&M-Commerce", "East Texas A&M",
      home_location == "IUPUI", "IU Indianapolis",
      default = home_location
    ),
    # fmt: skip
    away_team := fcase(
      away_location == "Maryland-Eastern Shore", "Maryland Eastern Shore",
      away_location == "Appalachian State", "App State",
      away_location == "Texas A&M-Commerce", "East Texas A&M",
      away_location == "IUPUI", "IU Indianapolis",
      default = away_location
    )
  )][
    home_team %in%
      VoATeams_df$school &
      away_team %in% VoATeams_df$school
  ]
  ### checking to make schools in df saved directly from CBBD API match team names used in hoopR games df
  for (x in VoATeams_df$school) {
    if (x %notin% AllGames_PY1$home_team & x %notin% AllGames_PY1$away_team) {
      print(x)
    }
  }
  ### loading PBP data
  PBP_PY2 <- as.data.table(load_mbb_pbp(cbb_season - 2))[, .(
    season,
    game_id,
    game_play_number,
    id,
    sequence_number,
    type_id,
    type_text,
    text,
    away_score,
    home_score,
    period_number,
    clock_display_value,
    scoring_play,
    score_value,
    team_id,
    shooting_play,
    home_team_id,
    home_team_name,
    home_team_abbrev,
    away_team_id,
    away_team_name,
    away_team_abbrev,
    game_spread,
    home_favorite,
    home_team_spread,
    half,
    time,
    clock_minutes,
    clock_seconds,
    home_timeout_called,
    away_timeout_called,
    start_period_seconds_remaining,
    end_period_seconds_remaining,
    start_game_seconds_remaining,
    end_game_seconds_remaining,
    game_date
  )][
    ### filtering game IDs so the PBP is only included for games where both teams are D1 teams
    game_id %in% AllGames_PY2$id
  ]
  ### PY1 PBP
  PBP_PY1 <- as.data.table(load_mbb_pbp(cbb_season - 1))[, .(
    season,
    game_id,
    game_play_number,
    id,
    sequence_number,
    type_id,
    type_text,
    text,
    away_score,
    home_score,
    period_number,
    clock_display_value,
    scoring_play,
    score_value,
    team_id,
    shooting_play,
    home_team_id,
    home_team_name,
    home_team_abbrev,
    away_team_id,
    away_team_name,
    away_team_abbrev,
    game_spread,
    home_favorite,
    home_team_spread,
    half,
    time,
    clock_minutes,
    clock_seconds,
    home_timeout_called,
    away_timeout_called,
    start_period_seconds_remaining,
    end_period_seconds_remaining,
    start_game_seconds_remaining,
    end_game_seconds_remaining,
    game_date
  )][
    ### filtering game IDs so the PBP is only included for games where both teams are D1 teams
    game_id %in% AllGames_PY1$id
  ]
} else {
  ##### November Data Load (PY1, current season) #####
  print("PY data should be saved already, only loading current season")
  ###
  AllGames <- as.data.table(load_mbb_schedule(cbb_season))[, `:=`(
    # fmt: skip
    home_team = fcase(
      home_location == "Maryland-Eastern Shore", "Maryland Eastern Shore",
      home_location == "Appalachian State", "App State",
      home_location == "Texas A&M-Commerce", "East Texas A&M",
      home_location == "IUPUI", "IU Indianapolis",
      default = home_location
    ),
    # fmt: skip
    away_team = fcase(
      away_location == "Maryland-Eastern Shore", "Maryland Eastern Shore",
      away_location == "Appalachian State", "App State",
      away_location == "Texas A&M-Commerce", "East Texas A&M",
      away_location == "IUPUI", "IU Indianapolis",
      default = away_location
    )
  )][
    ### filtering games so only games with teams in the VoA are included
    home_team %in%
      VoATeams_df$school &
      away_team %in% VoATeams_df$school &
      game_date < Sys.Date() &
      play_by_play_available == TRUE
  ]
  ### creating df of neutral site games
  NeutralSiteGames <- AllGames[neutral_site == TRUE]
  PBP <- as.data.table(hoopR::load_mbb_pbp(cbb_season))[, .(
    season,
    game_id,
    game_play_number,
    id,
    sequence_number,
    type_id,
    type_text,
    text,
    away_score,
    home_score,
    period_number,
    clock_display_value,
    scoring_play,
    score_value,
    team_id,
    shooting_play,
    home_team_id,
    home_team_name,
    home_team_abbrev,
    away_team_id,
    away_team_name,
    away_team_abbrev,
    game_spread,
    home_favorite,
    home_team_spread,
    half,
    time,
    clock_minutes,
    clock_seconds,
    home_timeout_called,
    away_timeout_called,
    start_period_seconds_remaining,
    end_period_seconds_remaining,
    start_game_seconds_remaining,
    end_game_seconds_remaining,
    game_date
  )][
    ### filtering game IDs so the PBP is only included for games where both teams are D1 teams
    game_id %in%
      AllGames$id &
      type_text %in%
        c(
          "DunkShot",
          "JumpShot",
          "Offensive Rebound",
          "Defensive Rebound",
          # "PersonalFoul",
          # "MadeFreeThrow",
          "Lost Ball Turnover",
          "Steal",
          "LayUpShot",
          "Block Shot",
          # "Dead Ball Rebound",
          # "ShortTimeout",
          "TipShot",
          # "RegularTimeOut",
          # "Technical Foul",
          "Shot"
        )
  ][, `:=`(
    ### holy god the way air formats this section is so fucking stupid
    ### apparently it formats fcase() "as a table"??????
    ## stupidest thing I've ever heard
    # fmt: skip
    pos_team = fctr(fcase(
      type_text %in% c("DunkShot", "JumpShot", "Offensive Rebound", "MadeFreeThrow", "Lost Ball Turnover", "LayUpShot", "TipShot", "Shot"), team_id,
      team_id == home_team_id, away_team_id,
      team_id == away_team_id, home_team_id,
      default = NA
    )),
    # fmt: skip
    def_team = fctr(fcase(
      type_text %notin% c("DunkShot", "JumpShot", "Offensive Rebound", "MadeFreeThrow", "Lost Ball Turnover", "LayUpShot", "TipShot", "Shot"), team_id,
      team_id == home_team_id, away_team_id,
      team_id == away_team_id, home_team_id,
      default = NA
    ))
  )][,
    ### adding column to represent homefield advantage during opponent adjustment
    # fmt: skip
    hfa := fctr(fcase(
      game_id %in% c(NeutralSiteGames$id), 0,
      pos_team == home_team_id, 1,
      def_team == home_team_id, -1,
      default = NA
    ))
  ]
  ### dropping NAs from these columns in PBP df
  PBP <- na.omit(PBP, cols = c("pos_team", "def_team", "hfa", "score_value"))
}

##### Extracting Opponent-adjusted stats using Ridge Regression #####
if (month(Sys.Date()) == 10) {
  print("add code to do preseason opponent-adjusted stuff")
} else if (month(Sys.Date()) == 11) {
  print("read in saved PY1 data here")
} else {
  ### creating opponent-adjusted stats here
  ## target columns used here: score_value
  ## may honestly just be points/possession to get target variable
  ## if I do anything else I'll save it for next season since the first VoA is going to go out basically right before conference tournaments start up in 2025/26 season
  ### creating dummy columns to use in ridge regression
  PPPAdj_dummycols <- dummy_cols(
    PBP[, c("pos_team", "def_team", "hfa")],
    remove_selected_columns = TRUE
  )
  score_values <- PBP$score_value
  ### removing objects from environment in the hope that it will ease memory issues
  rm(AllGames, NeutralSiteGames, PBP)
  ### identifying best lambda using cross validation
  set.seed(802)
  PPPAdj_cvglmnet <- cv.glmnet(
    x = as.matrix(PPPAdj_dummycols),
    y = score_values,
    alpha = 0
  )
  best_lambda <- PPPAdj_cvglmnet$lambda.min
  ### performing ridge regression using optimal lambda identified above
  set.seed(802)
  PPPAdj_glmnet <- glmnet(
    x = as.matrix(PPPAdj_dummycols),
    y = score_values,
    alpha = 0,
    lambda = best_lambda
  )

  ### extracting coefficients to adjust PPP with
  PPPAdj_glmnetcoef <- coef(PPPAdj_glmnet)
  PPPAdj_glmnetcoef_vals <- PPPAdj_glmnetcoef@x
  PPPAdj_adjcoefs <- data.frame(
    coef_name = colnames(PPPAdj_dummycols),
    ridge_reg_coef = PPPAdj_glmnetcoef_vals[
      2:length(PPPAdj_glmnetcoef_vals)
    ]
  )

  ### calculating adjusted coefficient
  PPPAdj_adjcoefs <- PPPAdj_adjcoefs |>
    mutate(adj_coef = ridge_reg_coef + PPPAdj_glmnetcoef_vals[1])

  ### strings used to help match up adjusted value with proper team below
  offstr = "pos_team"
  hfastr = "hfa"
  defstr = "def_team"
  stat = "score_value"

  ### calculating adjusted offensive PPP/play values, I think
  ## honestly I adapted all of this from Bud Davis's python code on the CFBD blog, I don't know what this does and I had to ask gemini to translate his python code to R and this is what it came up with and it seems to work, so I leave it as it is and pray to whichever deity is supposed to be running things around here that it doesn't break
  ## why does it create an index column only to immediately get rid of it, I don't know
  ## I've never seen colons before equal signs before, what the fuck is that
  PPPAdj_dfAdjOff <- PPPAdj_adjcoefs |>
    filter(str_sub(coef_name, 1, nchar(offstr)) == offstr) |>
    rename(!!stat := adj_coef) |>
    mutate(index = 1:n()) |>
    select(-index) |>
    mutate(coef_name = str_replace(coef_name, paste0("^", offstr, "_"), "")) |>
    select(-ridge_reg_coef)

  ### calculating adjusted defensive PPP/play values, I think
  PPPAdj_dfAdjdef <- PPPAdj_adjcoefs |>
    filter(str_sub(coef_name, 1, nchar(defstr)) == defstr) |>
    rename(!!stat := adj_coef) |>
    mutate(index = 1:n()) |>
    select(-index) |>
    mutate(coef_name = str_replace(coef_name, paste0("^", defstr, "_"), "")) |>
    select(-ridge_reg_coef)

  ### binding adjusted PPP values to main VoA_Variables df
  VoAVariables <- VoATeams_df |>
    left_join(PPPAdj_dfAdjOff |> rename(school = coef_name), by = "school") |>
    rename(adj_off_PPP = score_value) |>
    left_join(PPPAdj_dfAdjdef |> rename(school = coef_name), by = "school") |>
    rename(adj_def_PPP = score_value)
}


##### Poopypants Testing #####
### everything in this section is just scratch work
poopypants <- PBP[type_text == "Lost Ball Turnover"]

VoAVariables |>
  arrange(desc(adj_off_PPP - adj_def_PPP)) |>
  select(school, adj_off_PPP, adj_def_PPP) |>
  gt()
