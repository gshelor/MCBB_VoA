##### Men's College Basketball Vortex of Accuracy #####
### no longer the operational script what with the creation of the cbbd data api
### may be turned into script for producing plots and/or gt tables
### may also try to use hoopR for play by play data? CBBD API is disappointingly poor with providing that because the only way to access pbp data is to get it for individual dates or for specific games and that's just so incredibly inefficient for the purposes of a VoA
## which means that if I tried to get it that way for the VoA that I'd be using lots of API requests too and the CBBD API free tier is generous but not that generous
### hoopR seems to be good about getting PBP data?
##### Loading Packages #####
library(pacman)
p_load(
  hoopR,
  tidyverse,
  gtExtras,
  ggimage,
  # cbbdata,
  fastDummies,
  here,
  cfbfastR,
  glmnet,
  data.table
)

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
`%nin%` = Negate(`%in%`)

##### Reading in Data #####
### Reading in team info saved in MCBB_VoAPrep.py
VoATeams_df <- fread(here(
  "Data",
  paste0("VoA", cbb_season - 1),
  paste0("MCBBVoA", cbb_season - 1, "Teams.csv")
))
##### Loading Schedule and PBP Data #####
if (month(Sys.Date()) == 10) {
  ##### Preseason Data Load #####
  ### loading schedules to get neutral site info
  AllGames_PY2 <- as.data.table(load_mbb_schedule(cbb_season - 2))[
    home_location %in%
      VoATeams_df$school &
      away_location %in% VoATeams_df$school
  ]
  AllGames_PY1 <- as.data.table(load_mbb_schedule(cbb_season - 1))
  ### loading PBP data
  PBP_PY2 <- as.data.table(hoopR::load_mbb_pbp(cbb_season - 2)) |>
    select(
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
    )
  ### PY1 PBP
  PBP_PY1 <- as.data.table(hoopR::load_mbb_pbp(cbb_season - 1)) |>
    select(
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
    )
} else {
  ##### November Data Load (PY1, current season) #####
  print("PY data should be saved already, only loading current season")
  ###
  AllGames <- as.data.table(load_mbb_schedule(cbb_season))
  PBP <- as.data.table(load_mbb_pbp(cbb_season)) |>
    select(
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
    )
}
