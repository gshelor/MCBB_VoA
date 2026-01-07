##### Men's College Basketball Vortex of Accuracy #####

##### Loading Packages #####
library(pacman)
p_load(hoopR, tidyverse, gtExtras, ggimage, cbbdata, fastDummies)

##### Reading in Data #####
`%nin%` = Negate(`%in%`)
cbbdata::cbd_login()
D1Teams <- cbbdata::cbd_teams() |>
  filter(espn_display != "Hartford Hawks") |>
  filter(espn_short_display != "St Francis BK")
### App State, IUPUI, Maryland-Eastern Shore, A&M Commerce need to be renamed
ESPNTeams <- hoopR::espn_mbb_teams(2025)
### lindenwood (new D1 team) isn't in the espn_mbb_teams thing
## neither is Queens, Southern Indiana

PBP <- hoopR::load_mbb_pbp(2025) |>
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
test_PBP <- PBP |>
  filter(home_team_name == "IU Indianapolis")
### testing out dummy variables
poopypants_dummies <- dummy_cols(PBP[, c()])
poopypants3 <- hoopR::load_mbb_team_box(2025)
AllGames <- hoopR::load_mbb_schedule(2025)
CompletedGames <- AllGames |>
  filter(status_type_completed == TRUE)
# D1Teams <- hoopR::ncaa_mbb_teams(year = 2025, division = 1)

poopypants5_b <- D1Teams |>
  mutate(testcol = 1:362, test2col = (seq(1, 362)^2) / 2) |>
  filter(conference_short_name == "Mountain West")


ggplot(poopypants5_b, aes(x = testcol, y = test2col)) +
  geom_image(aes(image = logo), size = 0.1) +
  theme_bw()

poopypants5_c <- poopypants5_b |>
  select(logo, testcol, test2col)


# Create the gt table with team logos
poopypants5_c |>
  gt() |>
  gt_img_rows(columns = logo, img_source = "web", height = 30)
