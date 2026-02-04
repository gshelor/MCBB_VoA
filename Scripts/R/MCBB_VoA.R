##### Men's College Basketball Vortex of Accuracy #####
### no longer the operational script what with the creation of the cbbd data api
### may be turned into script for producing plots and/or gt tables
### may also try to use hoopR for play by play data? CBBD API is disappointingly poor with providing that because the only way to access pbp data is to get it for individual dates or for specific games and that's just so incredibly inefficient for the purposes of a VoA
## which means that if I tried to get it that way for the VoA that I'd be using lots of API requests too and the CBBD API free tier is generous but not that generous
### hoopR seems to be good about getting PBP data?
##### Loading Packages #####
library(pacman)
p_load(hoopR, tidyverse, gtExtras, ggimage, cbbdata, fastDummies, here)

##### Reading in Data #####
`%nin%` = Negate(`%in%`)
### App State, IUPUI, Maryland-Eastern Shore, A&M Commerce need to be renamed
# ESPNTeams <- hoopR::espn_mbb_teams(2026)
### lindenwood (new D1 team) isn't in the espn_mbb_teams thing
## neither is Queens, Southern Indiana

PBP <- hoopR::load_mbb_pbp(2026) |>
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
poopypants <- read_csv(here("poopypants.csv"))
for (x in poopypants$team) {
  if (x %nin% PBP$away_team_name) {
    print(x)
  }
}

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
