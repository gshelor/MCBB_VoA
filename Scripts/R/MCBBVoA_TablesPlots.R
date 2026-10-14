##### MCBB D1 Vortex of Accuracy Graphics Output #####
### gt isn't working in python so I'm taking the output csv, reading it in here, and making all the gt tables and plots that I want with it here in R since I've almost always gotten gt to work in R
##### loading packages, reading in data #####
library(pacman)
p_load(tidyverse, gt, gtExtras, here, webshot2, data.table, RColorBrewer, arrow)

year <- as.integer(format(Sys.Date(), "%Y"))
month <- as.integer(format(Sys.Date(), "%m"))
if (month >= 10) {
  cbb_season <- year
  cbb_season_str <- paste0(cbb_season, "/", cbb_season + 1)
} else {
  cbb_season <- year - 1
  cbb_season_str <- paste0(cbb_season, "/", year)
}


### getting VoA iteration (how many times the VoA has been run this season)
voa_iter <- readline(prompt = "Which iteration of the VoA is this? ")

### reading in VoA csv
VoAStats <- read_parquet(here(
  "Data",
  paste0("VoA", cbb_season),
  paste0("MCBBVoA", cbb_season, "VoA", voa_iter, ".parquet")
)) |>
  arrange(desc(OvrlVoA_MeanRating))

### selecting only ratings and rankings columns
VoAStats_gt <- VoAStats |>
  select(
    logo_0,
    team,
    OvrlVoA_MeanRating,
    OvrlVoARanking,
    OffVoA_MeanRating,
    OffVoARanking,
    DefVoA_MeanRating,
    DefVoARanking
  )

VoAStats_Top25gt <- VoAStats_gt |>
  filter(OvrlVoARanking <= 25)

##### Making gt tables #####
### Top 25 Table
VoATop25Table_gt <- VoAStats_Top25gt |>
  gt() |> # use 'gt' to make an awesome table...
  gt_theme_espn() |>
  tab_header(
    title = paste(cbb_season_str, "MCBB D1 Vortex of Accuracy Top 25"), # ...with this title
    subtitle = paste(
      "Supremely Excellent Yet Salaciously Godlike And Infallibly Magnificent Vortex of Accuracy,",
      Sys.Date()
    )
  ) |> # and this subtitle
  ##tab_style(style = cell_fill("bisque"),
  ## locations = cells_body()) |> # add fill color to table
  fmt_number(
    # A column (numeric data)
    columns = c(OvrlVoA_MeanRating, OffVoA_MeanRating, DefVoA_MeanRating), # What column variable? FinalVoATop25$VoA_Rating
    decimals = 3 # With four decimal places
  ) |>
  fmt_number(
    # Another column (also numeric data)
    columns = c(OvrlVoARanking, OffVoARanking, DefVoARanking), # What column variable? FinalVoATop25$VoA_Ranking
    decimals = 0 # I want this column to have zero decimal places
  ) |>
  data_color(
    # Update cell colors, testing different color palettes
    columns = c(OvrlVoA_MeanRating, OffVoA_MeanRating), # ...for dose column
    fn = scales::col_numeric(
      # <- bc it's numeric
      palette = brewer.pal(11, "RdYlGn"), # A color scheme (gradient)
      domain = c(), # Column scale endpoints
      reverse = FALSE
    )
  ) |>
  data_color(
    # Update cell colors, testing different color palettes
    columns = c(DefVoA_MeanRating), # ...for dose column
    fn = scales::col_numeric(
      # <- bc it's numeric
      palette = brewer.pal(11, "RdYlGn"), # A color scheme (gradient)
      domain = c(), # Column scale endpoints
      reverse = TRUE
    )
  ) |>
  gt_img_rows(columns = logo_0) |>
  cols_label(
    logo_0 = "",
    team = "Team",
    OvrlVoA_MeanRating = "Overall VoA Rating",
    OvrlVoARanking = "VoA Ranking",
    OffVoA_MeanRating = "Off VoA Rating",
    OffVoARanking = "Off Ranking",
    DefVoA_MeanRating = "Def VoA Rating",
    DefVoARanking = "Def Ranking",
  ) |> # Update labels
  # cols_move_to_end(columns = "VoA_Rating") |>
  # cols_hide(c(week, VoA_Output)) |>
  # cols_merge(columns = c(logo_0, "team")) |>
  tab_footnote(
    footnote = "Table by @gshelor, Data from CBBD API"
  )

### displaying and saving table
VoATop25Table_gt
VoATop25Table_gt |>
  gtsave(
    paste0("MCBBVoA", cbb_season, "VoA", voa_iter, "Top25.png"),
    expand = 5,
    path = here("Outputs", paste0("VoA", cbb_season))
  )

### Full Table
VoATable_gt <- VoAStats_gt |>
  gt() |> # use 'gt' to make an awesome table...
  gt_theme_espn() |>
  tab_header(
    title = paste(cbb_season_str, "MCBB D1 Vortex of Accuracy"), # ...with this title
    subtitle = paste(
      "Supremely Excellent Yet Salaciously Godlike And Infallibly Magnificent Vortex of Accuracy,",
      Sys.Date()
    )
  ) |> # and this subtitle
  ##tab_style(style = cell_fill("bisque"),
  ## locations = cells_body()) |> # add fill color to table
  fmt_number(
    # A column (numeric data)
    columns = c(OvrlVoA_MeanRating, OffVoA_MeanRating, DefVoA_MeanRating), # What column variable? FinalVoA$VoA_Rating
    decimals = 3 # With four decimal places
  ) |>
  fmt_number(
    # Another column (also numeric data)
    columns = c(OvrlVoARanking, OffVoARanking, DefVoARanking), # What column variable? FinalVoA$VoA_Ranking
    decimals = 0 # I want this column to have zero decimal places
  ) |>
  data_color(
    # Update cell colors, testing different color palettes
    columns = c(OvrlVoA_MeanRating, OffVoA_MeanRating), # ...for dose column
    fn = scales::col_numeric(
      # <- bc it's numeric
      palette = brewer.pal(11, "RdYlGn"), # A color scheme (gradient)
      domain = c(), # Column scale endpoints
      reverse = FALSE
    )
  ) |>
  data_color(
    # Update cell colors, testing different color palettes
    columns = c(DefVoA_MeanRating), # ...for dose column
    fn = scales::col_numeric(
      # <- bc it's numeric
      palette = brewer.pal(11, "RdYlGn"), # A color scheme (gradient)
      domain = c(), # Column scale endpoints
      reverse = TRUE
    )
  ) |>
  gt_img_rows(columns = logo_0) |>
  cols_label(
    logo_0 = "",
    team = "Team",
    OvrlVoA_MeanRating = "Overall VoA Rating",
    OvrlVoARanking = "VoA Ranking",
    OffVoA_MeanRating = "Off VoA Rating",
    OffVoARanking = "Off Ranking",
    DefVoA_MeanRating = "Def VoA Rating",
    DefVoARanking = "Def Ranking",
  ) |> # Update labels
  # cols_move_to_end(columns = "VoA_Rating") |>
  # cols_hide(c(week, VoA_Output)) |>
  # cols_merge(columns = c(logo_0, "team")) |>
  tab_footnote(
    footnote = "Table by @gshelor, Data from CBBD API"
  )

### displaying and saving table
VoATable_gt
VoATable_gt |>
  gtsave(
    paste0("MCBBVoA", cbb_season, "VoA", voa_iter, ".png"),
    expand = 5,
    path = here("Outputs", paste0("VoA", cbb_season))
  )
