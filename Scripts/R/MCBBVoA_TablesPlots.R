##### MCBB D1 Vortex of Accuracy Graphics Output #####
### gt isn't working in python so I'm taking the output csv, reading it in here, and making all the gt tables and plots that I want with it here in R since I've almost always gotten gt to work in R
##### loading packages, reading in data #####
library(pacman)
p_load(tidyverse, gt, gtExtras, here, webshot2, data.table, RColorBrewer)

### reading in VoA csv
VoAVariables <- read_csv(here("Data", "VoA2025", "MCBBVoA2025VoA1.csv")) |>
  arrange(desc(OvrlVoA_MeanRating))

### selecting only ratings and rankings columns
VoAVariables_gt <- VoAVariables |>
  select(
    team,
    OvrlVoA_MeanRating,
    OvrlVoARanking,
    OffVoA_MeanRating,
    OffVoARanking,
    DefVoA_MeanRating,
    DefVoARanking
  )

VoAVariables_Top25gt <- VoAVariables_gt |>
  filter(OvrlVoARanking <= 25)

##### Making gt table #####
VoATop25Table_gt <- VoAVariables_Top25gt |>
  gt() |> # use 'gt' to make an awesome table...
  gt_theme_espn() |>
  tab_header(
    title = "MCBB D1 Vortex of Accuracy", # ...with this title
    subtitle = "Supremely Excellent Yet Salaciously Godlike And Infallibly Magnificent Vortex of Accuracy"
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
  cols_label(
    OvrlVoA_MeanRating = "Overall VoA Rating",
    OvrlVoARanking = "VoA Ranking",
    OffVoA_MeanRating = "Off VoA Rating",
    OffVoARanking = "Off Ranking",
    DefVoA_MeanRating = "Def VoA Rating",
    DefVoARanking = "Def Ranking",
  ) |> # Update labels
  # cols_move_to_end(columns = "VoA_Rating") |>
  # cols_hide(c(week, VoA_Output)) |>
  tab_footnote(
    footnote = "Table by @gshelor, Data from CBBD API"
  )

### displaying and saving table
VoATop25Table_gt
VoATop25Table_gt |>
  gtsave(
    "MCBBVoA2025VoA1Top25.png",
    expand = 5,
    path = here("Outputs", "VoA2025")
  )
