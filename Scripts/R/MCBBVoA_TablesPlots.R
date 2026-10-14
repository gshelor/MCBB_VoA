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
voa_num <- readline(prompt = "Which iteration of the VoA is this? ")

### reading in VoA csv
VoAStats <- read_parquet(here(
  "Data",
  paste0("VoA", cbb_season),
  paste0("MCBBVoA", cbb_season, "VoA", voa_num, ".parquet")
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
    paste0("MCBBVoA", cbb_season, "VoA", voa_num, "Top25.png"),
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
    paste0("MCBBVoA", cbb_season, "VoA", voa_num, ".png"),
    expand = 5,
    path = here("Outputs", paste0("VoA", cbb_season))
  )


##### Tracking Ratings Charts for Each Conference #####

if (as.integer(voa_num) >= 4) {
  ### reading in all the VoAs and binding them together to make plots tracking ratings by conference
  for (i in 1:as.integer(voa_num)) {
    if (as.integer(voa_num) == 0) {
      ### reading in the first VoA, to be row-bound to the other VoA releases during the rest of the for loop
      VoARatingsRanks <- read_parquet(here(
        "Data",
        paste0("VoA", cbb_season),
        paste0("MCBBVoA", cbb_season, "VoA", voa_num, ".parquet")
      )) |>
        arrange(desc(OvrlVoA_MeanRating)) |>
        select(
          voa_iter,
          team,
          conference,
          logo_0,
          OvrlVoA_MeanRating,
          OvrlVoARanking
        )
    } else {
      ### reading in later VoA after the first
      temp_voa <- read_parquet(here(
        "Data",
        paste0("VoA", cbb_season),
        paste0("MCBBVoA", cbb_season, "VoA", voa_num, ".parquet")
      )) |>
        arrange(desc(OvrlVoA_MeanRating)) |>
        select(
          voa_iter,
          team,
          conference,
          logo_0,
          OvrlVoA_MeanRating,
          OvrlVoARanking
        )

      ### binding more recent VoAs to get all VoA releases in 1 df for plotting
      VoARatingsRanks <- rbind(VoARatingsRanks, temp_voa)
    }
  }

  ### making charts, ratings and ranks charts for each conference
  for (i in unique(VoARatingsRanks$conference)) {
    ### shortening conference names so they don't have spaces
    conference_filename <- str_replace_all(i, pattern = " ", replacement = "")
    ### filtering VoA Ratings and Ranks by conference
    ConferenceVoA <- VoARatingsRanks |>
      filter(conference == i)

    ### charting VoA_Rating and VoA_Ranking for each week from week 2 on
    VoA_Rating_Chart <- ggplot(
      ConferenceVoA,
      aes(x = voa_iter, y = OvrlVoA_MeanRating, group = team)
    ) +
      theme_bw() +
      geom_line(linewidth = 1.5) +
      geom_image(aes(image = logo_0), size = 0.055) +
      # geom_point(size = 5) +
      xlab("VoA Release") +
      ylab("VoA Overall Rating") +
      labs(
        caption = "chart by @gshelor, data from collegebasketballdata.com API via cfbfastR"
      ) +
      ggtitle(paste(
        conference,
        "Vortex of Accuracy Overall Ratings by VoA Release"
      )) +
      expand_limits(
        y = c(
          floor(floor(min(ConferenceVoA$OvrlVoA_MeanRating)) / 10) * 10,
          ceiling((ceiling(max(ConferenceVoA$OvrlVoA_MeanRating)) / 10)) * 10
        )
      ) +
      scale_y_continuous(
        breaks = seq(
          (floor((floor(min(ConferenceVoA$OvrlVoA_MeanRating)) / 10)) * 10),
          (ceiling((ceiling(max(ConferenceVoA$OvrlVoA_MeanRating)) / 10)) * 10),
          by = 5
        )
      ) +
      scale_x_continuous(
        breaks = c(0:as.integer(voa_num))
      ) +
      # geom_cfb_logos(aes(team = school, width = 0.035)) +
      theme(
        plot.title = element_text(size = 35, hjust = 0.5),
        axis.text.x = element_text(size = 20),
        axis.text.y = element_text(size = 20),
        axis.title.x = element_text(size = 22),
        axis.title.y = element_text(size = 22),
        legend.text = element_text(size = 20)
      )
    VoA_Rating_Chart
    ggsave(
      paste0("VoARatings", conference_filename, ".png"),
      path = here("Outputs", paste0("VoA", cbb_season)),
      width = 50,
      height = 40,
      units = 'cm'
    )

    VoA_Ranking_Chart <- ggplot(
      ConferenceVoA,
      aes(x = voa_iter, y = OvrlVoARanking, group = team)
    ) +
      theme_bw() +
      geom_line(linewidth = 1.5) +
      geom_image(aes(image = logo_0), size = 0.055) +
      # geom_point(size = 5) +
      xlab("VoA Release") +
      ylab("VoA Ranking") +
      labs(
        caption = "chart by @gshelor, data from collegebasketballdata.com API via cbbd python library"
      ) +
      ggtitle("UAC Vortex of Accuracy Rankings by Week") +
      expand_limits(y = c(0, 365)) +
      scale_y_continuous(breaks = c(0, 50, 100, 150, 200, 250, 300, 350, 400)) +
      scale_y_reverse() +
      scale_x_continuous(
        breaks = c(0:as.integer(voa_num))
      ) +
      # geom_cfb_logos(aes(team = school, width = 0.035)) +
      theme(
        plot.title = element_text(size = 35, hjust = 0.5),
        axis.text.x = element_text(size = 20),
        axis.text.y = element_text(size = 20),
        axis.title.x = element_text(size = 22),
        axis.title.y = element_text(size = 22),
        legend.text = element_text(size = 20)
      )
    VoA_Ranking_Chart
    ggsave(
      paste0("VoARankings", conference_filename, ".png"),
      path = here("Outputs", paste0("VoA", cbb_season)),
      width = 50,
      height = 40,
      units = 'cm'
    )
  }
}
