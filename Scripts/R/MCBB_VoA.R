library(pacman)
p_load(hoopR, tidyverse, gtExtras, ggimage, cbbdata)

poopypants2 <- hoopR::load_mbb_pbp(2025)
poopypants3 <- hoopR::load_mbb_team_box(2025)
poopypants4 <- hoopR::load_mbb_schedule(2025)
poopypants5 <- hoopR::espn_mbb_teams(2025)
cbbdata::cbd_login()


poopypants5_b <- poopypants5 |>
  mutate(testcol = 1:361,
         test2col = (seq(1,361)^2)/2) |>
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
