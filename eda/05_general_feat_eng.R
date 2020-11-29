
library(ggplot2)
library(dplyr)
library(slider)
library(ggridges)


source("tools/feat_eng.R")

gx_merged_raw <- read.csv("data/gx_merged_lags_months.csv") %>% as_tibble()
gx_volume <- read.csv("data/gx_volume.csv") %>% select(-X) %>% as_tibble()

volume_summaries <- gx_volume %>% 
  group_by(country, brand) %>% 
  summarise(
    max_vol = max(volume[month_num < 0]),
    min_vol = min(volume[month_num < 0]),
    min_month = min(month_num),
    last_vol = last(volume[month_num < 0]),
    last_vol_6 = last(volume[month_num <= -6]),
    last_vol_12 = last(volume[month_num <= -12]),
    # vol_12th = nth(volume, 12),
    # vol_6th = nth(volume, 12),
    ratios_vol_12 = last_vol_12 / last_vol,
    ratios_vol_6 = last_vol_6 / last_vol,
    # ratios_vol_12_ini = vol_12th / last_vol,
    # ratios_vol_6_ini = vol_6th / last_vol,
    max_last_diff = (max_vol - last_vol) / max_vol,
    min_last_diff = (min_vol - last_vol) / min_vol,
  ) %>% 
  select(-last_vol)

summary(volume_summaries)


write.csv(volume_summaries, file = "data/volume_features.csv", row.names = FALSE)


