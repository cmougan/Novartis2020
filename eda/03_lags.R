
library(dplyr)
library(slider)

source("tools/feat_eng.R")

gx_merged_raw <- read.csv("data/gx_merged.csv") %>% as_tibble()
gx_volume <- read.csv("data/gx_volume.csv") %>% select(-X) %>% as_tibble()


View(gx_merged_raw)

gx_volume_brand_country <- gx_volume %>% 
  add_rolling_stats(24, 0) %>% 
  add_rolling_stats(12, 0) %>% 
  add_rolling_stats(Inf, 0) %>% 
  add_rolling_stats(3, 0)

gx_volume_brand_country <- gx_volume_brand_country %>% 
  filter(month_num == -1) %>% 
  select(-month_name, -month_num, -volume)


gx_volume_country <- gx_volume %>% 
  group_by(country, month_num) %>%
  summarise(volume = sum(volume)) %>% 
  ungroup() %>% 
  add_rolling_stats_country(24, 0) %>% 
  add_rolling_stats_country(12, 0) %>% 
  add_rolling_stats_country(Inf, 0) %>% 
  add_rolling_stats_country(3, 0)

gx_volume_country <- gx_volume_country %>% 
  filter(month_num == -1) %>% 
  select(-month_num, -volume)


gx_volume_brand <- gx_volume %>% 
  group_by(brand, month_num) %>%
  summarise(volume = sum(volume)) %>% 
  ungroup() %>% 
  add_rolling_stats_brand(24, 0) %>% 
  add_rolling_stats_brand(12, 0) %>% 
  add_rolling_stats_brand(Inf, 0) %>% 
  add_rolling_stats_brand(3, 0)

gx_volume_brand <- gx_volume_brand %>% 
  filter(month_num == -1) %>% 
  select(-month_num, -volume)

# gx_volume %>% View
gx_volume_brand %>% View

gx_merged <- gx_merged_raw %>% 
  left_join(gx_volume_brand_country, by = c("country", "brand")) # %>% 
gx_merged <- gx_merged %>% left_join(gx_volume_brand, by = c("brand"))
gx_merged <- gx_merged %>% left_join(gx_volume_country, by = c("country")) 

# gx_merged
# View(gx_volume_brand)
# View(gx_volume_country)

write.csv(gx_merged, file = "data/gx_merged_lags.csv", row.names = F)

gx_merged %>% 
  group_by(test) %>% 
  count(therapeutic_area) %>% 
  mutate(n / sum(n)) %>% 
  arrange(therapeutic_area, test) %>% View


gx_merged_raw %>% 
  group_by(test) %>% 
  count(brand) %>% 
  mutate(n / sum(n)) %>% 
  arrange(brand, test) %>% View
