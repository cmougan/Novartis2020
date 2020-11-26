

library(dplyr)
library(tidyr)


gx_n_generics <- read.csv("data/gx_num_generics.csv") %>% select(-X) %>% as_tibble()
gx_package <- read.csv("data/gx_package.csv") %>% select(-X) %>% as_tibble()
gx_panel <- read.csv("data/gx_panel.csv") %>% select(-X) %>% as_tibble()
gx_volume <- read.csv("data/gx_volume.csv") %>% select(-X) %>% as_tibble()
gx_ther_area <- read.csv("data/gx_therapeutic_area.csv") %>% select(-X) %>% as_tibble()
submission <- read.csv("data/submission_template.csv") %>% as_tibble()


max_months <- gx_volume %>% 
  group_by(country, brand) %>% 
  summarise(
    max_month = max(month_num)
  ) %>% ungroup()

max_months %>% count(max_month, sort = T)
max_months %>% count(max_month <= -1)

max_months %>% filter(max_month < 23, max_month >= 0)
gx_volume %>% filter(country == "country_1", brand == "brand_121") %>% tail()
submission %>% filter(country == "country_1", brand == "brand_121")

# 191 test data
submission %>% count(country, brand) %>% dim


target <- expand.grid(
  unique(gx_volume$country),
  unique(gx_volume$brand),
  0:23
)

submission
target

