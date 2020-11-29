
library(ggplot2)
library(dplyr)
library(slider)
library(ggridges)

theme_set(theme_minimal())

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
    max_last_diff = (max_vol - last_vol) / max_vol,
    min_last_diff = (min_vol - last_vol) / min_vol,
  ) %>% 
  select(-last_vol)

gx_full <- gx_merged_raw %>% 
  left_join(volume_summaries) %>% 
  filter(test == 0) %>% 
  ungroup

gx_merged_raw %>% View

gx_full$target <- (gx_full$volume - gx_full$last_before_12_after_0) / gx_full$last_before_12_after_0

gx_full %>% 
  select_if(is.numeric) %>% 
  as.matrix() %>% 
  cor


ggplot(gx_full, aes(x = log(max_vol), y = target)) +
  geom_point() + 
  geom_smooth()

ggplot(gx_full, aes(x = log(min_vol), y = target)) +
  geom_point() + 
  geom_smooth()


ggplot(gx_full, aes(x = log(max_vol / min_vol), y = target)) +
  geom_point() + 
  geom_smooth()


gx_full %>% 
  mutate(
    channel = case_when(
      D > 90 ~ "Channel D",
      C > 90 ~ "Channel C",
      B > 90 ~ "Channel B",
      T ~ "Mixed"
    )
  ) %>% 
  ggplot() + 
  geom_density_ridges2(aes(y = channel, x = target), quantile_lines = T)


gx_full %>% 
  group_by(country) %>% 
  summarise(
    mean(target)
  )

gx_full %>% 
  ggplot(aes(x = country, y = target)) + 
  geom_boxplot() + 
  coord_flip()

gx_full %>% 
  mutate(
    num_generics = case_when(
      num_generics < 5 ~ "1. Less than 5",
      num_generics < 10 ~ "2. Less than 10, more than 5",
      num_generics < 20 ~ "3. Less than 20, more than 10",
      T ~ "4. More than 20",
    )
  ) %>% 
ggplot() + 
  geom_density_ridges2(
    aes(y = num_generics, x = target),
    quantile_lines = T
    )


# gx_full %>% View
gx_full %>% 
  ggplot(aes(x = therapeutic_area, y = target)) + 
  geom_boxplot() + 
  coord_flip()


gx_full %>% 
  ggplot(aes(y = therapeutic_area, x = target)) + 
  geom_density_ridges2()


gx_full %>% 
  ggplot(aes(x = presentation, y = target)) + 
  geom_boxplot() + 
  coord_flip()


gx_full %>% 
  ggplot(aes(x = min_month, y = target)) + 
  geom_smooth() + 
  geom_point(alpha = 0.1)


gx_full %>% 
  ggplot(aes(x = log(median_before_3_after_0), y = target)) + 
  geom_smooth() + 
  geom_point(alpha = 0.1)


gx_full %>% 
  group_by(country, month_num) %>% 
  summarise(
    target_50 = median(target),
    target_avg = mean(target),
    target_25 = quantile(target, .25),
    target_75 = quantile(target, .75),
    n = n()
  ) %>% 
  filter(n > 10) %>% 
  ggplot(aes(x = month_num, y = target_avg)) + 
  geom_line() + 
  geom_ribbon(aes(ymin = target_25, ymax = target_75), alpha = 0.1) +
  facet_wrap(~country)



gx_full %>% 
  group_by(brand, month_num) %>% 
  summarise(
    target_50 = median(target),
    target_avg = mean(target),
    target_25 = quantile(target, .25),
    target_75 = quantile(target, .75),
    n = n()
  ) %>% 
  ungroup() %>% 
  filter(n > 5) %>%
  # filter(brand %in% c("brand_1", "brand_2")) %>% 
  ggplot(aes(x = month_num, y = target_avg)) + 
  geom_line() + 
  geom_ribbon(aes(ymin = target_25, ymax = target_75), alpha = 0.1) +
  facet_wrap(~brand)


gx_full %>% 
  group_by(therapeutic_area, month_num) %>% 
  summarise(
    target_50 = median(target),
    target_avg = mean(target),
    target_25 = quantile(target, .25),
    target_75 = quantile(target, .75),
    n = n()
  ) %>% 
  filter(n > 10) %>% 
  ggplot(aes(x = month_num, y = target_avg)) + 
  geom_line() + 
  geom_ribbon(aes(ymin = target_25, ymax = target_75), alpha = 0.1) +
  facet_wrap(~therapeutic_area)


gx_full %>% 
  group_by(presentation, month_num) %>% 
  summarise(
    target_50 = median(target),
    target_avg = mean(target),
    target_25 = quantile(target, .25),
    target_75 = quantile(target, .75),
    n = n()
  ) %>% 
  filter(n > 10) %>% 
  ggplot(aes(x = month_num, y = target_avg)) + 
  geom_line() + 
  geom_ribbon(aes(ymin = target_25, ymax = target_75), alpha = 0.1) +
  facet_wrap(~presentation)


gx_full %>% 
  mutate(
    num_generics = case_when(
      num_generics <= 1 ~ "0. Only 1",
      num_generics < 5 ~ "1. Less than 5, more than 1",
      num_generics < 10 ~ "2. Less than 10, more than 5",
      num_generics < 20 ~ "3. Less than 20, more than 10",
      T ~ "4. More than 20",
    )
  ) %>% 
  group_by(num_generics, month_num) %>% 
  summarise(
    target_50 = median(target),
    target_avg = mean(target),
    target_25 = quantile(target, .25),
    target_75 = quantile(target, .75),
    n = n()
  ) %>% 
  filter(n > 10) %>% 
  ggplot(aes(x = month_num, y = target_avg)) + 
  geom_line() + 
  geom_ribbon(aes(ymin = target_25, ymax = target_75), alpha = 0.1) +
  facet_wrap(~num_generics)



gx_full %>% 
  group_by(month_name, month_num) %>% 
  summarise(
    target_50 = median(target),
    target_avg = mean(target),
    target_25 = quantile(target, .25),
    target_75 = quantile(target, .75),
    n = n()
  ) %>% 
  filter(n > 10) %>% 
  ggplot(aes(x = month_num, y = target_avg)) + 
  geom_line() + 
  geom_ribbon(aes(ymin = target_25, ymax = target_75), alpha = 0.1) +
  facet_wrap(~month_name)
