
library(ggplot2)
library(dplyr)
library(slider)
library(ggridges)

theme_set(theme_minimal())

source("tools/feat_eng.R")

gx_merged_raw <- read.csv("data/gx_raw.csv") %>% as_tibble()
gx_volume <- read.csv("data/gx_volume.csv") %>% select(-X) %>% as_tibble()

train_ids <- read.csv("data/train_split.csv")
train_ids$train <- 1

gx_merged_raw <- gx_merged_raw %>% left_join(train_ids)
gx_merged_raw$train <- coalesce(gx_merged_raw$train, 0)

volume_summaries <- gx_volume %>% 
  group_by(country, brand) %>% 
  summarise(
    max_vol = max(volume[month_num < 0]),
    min_vol = min(volume[month_num < 0]),
    min_month = min(month_num),
    last_vol = last(volume[month_num < 0]),
    max_last_diff = (max_vol - last_vol) / max_vol,
    min_last_diff = (min_vol - last_vol) / min_vol,
  )

gx_full <- gx_merged_raw %>% 
  left_join(volume_summaries) %>% 
  filter(test == 0) %>% 
  ungroup

gx_full$target <- (gx_full$volume - gx_full$last_vol) / gx_full$last_vol


# Plot target -------------------------------------------------------------

gx_full %>% 
  filter(month_num >= 0) %>% 
  mutate(train = as.factor(train)) %>% 
  group_by(month_num, train) %>% 
  summarise(
    mean_vol = mean(volume)
  ) %>% 
  ggplot(aes(x = month_num, y = mean_vol, color = train, group = train)) +
  geom_line()


gx_full %>% 
  filter(month_num >= 0) %>% 
  mutate(train = as.factor(train)) %>% 
  group_by(month_num, train) %>% 
  summarise(
    mean_vol = mean(target)
  ) %>% 
  ggplot(aes(x = month_num, y = mean_vol, color = train, group = train)) +
  geom_line()


# Time evolution ----------------------------------------------------------


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



# gx_full %>% 
#   group_by(brand, month_num) %>% 
#   summarise(
#     target_50 = median(target),
#     target_avg = mean(target),
#     target_25 = quantile(target, .25),
#     target_75 = quantile(target, .75),
#     n = n()
#   ) %>% 
#   ungroup() %>% 
#   filter(n > 5) %>%
#   filter(brand %in% c("brand_1", "brand_2")) %>%
#   ggplot(aes(x = month_num, y = target_avg)) + 
#   geom_line() + 
#   geom_ribbon(aes(ymin = target_25, ymax = target_75), alpha = 0.1) +
#   facet_wrap(~brand)


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


target_23 <- gx_full %>% 
  filter(month_num == 23) %>% 
  select(country, brand, target)

target_0 <- gx_full %>% 
  group_by(country, brand) %>%
  summarise(target = first(target)) %>% 
  ungroup()

target_0 <- gx_full %>% 
  filter(month_num == -12) %>%
  select(country, brand, target)




target_0 %>% 
  left_join(target_23, by = c("country", "brand"), suffix = c("_0", "_23")) %>% 
  ggplot(aes(x = target_0, y = target_23)) + 
  geom_point()



# Many lines together -----------------------------------------------------

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
  ggplot(aes(x = month_num, y = target_avg, color = presentation)) + 
  geom_line() 


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
  ggplot(aes(x = month_num, y = target_avg, color = country)) + 
  geom_line()



# gx_full %>% 
#   group_by(brand, month_num) %>% 
#   summarise(
#     target_50 = median(target),
#     target_avg = mean(target),
#     target_25 = quantile(target, .25),
#     target_75 = quantile(target, .75),
#     n = n()
#   ) %>% 
#   ungroup() %>% 
#   filter(n > 5) %>%
#   filter(brand %in% c("brand_1", "brand_2")) %>%
#   ggplot(aes(x = month_num, y = target_avg)) + 
#   geom_line() + 
#   geom_ribbon(aes(ymin = target_25, ymax = target_75), alpha = 0.1) +
#   facet_wrap(~brand)


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
  ggplot(aes(x = month_num, y = target_avg, color = therapeutic_area)) + 
  geom_line()


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
  ggplot(aes(x = month_num, y = target_avg, color = num_generics)) + 
  geom_line()
