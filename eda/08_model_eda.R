

library(dplyr)
library(ggplot2)
theme_set(theme_minimal())


val_linear <- read.csv("data/blend/val_linear_base_08_12_qe.csv")
# val_linear <- read.csv("data/blend/val_quantiles.csv")

gx <- read.csv("data/gx_merged.csv")
lags <- read.csv("data/gx_merged_lags.csv")

lags <- lags %>% select(country, brand, month_num, offset = last_before_12_after_0)

gx <- gx %>% inner_join(val_linear, by = c("country", "brand", "month_num"))
gx <- gx %>% left_join(lags, by = c("country", "brand", "month_num"))

gx$target <- (gx$volume - gx$offset) / gx$offset

gx$B

gx <- gx %>% mutate(
  absolute_difference = abs(preds - target), 
  channel = case_when(
    B > 75 ~ "B",
    C > 75 ~ "C",
    D > 75 ~ "D",
    T ~ "Mixed",
  ),
  num_generics_fctr = case_when(
    num_generics <= 1 ~ "0. Only 1",
    num_generics < 5 ~ "1. Less than 5, more than 1",
    num_generics < 10 ~ "2. Less than 10, more than 5",
    num_generics < 20 ~ "3. Less than 20, more than 10",
    T ~ "4. More than 20",
  )
)

gx %>%
  ggplot() + 
  # geom_violin(aes(x = as.factor(month_num), y = target)) + 
  geom_smooth(aes(x = month_num, y = target, color = "Target"), se = FALSE) +
  geom_smooth(aes(x = month_num, y = preds), se = FALSE) +
  geom_smooth(aes(x = month_num, y = lower), se = FALSE) +
  geom_smooth(aes(x = month_num, y = upper), se = FALSE)
  


gx %>% 
  group_by(
    month_num
  ) %>% 
  summarise(
    median_diff = median(absolute_difference)
  ) %>% 
  ggplot(aes(x = month_num, y = median_diff)) + 
  geom_line() + 
  geom_smooth()


gx %>% 
  ggplot(aes(x = as.factor(month_num), y = absolute_difference)) + 
  geom_boxplot() + 
  geom_smooth(aes(x = month_num, y = absolute_difference))




gx %>% 
  filter(country %in% c("country_12", "country_7", "country_3")) %>% 
  ggplot(aes(x = as.factor(month_num), y = absolute_difference)) + 
  geom_boxplot() + 
  geom_smooth(aes(x = month_num, y = absolute_difference)) + 
  facet_wrap(~ country)


gx %>% 
  ggplot(aes(x = as.factor(month_num), y = absolute_difference)) + 
  geom_boxplot() + 
  geom_smooth(aes(x = month_num, y = absolute_difference)) + 
  facet_wrap(~presentation)



gx %>% 
  ggplot(aes(x = as.factor(month_num), y = absolute_difference)) + 
  geom_boxplot() + 
  geom_smooth(aes(x = month_num, y = absolute_difference)) + 
  facet_wrap(~num_generics_fctr)

gx %>% 
  ggplot(aes(x = month_num, y = target, color = channel)) + 
  geom_smooth(se = F) + 
  facet_wrap(~country)


gx %>% 
  ggplot(aes(x = month_num, y = target, color = channel)) + 
  geom_smooth(se = F) + 
  facet_wrap(~presentation)


gx %>% 
  ggplot(aes(x = month_num, y = target, color = channel)) + 
  geom_smooth(se = F)


gx %>% 
  filter(
    country %in% c("country_12", "country_15", "country_7", "country_3"),
    presentation %in% c("OTHER", "PILL")
    ) %>% 
  ggplot(aes(x = month_num, y = target, color = presentation)) + 
  geom_smooth(se = F) + 
  facet_wrap(~country)


gx %>% 
  filter(
    country %in% c("country_12", "country_3"),
    presentation %in% c("CREAM", "PILL")
  ) %>%
  ggplot(aes(x = month_num, y = target, color = presentation)) + 
  geom_smooth(se = F, span = 1.) + 
  facet_wrap(~country)

gx %>% 
  ggplot(aes(x = month_num, y = target, color = presentation)) + 
  geom_smooth(se = F) + 
  facet_wrap(~country)

