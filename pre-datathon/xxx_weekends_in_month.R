library(chron)
library(dplyr)
library(lubridate)



dates <- seq(from=as.Date("2001-01-01"), to=as.Date("2020-12-31"), by = "day")

df <- tibble(
  weekday = lubridate::wday(dates, label = T),
  date = dates,
  month = month(dates),
  year = year(dates)
  
)

count_months <- df %>% count(year, month, weekday)

week_day_month <- count_months %>% 
  tidyr::pivot_wider(
    id_cols = c('year', 'month'), 
    names_from = weekday, 
    names_prefix = 'n_',
    values_from = n
    ) %>% 
  mutate(
    n_week_day = n_Mon + n_Tue + n_Wed + n_Thu + n_Fri,
    n_real_week_day = n_Mon + n_Tue + n_Wed + n_Thu,
    n_hard_week_day = n_Mon + n_Tue + n_Wed,
  )


week_day_month %>% View

save(week_day_month, file = 'data/week_day_month.RData')

