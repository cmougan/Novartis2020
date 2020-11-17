library(tidyverse)


load('data/target_tbl_rf.RData')
load('data/target_tbl_glm.RData')
load('data/target_tbl_xgb.RData')






predict_tbl <- function(target_tbl){
  
  target_tbl$target_predict <- NA
  for(month_predict in 1:12){
    
    preds_col <- paste0("preds_", month_predict)
    
    target_tbl[(target_tbl$month == month_predict), "target_predict"] <- 
      target_tbl[(target_tbl$month == month_predict), preds_col]
  }
  
  target_tbl
}

target_tbl_glm <- predict_tbl(target_tbl_glm)
target_tbl_rf <- predict_tbl(target_tbl_rf)
target_tbl_xgb <- predict_tbl(target_tbl_xgb)


# MAE glm + rf + xgb
yardstick::mae_vec(
  target_tbl_glm$target,
  (target_tbl_glm$target_predict + 
    target_tbl_rf$target_predict +
     target_tbl_xgb$target_predict
     )/3
)    
# 112.8!


yardstick::mae_vec(
  target_tbl_glm$target,
  (
    target_tbl_rf$target_predict +
      target_tbl_xgb$target_predict
  )/2
)    
# Without glm it is 115, so it actually helps!!

median_agg <- pmap_dbl(
  list(
    target_tbl_glm$target_predict, target_tbl_rf$target_predict, target_tbl_xgb$target_predict),
     median
  )

# Median aggregation doesn't work well
yardstick::mae_vec(target_tbl_glm$target, median_agg)

