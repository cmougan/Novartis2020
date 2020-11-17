

error_metric <- function(real, prediction, upper_bound, lower_bound) {
  
  real_tilde <- pmax(real, 1)
  
  general_term <- (abs(real - prediction) + abs(real - upper_bound) + abs(real - lower_bound)) / real_tilde
  
  relative_term_upper <- (1.5 + (2 * abs(real - upper_bound) / real_tilde))
  relative_term_lower <- (1.5 + (2 * abs(real - lower_bound) / real_tilde))
  
  
  relative_term <- rep(1, length(general_term))
  relative_term[real > upper_bound] <- relative_term_upper[real > upper_bound]
  
  relative_term[real < lower_bound] <- relative_term_lower[real < lower_bound]
  print("Relative term")
  print(mean(relative_term))
  print("General term")
  print(mean(general_term))
  
  mean(relative_term * general_term)
  
}


# real = 0 * 1:100 
# prediction = real + 1
# upper_bound = prediction + 0:99
# lower_bound = prediction - 0:99

# print(error_metric(real, prediction, upper_bound, lower_bound))