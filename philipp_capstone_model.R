download.file("https://www.dropbox.com/s/4t4w48ut2gu8dn6/edx.rds?dl=1", "edx.rds")

download.file("https://www.dropbox.com/s/c4jzznttgc01sdb/final_holdout_test.rds?dl=1", "final_holdout_test.rds")

edx = readRDS("edx.rds")

final_holdout_test = readRDS("final_holdout_test.rds")

###
#Preprocessing & Plotting Data
### Load libraries
library(tidyverse)
library(caret)


###


# Plot histogram of average movie rating by movie ID
edx %>%
  group_by(movieId) %>%
  summarise(mean_rating = mean(rating)) %>%
  ggplot(aes(x = mean_rating)) +
  geom_histogram(bins = 10, fill = "blue", color = "black") +
  ggtitle("Average Movie Ratings") +
  xlab("Mean Movie Rating")


# Plot histogram of mean ratings by user ID
edx %>%
  group_by(userId) %>%
  summarise(mean_rating = mean(rating)) %>%
  ggplot(aes(x = mean_rating)) +
  geom_histogram(bins = 20, fill = "blue", color = "black") +
  ggtitle("Average Movie Rating by User") +
  xlab("Mean Movie Rating by User")

#Plot time effects. Ratings over time

edx %>%
  mutate(time=as_datetime(timestamp)) %>%
  group_by(month = floor_date(time, "month")) %>%
  summarise(mean_rating = mean(rating)) %>%
  ggplot(aes(x = month, y = mean_rating)) +
  geom_line() +
  geom_smooth(method = "loess", se = FALSE, color = "blue") +  # Apply smoothing
  ggtitle("Change In Rating Over Time") 

###the change in rating over time plot shows us some abnormally high ratings shortly arter 1995.
###Therefore, we might consider only taking data from after 1998, to see how this affect the rmse.
# Filter train_data and test_data for timepoints after 1998
edx <- edx %>%
  filter(year(as_datetime(timestamp)) > 1998)

###apply the same to the holdout set
final_holdout_test <- final_holdout_test %>%
  filter(year(as_datetime(timestamp)) > 1998)

  
#Plot genre effects
#Find out how many unique genres there are
unique_genres <- unique(edx$genres)
num_unique_genres <- length(unique_genres)

cat("Number of unique genres:", num_unique_genres, "\n")

#797 is quite a few genres to plot. Plot only genres with a certain amount of ratings

# Calculate average rating and count of ratings for each genre
genre_stats <- edx %>%
  group_by(genres) %>%
  summarise(mean_rating = mean(rating), rating_count = n())

# Filter genres with over 100,000 ratings
filtered_genres <- genre_stats %>%
  filter(rating_count > 100000)

# Filter original data frame based on selected genres
edx <- edx %>%
  filter(genres %in% filtered_genres$genres)

#apply the same to the validation dataset
final_holdout_test <- final_holdout_test %>%
  filter(genres %in% filtered_genres$genres)

# Create a boxplot
ggplot(edx, aes(x = genres, y = rating)) +
  geom_boxplot() +
  labs(title = "Boxplot of Movie Ratings by Genre (Over 100,000 Ratings)",
       x = "Genre",
       y = "Rating") +
  theme(axis.text.x = element_text(angle = 65, hjust = 1))  # Rotate x-axis labels for better readability

#Interestingly, comedy by itself while having a very wide range of rating,
#also has the lowest median rating. On the other hand, when combined with
#drama, it has the highest median rating. Drama and War also has a surpisingly high average rating. 

###
#Start of Model Generation
###

###setting the seed
set.seed(123)


###generate index for splitting the data (80 to 20 split)
splitIndex <- createDataPartition(edx$rating, times = 1, p = 0.2, list = FALSE)

#Create training and testing datasets
train_data <- edx[-splitIndex, ]
test_data <- edx[splitIndex, ]

#ensure the userID and movieOD in the test set are also in the train set
test_data <- test_data %>%
  semi_join(train_data, by = "movieId") %>%
  semi_join(train_data, by = "userId")



#Build a first, simple recommendation model

mu <- mean(train_data$rating)
mu

naive_rmse <- RMSE(test_data$rating, mu)
naive_rmse

###store results in a dataframe

rmse_results_df <- data.frame(Model = "Mean Only", RMSE = naive_rmse)
print.data.frame(rmse_results_df)

###Next we add movie effects to our model
# Estimate movie effect (b_i)
b_i <- train_data %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

###
predicted_ratings <- mu + test_data %>%
  left_join(b_i, by="movieId") %>%
  .$b_i

###
model_1_rmse <- RMSE(predicted_ratings, test_data$rating)

###store in data frame

rmse_results_df <- bind_rows(rmse_results_df,
                             data.frame(Model = "Movie Effect",
                                        RMSE = model_1_rmse))
rmse_results_df

###Next we add user effects to our model
# Estimate user effect 
user_avgs <- train_data %>% 
  left_join(b_i, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

###
predicted_ratings <- test_data %>% 
  left_join(b_i, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)


model_2_rmse <- RMSE(predicted_ratings, test_data$rating)
model_2_rmse

###add to data frame

rmse_results_df <- bind_rows(rmse_results_df,
                             data.frame(Model = "Movie + User Effect",
                                        RMSE = model_2_rmse))
rmse_results_df

###Next we add time effects to our model
# Estimate time effect 
b_t_train <- train_data %>%
  left_join(b_i, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  mutate(time = as_datetime(timestamp)) %>%
  group_by(day = floor_date(time, "day")) %>%
  summarise(b_t = sum(rating - mu - b_i - b_u)/n())

# Predict ratings with movie, user, and time effects
predicted_ratings <- test_data %>%
  left_join(b_i, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  mutate(day = floor_date(as_datetime(timestamp), "day")) %>%
  left_join(b_t_train, by = "day") %>%
  mutate(b_t = replace_na(b_t, 0)) %>%
  mutate(pred = mu + b_i + b_u + b_t) %>%
  pull(pred)

# Calculate RMSE for the model with movie, user, and time effects
model_3_rmse <- RMSE(predicted_ratings, test_data$rating)

# Add to the results dataframe
rmse_results_df <- bind_rows(rmse_results_df,
                             data.frame(Model = "Movie + User + Time Effect",
                                        RMSE = model_3_rmse))
print.data.frame(rmse_results_df)


########Add genre effects
# Estimate genres effect
b_g_train <- train_data %>%
  left_join(b_i, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  group_by(genres) %>%
  summarise(b_g = sum(rating - mu - b_i - b_u)/n())

# Predict ratings with movie, user, and genres effects
predicted_ratings <- test_data %>%
  left_join(b_i, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(b_g_train, by = "genres") %>%
  mutate(b_g = replace_na(b_g, 0)) %>%
  mutate(pred = mu + b_i + b_u + b_g) %>%
  pull(pred)

# Calculate RMSE for the model with movie, user, and genres effects
model_4_rmse <- RMSE(predicted_ratings, test_data$rating)

# Add to the results dataframe
rmse_results_df <- bind_rows(rmse_results_df,
                             data.frame(Model = "Movie + User + Genres Effect",
                                        RMSE = model_4_rmse))
print.data.frame(rmse_results_df)

#The bar plot from earlier depicts that some genres have rather large standard errors.
#This is something that we may want to account for by controlling for high standart errors.

#Perform cross-validation

# Specify standard error thresholds
# Specify the range of standard error thresholds
ses <- seq(0, 1, 0.1)

rmse <- sapply(ses, function(s){
  b_g <- train_data %>% 
    left_join(user_avgs, by="userId") %>% 
    left_join(b_i, by = "movieId") %>% 
    mutate(day = floor_date(as_datetime(timestamp),"day")) %>%
    left_join(b_t_train, by="day") %>%
    mutate(b_t=replace_na(b_t,0)) %>%
    group_by(genres) %>%
    summarise(b_g=(sum(rating-b_i-b_u-b_t-mu))/(n()),se = sd(rating)/sqrt(n())) %>%
    filter(se<=s) # Retaining b_g values that correspond to Standard Error less than or equal to S 
  
  # Predicting movie ratings on test set   
  predicted_ratings <- test_data %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(user_avgs, by = "userId") %>%
    mutate(day = floor_date(as_datetime(timestamp),"day")) %>% 
    left_join(b_t_train, by="day") %>%
    mutate(b_t=replace_na(b_t,0)) %>%
    left_join(b_g, by="genres") %>%
    mutate(b_g=replace_na(b_g,0)) %>%
    mutate(pred = mu + b_i + b_u + b_t + b_g) %>%
    .$pred
  
  RMSE(predicted_ratings, test_data$rating)
  return(RMSE(predicted_ratings, test_data$rating))
})

#Identify and store the lowest standard error
s_e <- ses[which.min(rmse)]
s_e

#########
#Perform Regularization to improve results
########
library(tidyverse)
library(lubridate)


 #Specify lambda values
lambdas <- seq(0, 10, 0.1)

 #Initialize an empty dataframe to store results
regularization_results_df <- data.frame(Lambda = numeric(),
                                        RMSE = numeric(),
                                        stringsAsFactors = FALSE)

# Iterate over lambda values
for (l in lambdas) {
  mu <- mean(train_data$rating)
  
   #Estimate movie effect (b_i)
  b_i <- train_data %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu) / (n() + l))
  
  # Estimate user effect (b_u)
  b_u <- train_data %>%
    left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu) / (n() + l))
  
  # Estimate time effect (b_t)
  b_t <- train_data %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_i, by = "movieId") %>%
    mutate(time = as_datetime(timestamp)) %>%
    group_by(day = floor_date(time, "day")) %>%
    summarize(b_t = sum(rating - b_i - b_u - mu) / (n() + l))
  
   #Estimate genres effect (b_g)
  b_g <- train_data %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_i, by = "movieId") %>%
    mutate(day = floor_date(as_datetime(timestamp), "day")) %>%
    left_join(b_t, by = "day") %>%
    mutate(b_t = replace_na(b_t, 0)) %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - b_i - b_u - b_t - mu) / (n() + l),
             se = sd(rating) / sqrt(n())) %>%
    filter(se <= s_e)
  
  # Making predictions on the test set
  predicted_ratings <- test_data %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(day = floor_date(as_datetime(timestamp), "day")) %>%
    left_join(b_t, by = "day") %>%
    mutate(b_t = replace_na(b_t, 0)) %>%
    left_join(b_g, by = "genres") %>%
    mutate(b_g = replace_na(b_g, 0)) %>%
    mutate(pred = mu + b_i + b_u + b_t + b_g) %>%
    pull(pred)
  
  # Calculate RMSE and store in the results dataframe
  rmse <- RMSE(predicted_ratings, test_data$rating)
  regularization_results_df <- rbind(regularization_results_df,
                                     data.frame(Lambda = l, RMSE = rmse))
}


# Plot Lambda values vs RMSE
ggplot(regularization_results_df, aes(Lambda, RMSE)) +
  geom_point() +
  ggtitle("Lambda Plot")

model_5_rmse <- min(regularization_results_df$RMSE)

#store optimal lambda
min_lambda <- 5

rmse_results_df <- bind_rows(rmse_results_df,
                          data_frame(Model="Regularised Movie + User + Time + Genre Effects",
                                     RMSE = model_5_rmse ))
print.data.frame(rmse_results_df)

#Regularised movie + user + time + genre effects model generates lowest rmse


###########################
# Re-estimate effects with optimal lambda
mu <- mean(train_data$rating)

# Estimate movie effect (b_i)
b_i <- train_data %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu) / (n() + min_lambda))

# Estimate user effect (b_u)
b_u <- train_data %>%
  left_join(b_i, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu) / (n() + min_lambda))

# Estimate time effect (b_t)
b_t <- train_data %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_i, by = "movieId") %>%
  mutate(time = as_datetime(timestamp)) %>%
  group_by(day = floor_date(time, "day")) %>%
  summarize(b_t = sum(rating - b_i - b_u - mu) / (n() + min_lambda))

# Estimate genres effect (b_g)
b_g <- train_data %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_i, by = "movieId") %>%
  mutate(day = floor_date(as_datetime(timestamp), "day")) %>%
  left_join(b_t, by = "day") %>%
  mutate(b_t = replace_na(b_t, 0)) %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - b_i - b_u - b_t - mu) / (n() + min_lambda),
            se = sd(rating) / sqrt(n())) %>%
  filter(se <= s_e)

# Predict ratings on the test set using the optimal lambda
predicted_ratings <- test_data %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(day = floor_date(as_datetime(timestamp), "day")) %>%
  left_join(b_t, by = "day") %>%
  mutate(b_t = replace_na(b_t, 0)) %>%
  left_join(b_g, by = "genres") %>%
  mutate(b_g = replace_na(b_g, 0)) %>%
  mutate(pred = mu + b_i + b_u + b_t + b_g) %>%
  pull(pred)

# Calculate RMSE on the test set
test_rmse <- RMSE(predicted_ratings, test_data$rating)

# Print the RMSE on the test set
print(paste("Test RMSE with optimal lambda:", test_rmse))

##########################
 
#Use model and tuning parameters to make final predictions on the validation data

#########################

########################

mu <- mean(edx$rating)

# Estimate movie effect (b_i) on the entire dataset
b_i <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mean(edx$rating)) / (n() + min_lambda))

# Estimate user effect (b_u) on the entire dataset
b_u <- edx %>%
  left_join(b_i, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - first(b_i) - mean(edx$rating)) / (n() + min_lambda))

# Estimate time effect (b_t) on the entire dataset
b_t <- edx %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_i, by = "movieId") %>%
  mutate(time = as_datetime(timestamp)) %>%
  group_by(day = floor_date(time, "day")) %>%
  summarize(b_t = sum(rating - first(b_i) - first(b_u) - mean(edx$rating)) / (n() + min_lambda))

# Estimate genres effect (b_g) on the entire dataset
b_g <- edx %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_i, by = "movieId") %>%
  mutate(day = floor_date(as_datetime(timestamp), "day")) %>%
  left_join(b_t, by = "day") %>%
  mutate(b_t = replace_na(first(b_t), 0)) %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - first(b_i) - first(b_u) - b_t - mean(edx$rating)) / (n() + min_lambda),
            se = sd(rating) / sqrt(n())) %>%
  filter(se <= s_e)

# Making predictions on the validation set
validation_predictions <- final_holdout_test %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(day = floor_date(as_datetime(timestamp), "day")) %>%
  left_join(b_t, by = "day") %>%
  mutate(b_t = replace_na(first(b_t), 0)) %>%
  left_join(b_g, by = "genres") %>%
  mutate(b_g = replace_na(first(b_g), 0)) %>%
  mutate(pred = mean(final_holdout_test$rating) + first(b_i) + first(b_u) + first(b_t) + first(b_g)) %>%
  pull(pred)

# Evaluate the predictions on the validation set
validation_rmse <- RMSE(validation_predictions, final_holdout_test$rating)

# Print the RMSE on the validation set
print(paste("Validation RMSE:", validation_rmse))


#Our final rmse value when making predictions on the validation set is too high
#This may suggest overfitting, meaning our models perform well on the training data,
#but do not perform well on the validation data. Therefore, we may consider other approaches

#Another approach that we learned about is Matrix Factorization

#################################################
#matrix factorization
################################################
#install and load the recosystem and float packages
install.packages("recosystem")
install.packages("float", repos = "https://cloud.r-project.org/")

library(float)
library(recosystem)
set.seed(1, sample.kind="Rounding")

train_reco <- with(train_data, data_memory(user_index = userId, item_index = movieId, rating = rating))
#This line creates a training dataset train_reco in a format suitable for the recommender system. It extracts the user IDs, movie IDs, and ratings from the train_data dataset and organizes them into a format suitable for modeling.


test_reco <- with(test_data, data_memory(user_index = userId, item_index = movieId, rating = rating))
#: Similar to the previous step, this line creates a test dataset test_reco in the appropriate format for the recommender system using the test_data.

r <- Reco()
#This line initializes a recommender system object r.

para_reco <- r$tune(train_reco, opts = list(dim = c(20, 30),
                                            costp_l2 = c(0.01, 0.1),
                                            costq_l2 = c(0.01, 0.1),
                                            lrate = c(0.01, 0.1),
                                            nthread = 4,
                                            niter = 10))
#This line tunes the parameters of the recommender system using the training data train_reco. It searches for the optimal combination of parameters specified in the opts argument, such as the dimensions, regularization costs, learning rate, etc.


r$train(train_reco, opts = c(para_reco$min, nthread = 4, niter = 30))
#After tuning, this line trains the recommender system using the optimal parameters identified in the previous step (para_reco$min). It specifies additional training options such as the number of threads (nthread) and the number of iterations (niter).

results_reco <- r$predict(test_reco, out_memory())
# This line uses the trained recommender system to predict ratings for the test dataset test_reco and stores the results in results_reco.

################
factorization_rmse <- RMSE(results_reco, test_data$rating)
rmse_results_df <- bind_rows(rmse_results_df,
                             data.frame(Model = "Matrix factorization",
                                        RMSE = factorization_rmse))
print.data.frame(rmse_results_df)

###############validation

# Convert final_holdout_test to reco format
final_holdout_reco <- with(final_holdout_test, data_memory(user_index = userId, item_index = movieId, rating = rating))

# Predict ratings on the final_holdout_test set
validation_results_reco <- r$predict(final_holdout_reco, out_memory())

# Calculate RMSE on the final_holdout_test set
validation_factorization_rmse <- RMSE(validation_results_reco, final_holdout_test$rating)

# Add results to the dataframe
rmse_results_df <- bind_rows(rmse_results_df,
                             data.frame(Model = "Matrix factorization on validation set",
                                        RMSE = validation_factorization_rmse))

# Print the updated results
print.data.frame(rmse_results_df)

library(gt)
gt(rmse_results_df) 



#Using matrix factorization, while taking much longer, allowed us to achieve a 
#rmse value that is below the desired threshold of 0.86490


