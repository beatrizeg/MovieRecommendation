##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")


# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#START OF MY OWN CODE

#1. INTRODUCTION
#Load lubridate and ggplot2 libraries and work with 7 digit options
library(lubridate)
library(ggplot2)
library(tidyverse)
library(caret)
library(data.table)
options(digits=7)

#Create RMSE function
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings-predicted_ratings)^2))
}

#Check of NAs in rating
nas <- sum(is.na(edx$rating)) #no NAs


#2. METHOD AND ANALISYS
#2.1. Exploring edx dataset
h_movie <- hist(edx$movieId,
                main="Histogram for MovieId", 
                xlab="MovieId", 
                border="blue", 
                col="green",
                xlim=c(0,11000)) #create movieId histogram
h_user <- hist(edx$userId,
               main="Histogram for UserId", 
               xlab="UserId", 
               border="blue", 
               col="green",
               xlim=c(0,80000),
               ylim=c(10000,1000000)) #create userId histogram
h_rating <- hist(edx$rating,
                 main="Histogram for ratings", 
                 xlab="Rating", 
                 border="blue", 
                 col="green") #create rating histogram

#2.2. Check different effects
#2.2.1. Check timestamp effect
#Converting timestamp variable from integer to date format and introducing new columns: weekday, month and year
edx <- edx %>%
  mutate(date=as_datetime(timestamp), weekday = wday(date), month=month(date), year=year(date)) %>%
  select(userId, movieId, rating, title, genres, weekday, month, year, date)

#Plot average ratings vs weekday to see if there is an effect to consider
edx %>% 
  group_by(weekday) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(weekday, rating)) +
  geom_point() +
  geom_smooth()

#We see on Wednesdays and Thursdays average rating decreases, but will not consider this effect in the model
edx <- edx %>% select(-weekday) #Remove weekday column

#Plot average ratings vs month to see if there is an effect to consider
edx %>% 
  group_by(month) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(month, rating)) +
  geom_point() +
  geom_smooth()

#As there is no month effect we remove the column
edx <- edx %>% select(-month) #Remove month column

#Plot average ratings vs year to see if there is an effect to consider
edx %>% 
  group_by(year) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(year, rating)) +
  geom_point() +
  geom_smooth()

#Separate release year from title and keep in a "year_release" new column. Also add new column named time which is the time between
#user rated the movie and the release year.
library(stringr)
pattern <- "\\(\\d{4}\\)"
edx <- edx %>% 
  mutate(temp_title=str_extract(title, pattern)) %>% #extract string starting with a ( and followed by 4 digits, keep in temp_title
  mutate(year_release=as.numeric(str_extract(temp_title, "\\d{4}"))) %>% #from temp_title we extract the 4 digits (year_release) and convert to numeric
  mutate(time=year-year_release) %>% #time between user rated the movie and the release year
  mutate(time=ifelse(time>=0, time, 0)) %>% #in a few cases year of rating was prior to year_release which is impossible, that is why we replace those by 0.
  select(-temp_title) #remove temp_title column

#Plot average ratings vs "time" to see if there is an effect to consider
edx %>% 
  group_by(time) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(time, rating)) +
  geom_point() +
  geom_smooth()
#We can see that there is an effect to consider

#2.2.2. Check number of ratings per movie effect
#Add column with number of ratings per movie
edx <- edx %>% 
  group_by(movieId) %>%
  mutate(n=n())

#Check if number of ratings of movie affect rating
edx %>% 
  group_by(n) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(n, rating)) +
  geom_point() +
  geom_smooth()
#We can see that there is an effect to consider  

#2.3 Creating test set and train set
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index2 <- createDataPartition(y = edx$rating, times = 1, p = 0.2,
                                   list = FALSE) #Test set is 20% of the edx data
train_set <- edx[-test_index2,]
test_set <- edx[test_index2,]

test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>% 
  semi_join(train_set, by = "userId") #so that test set does not contain users or movies that do not appear in train set

#3 Results
#3.1 Baseline model


#We predict the same rating for every movie and user, which is the average rating
mu_hat <- mean(train_set$rating)
baseline_rmse <- RMSE(test_set$rating, mu_hat)

options(pillar.sigfig = 5) #tibble to show 5 significant digits
rmse_results <- tibble(method = "Just the average", RMSE = baseline_rmse) #Create table to store different RMSEs results to compare

#3.2 We add movie bias calculating average rating per movie and including effect in the predictions
movie_avgs <- train_set %>%
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu_hat))

predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>% 
  mutate(pred=mu_hat+b_i) %>%
  pull(pred)

movie_rmse <- RMSE(test_set$rating, predicted_ratings)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie Effect Model",
                                     RMSE = movie_rmse))

#3.3 We add user bias calculating average rating per user and including effect in the predictions
user_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>% group_by(userId) %>%
  summarize(b_u = mean(rating - mu_hat - b_i))

predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>% 
  left_join(user_avgs, by='userId') %>% 
  mutate(pred = mu_hat + b_i + b_u) %>% pull(pred)

user_rmse <- RMSE(test_set$rating, predicted_ratings)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="User+Movie Effect Model",
                                     RMSE = user_rmse))



#3.4 We add number of ratings per movieId effect and include effect in the predictions
n_rating_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>% 
  left_join(user_avgs, by='userId') %>%
  group_by(n) %>%
  summarize(b_n = mean(rating - mu_hat - b_i - b_u))

predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>% 
  left_join(user_avgs, by='userId') %>% 
  left_join(n_rating_avgs, by='n') %>%
  mutate(pred = mu_hat + b_i + b_u + b_n) %>% pull(pred)

ratings_rmse <- RMSE(test_set$rating, predicted_ratings)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="User+Movie+#Ratings Effect Model",
                                     RMSE = ratings_rmse))

#3.5 We add the effect of time from release to rating and include it in the predictions
time_avgs <- train_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(n_rating_avgs, by='n') %>%
  group_by(time) %>%
  summarize(b_t = mean(rating - mu_hat - b_i - b_u - b_n))

predicted_ratings <- test_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(n_rating_avgs, by='n') %>%
  left_join(time_avgs, by='time') %>%
  mutate(pred = mu_hat + b_i + b_u + b_n + b_t) %>% pull(pred)


time_rmse <- RMSE(test_set$rating, predicted_ratings)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="User+Movie+#Ratings+Time Effect Model",
                                     RMSE = time_rmse))

# 3.6 Adding regularization to each of the previous effects

# 3.6.1 Introducing cross-validation

require(caret)
k <- 5
set.seed(5, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(5)`

#Create 5 different partitions from train_set for cross-validation to choose lambda
flds <- createDataPartition(y=train_set$rating, times = k, p=0.2, list = FALSE)


best_lambda <- rep(NA, k) #empty vector to store the lambda that minimizes RMSE in each loop
for (i in 1:k){
flds_index <- flds[,i]
train_i <- train_set[-flds_index,]
temp_test_i <- train_set[flds_index,]
test_i <- temp_test_i %>%
  semi_join(train_i, by = "movieId") %>%
  semi_join(train_i, by = "userId") # Make sure userId and movieId in test_i are also in train_i set
removed_i <- anti_join(temp_test_i, test_i)
train_i <- rbind(train_i, removed_i) # We add removed rows to train_i
lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  mu <- mean(train_i$rating)
  b_i <- train_i %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- train_i %>%
    left_join(b_i, by="movieId") %>% 
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  b_n <- train_i %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    group_by(n) %>%
    summarize(b_n = sum(rating - b_u - b_i - mu)/(n()+l))
  b_t <- train_i %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    left_join(b_n, by="n") %>%
    group_by(time) %>%
    summarize(b_t = sum(rating - b_n - b_u - b_i - mu)/(n()+l))
  predicted_ratings_i <-
    test_i %>%
    left_join(b_i, by = "movieId") %>% 
    left_join(b_u, by = "userId") %>% 
    left_join(b_n, by="n") %>%
    left_join(b_t, by="time") %>%
    mutate(pred = mu + b_i + b_u + b_n + b_t) %>% pull(pred)
  return(RMSE(predicted_ratings_i,test_i$rating)) })
lambda_i <- lambdas[which.min(rmses)]
best_lambda[i] <- lambda_i
}

#3.6.2 Applying lambda to regularization
lambda <- mean(best_lambda)
mu <- mean(train_set$rating)
b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+lambda))
b_u <- train_set %>%
    left_join(b_i, by="movieId") %>% 
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))
b_n <- train_set %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    group_by(n) %>%
    summarize(b_n = sum(rating - b_u - b_i - mu)/(n()+lambda))
b_t <- train_set %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    left_join(b_n, by="n") %>%
    group_by(time) %>%
    summarize(b_t = sum(rating - b_n - b_u - b_i - mu)/(n()+lambda))
predicted_ratings <-
    test_set %>%
    left_join(b_i, by = "movieId") %>% 
    left_join(b_u, by = "userId") %>% 
    left_join(b_n, by="n") %>%
    left_join(b_t, by="time") %>%
    mutate(pred = mu + b_i + b_u + b_n + b_t) %>% pull(pred)

reg_rmse <- RMSE(predicted_ratings, test_set$rating)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized Total Effect Model",
                                    RMSE = reg_rmse))


#3.7 Final Model

lambda <- mean(best_lambda)
mu <- mean(train_set$rating)
b_i <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda))
b_u <- train_set %>%
  left_join(b_i, by="movieId") %>% 
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))
b_n <- train_set %>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  group_by(n) %>%
  summarize(b_n = sum(rating - b_u - b_i - mu)/(n()+lambda))
b_t <- train_set %>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  left_join(b_n, by="n") %>%
  group_by(time) %>%
  summarize(b_t = sum(rating - b_n - b_u - b_i - mu)/(n()+lambda))
predicted_ratings <-
  test_set %>%
  left_join(b_i, by = "movieId") %>% 
  left_join(b_u, by = "userId") %>% 
  left_join(b_n, by="n") %>%
  left_join(b_t, by="time") %>%
  mutate(pred = mu + b_i + b_u + b_n + b_t) %>% 
  mutate(pred = ifelse(pred>5, 5, pred)) %>% #we cap max rating to 5
  mutate(pred = ifelse(pred<0.5, 0.5, pred)) %>% #we cap min rating to 0.5
  pull(pred)

final_rmse <- RMSE(predicted_ratings, test_set$rating)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Capped Regularized Total Effect Model",
                                     RMSE = final_rmse))

#VALIDATED RATINGS

lambda <- 4.7
mu <- mean(edx$rating)
b_i <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda))
b_u <- edx %>%
  left_join(b_i, by="movieId") %>% 
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))
b_n <- edx %>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  group_by(n) %>%
  summarize(b_n = sum(rating - b_u - b_i - mu)/(n()+lambda))
b_t <- edx %>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  left_join(b_n, by="n") %>%
  group_by(time) %>%
  summarize(b_t = sum(rating - b_n - b_u - b_i - mu)/(n()+lambda))
predicted_ratings <-
  validation %>%
  left_join(b_i, by = "movieId") %>% 
  left_join(b_u, by = "userId") %>% 
  left_join(b_n, by="n") %>%
  left_join(b_t, by="time") %>%
  mutate(pred = mu + b_i + b_u + b_n + b_t) %>% 
  mutate(pred = ifelse(pred>5, 5, pred)) %>% #we cap max rating to 5
  mutate(pred = ifelse(pred<0.5, 0.5, pred)) %>% #we cap min rating to 0.5
  pull(pred)

validation_rmse <- RMSE(predicted_ratings, validation$rating)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Validation Model",
                                     RMSE = validation_rmse))