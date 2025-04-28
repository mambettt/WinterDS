library(readr)
library(caret)
library(ggplot2)
library(lattice)
library(plotly)
library(networkD3)
library(tidyverse)
library(randomForest)
library(xgboost)
library(class)

data <- read_csv("lung_cancer.csv")


#Проверка данных

#просмотр переменных
names(data)

#проверка структуры данных
str(data)

#просмотр первых строк
head(data)

#основная статистика
summary(data)

#проверка пропущеных значений
colSums(is.na(data))

#Определяем тип данных
sapply(data, class)


#Визуализация

#Построение графика скаттер плот
qplot(x = ALLERGY, data = data, y = LUNG_CANCER, color = GENDER,
      size = AGE, main = "Scatter Plot",
      xlab = "Allergy", ylab = "Lust cancer") +
  theme_minimal()

#График скаттер плот расширенный
qplot(x = AGE, data = data, y = LUNG_CANCER, color = GENDER,
      size = AGE, main = "Scatter Plot with Line",
      xlab = "Age", ylab = "Lust cancer")
geom_smooth(method = "lm")

#all скаттер плот
plot(data, main = "Scatter Plot All",
     col = "#FF5733", pch = 19)

#Построение графика pie chart
qplot(x = "" , data = data , geom = "bar", fill = GENDER,
      main = "Pie Chart",
      xlab = NULL, ylab = NULL) +
  coord_polar(theta = "y") +
  theme_void()

#Построение графика boxplot

qplot(x = SMOKING, data = data, y = 'CHEST PAIN', color = GENDER,
      geom = "boxplot",
      main = "Boxplot",
      xlab = "Smoking", ylab = "Chest Pain", fill = GENDER ) +
  theme_minimal()


#Построение графиков гистограмма
qplot(x = COUGHING, data = data,,
      main = "Histogramm",
      xlab = "COUGHING", fill = GENDER ) +
  theme_bw()

qplot(x = SMOKING, data = data,,
      main = "Histogramm",
      xlab = "Smoking", fill = GENDER ) +
  theme_bw()

qplot(x = YELLOW_FINGERS, data = data,,
      main = "Histogramm",
      xlab = "Yellow Fingers", fill = GENDER ) +
  theme_bw()

qplot(x = ANXIETY, data = data,
      main = "Histogramm",
      xlab = "Anxiety", fill = GENDER ) +
  theme_bw()

qplot(x = PEER_PRESSURE, data = data,,
      main = "Histogramm",
      xlab = "Peer Pressure", fill = GENDER ) +
  theme_bw()

qplot(x = "CHRONIC DISEASE", data = data,,
      main = "Histogramm",
      xlab = "Chronic Disease", fill = GENDER ) +
  theme_bw()

qplot(x = FATIGUE, data = data,,
      main = "Histogramm",
      xlab = "Fatigue", fill = GENDER ) +
  theme_bw()

qplot(x = ALLERGY, data = data,,
      main = "Histogramm",
      xlab = "Allergy", fill = GENDER ) +
  theme_bw()

qplot(x = WHEEZING, data = data,,
      main = "Histogramm",
      xlab = "Wheezing", fill = GENDER ) +
  theme_bw()

qplot(x = "ALCOHOL CONSUMING", data = data,,
      main = "Histogramm",
      xlab = "Alcohol Consuming", fill = GENDER ) +
  theme_bw()

qplot(x = "SHORTNESS OF BREATH", data = data,,
      main = "Histogramm",
      xlab = "Shortness of breath", fill = GENDER ) +
  theme_bw()

qplot(x = "SWALLOWING DIFFICULTY", data = data,,
      main = "Histogramm",
      xlab = "Swallowing difficulty", fill = GENDER ) +
  theme_bw()

qplot(x = "CHEST PAIN", data = data,,
      main = "Histogramm",
      xlab = "Chest pain", fill = GENDER ) +
  theme_bw()

qplot(x = LUNG_CANCER, data = data,,
      main = "Histogramm",
      xlab = "Lung cancer", fill = GENDER ) +
  theme_bw()

qplot(x = GENDER, data = data,,
      main = "Histogramm",
      xlab = "Gender", fill = GENDER ) +
  theme_bw()

#Тепловая карта
numeric_data <- Filter(is.numeric, data)

heatmap(as.matrix(numeric_data),
        color = colorRampPalette(c("blue", "white", "red"))(100),
        display_numbers = TRUE,
        main = "Теловая карта",
        Rowv = NA,
        Colv = NA,
        scale = 'column')




# Обучение модели randomForest


set.seed(2024)

# Разделение данных на обучающую и тестовую выборки
train_index <- createDataPartition(data$LUNG_CANCER, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Проверка классов в обучающих данных
print(table(train_data$LUNG_CANCER))

# Переименование переменной
names(train_data)[names(train_data) == "CHRONIC DISEASE"] <- "CHRONIC_DISEASE"
names(train_data)[names(train_data) == "ALCOHOL CONSUMING"] <- "ALCOHOL_CONSUMING"
names(train_data)[names(train_data) == "SHORTNESS OF BREATH"] <- "SHORTNESS_OF_BREATH"
names(train_data)[names(train_data) == "SWALLOWING DIFFICULTY"] <- "SWALLOWING_DIFFICULTY"
names(train_data)[names(train_data) == "CHEST PAIN"] <- "CHEST_PAIN"

names(test_data)[names(test_data) == "CHRONIC DISEASE"] <- "CHRONIC_DISEASE"
names(test_data)[names(test_data) == "ALCOHOL CONSUMING"] <- "ALCOHOL_CONSUMING"
names(test_data)[names(test_data) == "SHORTNESS OF BREATH"] <- "SHORTNESS_OF_BREATH"
names(test_data)[names(test_data) == "SWALLOWING DIFFICULTY"] <- "SWALLOWING_DIFFICULTY"
names(test_data)[names(test_data) == "CHEST PAIN"] <- "CHEST_PAIN"

model_rf <- randomForest(LUNG_CANCER ~ ., data = train_data, importance = TRUE)

# Предсказания на тестовых данных
predictions_rf <- predict(model_rf, newdata = test_data)

# Получение матрицы 
conf_matrix_rf <- confusionMatrix(predictions_rf, test_data$LUNG_CANCER)

# Вывод результатов
print(conf_matrix_rf)

# Получение точности
accuracy_rf <- conf_matrix_rf$overall["Accuracy"]
print(accuracy_rf)



# Обучение модели xgboost
set.seed(2024)
train_index <- createDataPartition(data$LUNG_CANCER, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]


train_data$LUNG_CANCER <- ifelse(train_data$LUNG_CANCER == "YES", 1, 0)
test_data$LUNG_CANCER <- ifelse(test_data$LUNG_CANCER == "YES", 1, 0)

train_data_numeric <- model.matrix(~ . - 1, data = train_data)
test_data_numeric <- model.matrix(~ . - 1, data = test_data)

# Подготовка данных для xgboost
dtrain <- xgb.DMatrix(data = train_data_numeric, label = train_data$LUNG_CANCER)
dtest <- xgb.DMatrix(data = test_data_numeric, label = test_data$LUNG_CANCER)

#Параметры
params <- list(objective = "binary:logistic", 
               eval_metric = "logloss", 
               scale_pos_weight = sum(train_data$LUNG_CANCER == 0) / sum(train_data$LUNG_CANCER == 1))

model_xgb <- xgb.train(params = params, 
                       data = dtrain, 
                       nrounds = 100)
#Предсказание на тестовых данных
predictions_xgb <- predict(model_xgb, dtest)
predictions_xgb <- ifelse(predictions_xgb > 0.5, 1, 0)

#Получение матрицы
conf_matrix_xgb <- confusionMatrix(as.factor(predictions_xgb), as.factor(test_data$LUNG_CANCER))
print(conf_matrix_xgb)

#Получение точности
accuracy <- conf_matrix_xgb$overall['Accuracy']
print (accuracy)



#knn

set.seed(2024)
train_labels <- as.factor(train_data$LUNG_CANCER)
test_labels  <- as.factor(test_data$LUNG_CANCER)

#Подготовка признаков
train_features <- train_data[, !(names(train_data) %in% "LUNG_CANCER")]
test_features  <- test_data[, !(names(test_data) %in% "LUNG_CANCER")]

#Объединение данных
tmp_train <- train_features
tmp_test <- test_features
tmp_train$dataset <- "train"
tmp_test$dataset <- "test"
all_features <- rbind(tmp_train, tmp_test)
dummies <- as.data.frame(model.matrix(~ . - 1, data = all_features))

#train и test
train_features_matrix <- dummies[all_features$dataset == "train", ]
test_features_matrix  <- dummies[all_features$dataset == "test", ]
train_features_matrix$dataset <- NULL
test_features_matrix$dataset <- NULL

#Проверка на пропуски
stopifnot(sum(is.na(train_features_matrix)) == 0)
stopifnot(sum(is.na(test_features_matrix)) == 0)

#Обучение
k <- 5  
predictions_knn <- knn(
  train = train_features_matrix,
  test  = test_features_matrix,
  cl    = train_labels,
  k     = k
)

#Получение матрицы
conf_matrix_knn <- confusionMatrix(predictions_knn, test_labels)
print(conf_matrix_knn)

#Получение точности
accuracy_knn <- conf_matrix_knn$overall['Accuracy']
print(accuracy_knn) 


#Вывод: лучший результат после обучения показала модель градиентного бустинга(1),
#худший knn(0.852)