library(tm)
library(wordcloud)
library(SnowballC)  
library(RWeka)
library(rpart)
library(caret)
library(class) # KNN model
library(rpart)
library(randomForest)

tp2.work$cant_amb <- ifelse(tp2.work$cant_amb>5,">6",tp2.work$cant_amb)
#tp2.work$cant_amb <- ifelse(tp2.work$cant_amb %in% c(">6","6"),">=6",tp2.work$cant_amb)
tp2.work$cant_amb <- ifelse(tp2.work$cant_amb %in% c("2","3"),"2-3",tp2.work$cant_amb)
tp2.work$cant_amb <- ifelse(tp2.work$cant_amb %in% c("4","5"),"4-5",tp2.work$cant_amb)



# Set seed for reproducible results
set.seed(100)

str(tp2.work)
#review_text <- paste(tp2.work$des, collapse="")
#review_source <- VectorSource(review_text)

corpus <- Corpus(VectorSource(tp2.work$des[!is.na(tp2.work$cant_amb)]))

#corpus <- Corpus(review_source)
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, stripWhitespace)
corpus <- tm_map(corpus, removeWords, stopwords("spanish"))
corpus <- tm_map(corpus, stemDocument, language = "spanish")
dtm <- DocumentTermMatrix(corpus)

# Transform dtm to matrix to data frame - df is easier to work with
mat.df <- as.data.frame(data.matrix(dtm), stringsAsfactors = FALSE)

# Column bind category (known classification)
mat.df <- cbind(mat.df[,colSums(mat.df)>100], factor(tp2.work$cant_amb[!is.na(tp2.work$cant_amb)]))
mat.df <- cbind(mat.df[,colSums(mat.df)>300], factor(tp2.work$cant_amb))


save(dtm,file = "dtm")
remove(dtm)
# Change name of new column to "category"
colnames(mat.df)[ncol(mat.df)] <- "category"

# Split data by rownumber into two equal portions
train <- sample(nrow(mat.df), ceiling(nrow(mat.df) * .50))
test <- (1:nrow(mat.df))[- train]

train <- which(!is.na(mat.df$category))
test <- which(is.na(mat.df$category))

# Isolate classifier
cl <- mat.df[, "category"]

# Create model data and remove "category"
modeldata <- mat.df[,!colnames(mat.df) %in% "category"]

# Create model: training set, test set, training set classifier
knn.pred <- knn(modeldata[train, ], modeldata[test, ], cl[train])

# Confusion matrix
conf.mat <- table("Predictions" = knn.pred, Actual = cl[test])
conf.mat

# Accuracy
(accuracy <- sum(diag(conf.mat))/length(test) * 100)

# Create data frame with test data and predicted category
df.pred <- cbind(knn.pred, modeldata[test, ])
write.table(df.pred, file="output.csv", sep=";")

tp2.work$cant_amb[test] <- as.character(knn.pred)
