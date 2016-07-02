library(tm)
library(wordcloud)
library(SnowballC)  
library(RWeka)
library(rpart)
library(caret)
library(class) # KNN model
library(arules)
library(randomForest)
library(stringi)


tp2.work$des <- stri_encode(tp2.work$des, "", "UTF-8") # re-mark encodings

tp2.work$sup_tot_m2 <- discretize(tp2.work$sup_tot_m2,method = "frequency",categories = 5)

# Set seed for reproducible results
set.seed(100)

str(tp2.work)
#review_text <- paste(tp2.work$des, collapse="")
#review_source <- VectorSource(review_text)

#corpus <- Corpus(VectorSource(tp2.work$des[!is.na(tp2.work$piso)]))
corpus <- Corpus(VectorSource(tp2.work$des[!is.na(tp2.work$sup_tot_m2)]))

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
mat.df <- cbind(mat.df[,colSums(mat.df)>300], factor(tp2.work$sup_tot_m2[!is.na(tp2.work$sup_tot_m2)]))
save(dtm,file = "dtm")
remove(dtm)
# Change name of new column to "category"
colnames(mat.df)[ncol(mat.df)] <- "category"

# Split data by rownumber into two equal portions
train <- sample(nrow(mat.df), ceiling(nrow(mat.df) * .70))
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


tp2.work$piso[test] <- as.character(knn.pred)