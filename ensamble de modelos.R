library(randomForest)
library(caret)
library(C50)
library(pvclust)
library(arules)
library(mice)
library(e1071)
library(cluster)
library(rpart)
library(RWeka)
library(MASS)
library(nnet)

tp2.work$sup_tot_dis <- tp2.work$sup_tot_m2
tp2.work$sup_tot_m2 <- tp2.original$sup_tot_m2
tp2.work$sup_cub_dis <- discretize(tp2.work$sup_cub_m2,method = "frequency",categories = 5)

#clustering de barrios


barrios <- as.numeric(gsub(",",".",tp2.work[,c("geoname_num")]))

set.seed(20)
barriosCluster <- kmeans(barrios, 5)
irisCluster

# Ward Hierarchical Clustering
d <- dist(barrios, method = "euclidean") # distance matrix
fit <- hclust(d, method="ward")
plot(fit) # display dendogram
groups <- cutree(fit, k=3) # cut tree into 5 clusters
# draw dendogram with red borders around the 5 clusters
rect.hclust(fit, k=3, border="red") 

tp2.work$cluster <- factor(groups)
tp2.work$Clase <- factor(tp2.work$Clase)
tp2.work$cant_amb <- factor(tp2.work$cant_amb)
tp2.work$lugar <- factor(tp2.work$lugar)
tp2.work$tipoprop <- factor(tp2.work$tipoprop)
tp2.work$piso <- factor(tp2.work$piso)

testmice <- mice(tp2.work,method = "pmm")
tp2.complete <- complete(testmice)

#clustering de barrios

barrios <- tp2.complete[,c("lugar","tipoprop","cant_amb")]

barrios$geoname_num <- as.numeric(gsub(",",".",barrios[,c("geoname_num")]))

set.seed(20)
barriosCluster <- kmeans(barrios[,2:3], 10)
irisCluster

#primer modelo rf
set.seed(123)
intrain <- createDataPartition(y=mydata$Clase,p=0.80,list = FALSE)
tp2_train <- mydata[intrain,]
tp2_test <- mydata[-intrain,]

model <- Clase ~ tipoprop+sup_tot_m2+cant_amb+sup_cub_m2+sup_cub_dis+fit.cluster+sup_tot_dis
rf.fit <- randomForest(Clase ~ tipoprop+sup_tot_m2+cant_amb+sup_cub_m2+sup_cub_dis+fit.cluster+sup_tot_dis, data = tp2_train, ntree=200)
rf.fit$confusion

rf.pred <- predict(rf.fit,tp2_test)
confusionMatrix(tp2_test$Clase,rf.pred) #Accuracy : 0.4234(clase 5 68%)

'                     Class: 1 Class: 2 Class: 3 Class: 4 Class: 5
Sensitivity            0.4631  0.42197  0.39535  0.49071   0.3807
Specificity            0.8291  0.84880  0.84742  0.86051   0.9036
Pos Pred Value         0.4288  0.14600  0.37268  0.43493   0.6891
Neg Pred Value         0.8479  0.95995  0.85940  0.88536   0.7223
Prevalence             0.2169  0.05772  0.18652  0.17951   0.3594
Detection Rate         0.1004  0.02436  0.07374  0.08809   0.1368
Detection Prevalence   0.2342  0.16683  0.19786  0.20254   0.1985
Balanced Accuracy      0.6461  0.63538  0.62138  0.67561   0.6422'

#modelo j48

tp2.complete$geoname_num <- as.numeric(gsub(",",".",tp2.complete[,c("geoname_num")]))

arbol.fit <- J48(model,data=tp2_train)
arbol.pred <- predict(arbol.fit,tp2_test)
confusionMatrix(tp2_test$Clase,arbol.pred) #Accuracy : 0.4154 clase 1 0.4715

'                     Class: 1 Class: 2 Class: 3 Class: 4 Class: 5
Sensitivity            0.4276  0.31595  0.39734  0.44724   0.4328
Specificity            0.8331  0.85137  0.84460  0.85833   0.8830
Pos Pred Value         0.4715  0.20600  0.35245  0.43987   0.5630
Neg Pred Value         0.8070  0.91069  0.86814  0.86192   0.8172
Prevalence             0.2583  0.10878  0.17551  0.19920   0.2583
Detection Rate         0.1104  0.03437  0.06974  0.08909   0.1118
Detection Prevalence   0.2342  0.16683  0.19786  0.20254   0.1985
Balanced Accuracy      0.6304  0.58366  0.62097  0.65278   0.6579'

#modelo SVM

modelo.svm    <- svm(model,data=tp2_train)
predict.svm   <- predict(modelo.svm,tp2_test)

confusionMatrix(tp2_test$Clase,predict.svm) #Accuracy : 0.2513 (clase 1 66%)

#red neuronal
model_nnet <- Clase ~ sup_tot_m2+sup_cub_m2+piso+sup_cub_dis+fit.cluster+sup_tot_dis

fit.nnet <- nnet(model,data=tp2_train,size=5) 

pr.nn   <- predict(fit.nnet,tp2_test,type="class")

confusionMatrix(tp2_test$Clase,factor(pr.nn)) #Accuracy : 0.3259927 (clase 1 66%)

# ----------------------------------------------------------------
# ensamble de los 4 modelos anteriores
ensamble          <- data.frame(rf.pred, arbol.pred, predict.svm,factor(pr.nn))
ensamble$uno <- rowSums(ensamble[, 1:4] == "1")
ensamble$dos <- rowSums(ensamble[, 1:4] == "2")
ensamble$tres <- rowSums(ensamble[, 1:4] == "3")
ensamble$cuatro <- rowSums(ensamble[, 1:4] == "4")
ensamble$cinco <- rowSums(ensamble[, 1:4] == "5")

ensamble$predict  <- ifelse(ensamble$uno>ensamble$dos & ensamble$uno>ensamble$tres & ensamble$uno>ensamble$cuatro & ensamble$uno>ensamble$cinco
                            ,"1",
                            ifelse(ensamble$dos>ensamble$uno & ensamble$dos>ensamble$tres & ensamble$dos>ensamble$cuatro & ensamble$dos>ensamble$cinco
                                   ,"2",
                                   ifelse(ensamble$tres>ensamble$uno & ensamble$tres>ensamble$dos & ensamble$tres>ensamble$cuatro & ensamble$tres>ensamble$cinco
                                          ,"3",
                                          ifelse(ensamble$cuatro>ensamble$uno & ensamble$cuatro>ensamble$tres & ensamble$cuatro>ensamble$dos & ensamble$cuatro>ensamble$cinco
                                                 ,"4",
                                                "5"))))

# ----------------------------------------------------------------
# Matriz de Confusion
confusionMatrix(tp2_test$Clase,ensamble$predict)

# Accuracy
(accuracy <- sum(diag(conf.mat))/length(test) * 100)