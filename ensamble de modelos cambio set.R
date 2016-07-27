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
library(ipred)
library(gbm)
library(Cubist)

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
intrain <- createDataPartition(y=tp2.work$Clase,p=0.80,list = FALSE)
tp2_train <- tp2.work[intrain,]
tp2_test <- tp2.work[-intrain,]

model <- Clase ~ tipoprop+sup_tot_m2+sup_cub_m2+sup_cub_dis+cant_amb+cluster
rf.fit <- randomForest(model, data = tp2_train, ntree=200, na.action = na.omit)
rf.fit$confusion

rf.pred <- predict(rf.fit,tp2_test)
confusionMatrix(tp2_test$Clase,rf.pred) #Accuracy : 0.4605

'                     Class: 1 Class: 2 Class: 3 Class: 4 Class: 5
Sensitivity           0.51007 0.447368  0.51639   0.5320   0.4061
Specificity           0.84980 0.883459  0.87458   0.8531   0.8559
Pos Pred Value        0.22419 0.072650  0.22028   0.6135   0.7625
Neg Pred Value        0.95324 0.987395  0.96344   0.8062   0.5584
Prevalence            0.07842 0.020000  0.06421   0.3047   0.5326
Detection Rate        0.04000 0.008947  0.03316   0.1621   0.2163
Detection Prevalence  0.17842 0.123158  0.15053   0.2642   0.2837
Balanced Accuracy     0.67993 0.665414  0.69549   0.6925   0.6310'

#modelo j48

tp2.complete$geoname_num <- as.numeric(gsub(",",".",tp2.complete[,c("geoname_num")]))

model_j48 <- Clase ~ tipoprop+sup_tot+cant_amb+cluster
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

model_svm <- Clase ~ tipoprop+cant_amb+cluster+piso 
modelo.svm    <- svm(model_svm,data=tp2_train)
predict.svm   <- predict(modelo.svm,tp2_test)

confusionMatrix(tp2_test$Clase,predict.svm) #Accuracy : 0.3263 (clase 3 39%)

'                     Class: 1 Class: 2 Class: 3 Class: 4 Class: 5
Sensitivity           0.36551       NA  0.30871  0.37622  0.28399
Specificity           0.80085   0.8332  0.83966  0.84222  0.84381
Pos Pred Value        0.32906       NA  0.39460  0.38056  0.47395
Neg Pred Value        0.82527       NA  0.78203  0.83975  0.70400
Prevalence            0.21088   0.0000  0.25292  0.20487  0.33133
Detection Rate        0.07708   0.0000  0.07808  0.07708  0.09409
Detection Prevalence  0.23423   0.1668  0.19786  0.20254  0.19853
Balanced Accuracy     0.58318       NA  0.57418  0.60922  0.56390'

#red neuronal
model_nnet <- Clase ~ sup_tot_m2+sup_cub_m2+piso+sup_cub_dis+cluster+sup_tot_dis

fit.nnet <- nnet(model_nnet,data=tp2_train,size=10,decay=0.0001, maxit=500) 

pr.nn   <- predict(fit.nnet,tp2_test,type="class")

confusionMatrix(tp2_test$Clase,factor(pr.nn)) #Accuracy : 0.41 (clase 2 52%)

'                     Class: 1 Class: 2 Class: 3 Class: 4 Class: 5
Sensitivity          0.272727 0.304348  0.53448   0.4346   0.3879
Specificity          0.822128 0.879062  0.86156   0.8562   0.8373
Pos Pred Value       0.008850 0.029915  0.10839   0.6813   0.7347
Neg Pred Value       0.994875 0.990396  0.98327   0.6817   0.5408
Prevalence           0.005789 0.012105  0.03053   0.4142   0.5374
Detection Rate       0.001579 0.003684  0.01632   0.1800   0.2084
Detection Prevalence 0.178421 0.123158  0.15053   0.2642   0.2837
Balanced Accuracy    0.547428 0.591705  0.69802   0.6454   0.6126'

#bagging
model_bagging <- Clase ~ sup_tot_m2+sup_cub_m2+piso+sup_cub_dis+cluster+sup_tot_dis

fit.bag <- bagging(model,data=tp2_train) 

pr.bag   <- predict(fit.bag,tp2_test)

confusionMatrix(tp2_test$Clase,pr.bag) #Accuracy : 0.3534

'
                     Class: 1 Class: 2 Class: 3 Class: 4 Class: 5
Sensitivity           0.40938   0.3156  0.41618   0.4065   0.2766
Specificity           0.79826   0.8498  0.83063   0.8755   0.8437
Pos Pred Value        0.27350   0.1900  0.24283   0.5552   0.4891
Neg Pred Value        0.87930   0.9175  0.91597   0.7941   0.6832
Prevalence            0.15649   0.1004  0.11545   0.2766   0.3510
Detection Rate        0.06406   0.0317  0.04805   0.1124   0.0971
Detection Prevalence  0.23423   0.1668  0.19786   0.2025   0.1985
Balanced Accuracy     0.60382   0.5827  0.62341   0.6410   0.5602'

#gbm no funciona
model_bagging <- Clase ~ sup_tot_m2+sup_cub_m2+piso+sup_cub_dis+cluster+sup_tot_dis

fit.gbm <- gbm(model,data=tp2_train,distribution="gaussian",n.trees = 100) 

pr.gbm   <- predict(fit.gbm,tp2_test,n.trees=fit.gbm$n.trees)

confusionMatrix(tp2_test$Clase,pr.gbm) #Accuracy : 0.41

'                     Class: 1 Class: 2 Class: 3 Class: 4 Class: 5
Sensitivity          0.272727 0.304348  0.53448   0.4346   0.3879
Specificity          0.822128 0.879062  0.86156   0.8562   0.8373
Pos Pred Value       0.008850 0.029915  0.10839   0.6813   0.7347
Neg Pred Value       0.994875 0.990396  0.98327   0.6817   0.5408
Prevalence           0.005789 0.012105  0.03053   0.4142   0.5374
Detection Rate       0.001579 0.003684  0.01632   0.1800   0.2084
Detection Prevalence 0.178421 0.123158  0.15053   0.2642   0.2837
Balanced Accuracy    0.547428 0.591705  0.69802   0.6454   0.6126'


#cubist
model_bagging <- Clase ~ sup_tot_m2+sup_cub_m2+piso+sup_cub_dis+cluster+sup_tot_dis

fit.cub <-  cubist(model,data=tp2_train) 

pr.cub   <- predict(fit.gbm,tp2_test,n.trees=fit.gbm$n.trees)

confusionMatrix(tp2_test$Clase,pr.gbm) #Accuracy : 0.41

'                     Class: 1 Class: 2 Class: 3 Class: 4 Class: 5
Sensitivity          0.272727 0.304348  0.53448   0.4346   0.3879
Specificity          0.822128 0.879062  0.86156   0.8562   0.8373
Pos Pred Value       0.008850 0.029915  0.10839   0.6813   0.7347
Neg Pred Value       0.994875 0.990396  0.98327   0.6817   0.5408
Prevalence           0.005789 0.012105  0.03053   0.4142   0.5374
Detection Rate       0.001579 0.003684  0.01632   0.1800   0.2084
Detection Prevalence 0.178421 0.123158  0.15053   0.2642   0.2837
Balanced Accuracy    0.547428 0.591705  0.69802   0.6454   0.6126'

#bayes ingenuo
fit.bayes <- naiveBayes(model,data=tp2_train) 

pr.bayes  <- predict(fit.bayes,tp2_test,na.action=na.omit)

confusionMatrix(tp2_test$Clase,pr.bayes) #Accuracy : 0.2135

'                    Class: 1 Class: 2 Class: 3 Class: 4 Class: 5
Sensitivity           0.32759   0.1956 0.272727  0.29389  0.17769
Specificity           0.77360   0.9182 0.802689  0.80622  0.79964
Pos Pred Value        0.10826   0.8760 0.010118  0.12685  0.07227
Neg Pred Value        0.93203   0.2787 0.993344  0.92259  0.91715
Prevalence            0.07741   0.7471 0.007341  0.08742  0.08075
Detection Rate        0.02536   0.1461 0.002002  0.02569  0.01435
Detection Prevalence  0.23423   0.1668 0.197865  0.20254  0.19853
Balanced Accuracy     0.55059   0.5569 0.537708  0.55005  0.48866'

# ----------------------------------------------------------------
# ensamble de los 4 modelos anteriores
ensamble          <- data.frame(rf.pred, pr.nn, pr.bag)
ensamble$uno <- rowSums(ensamble[, 1:3] == "1")
ensamble$dos <- rowSums(ensamble[, 1:3] == "2")
ensamble$tres <- rowSums(ensamble[, 1:3] == "3")
ensamble$cuatro <- rowSums(ensamble[, 1:3] == "4")
ensamble$cinco <- rowSums(ensamble[, 1:3] == "5")

ensamble$predict  <- ifelse(ensamble$uno>ensamble$dos & ensamble$uno>ensamble$tres & ensamble$uno>ensamble$cuatro & ensamble$uno>ensamble$cinco
                            ,"1",
                            ifelse(ensamble$dos>ensamble$uno & ensamble$dos>ensamble$tres & ensamble$dos>ensamble$cuatro & ensamble$dos>ensamble$cinco
                                   ,"2",
                                   ifelse(ensamble$tres>ensamble$uno & ensamble$tres>ensamble$dos & ensamble$tres>ensamble$cuatro & ensamble$tres>ensamble$cinco
                                          ,"3",
                                          ifelse(ensamble$cuatro>ensamble$uno & ensamble$cuatro>ensamble$tres & ensamble$cuatro>ensamble$dos & ensamble$cuatro>ensamble$cinco
                                                 ,"4",
                                                "5"))))

#bagging


Train        <- p[sample(3333,1e5,replace = T), ]
Iteraciones  <- 4  
n            <- nrow(Train)
inicio       <- Sys.time()
modelo       <- foreach(i=1:Iteraciones) %do% {
  muestra    <- sample(n, n, replace = T)  
  rpart(churn ~ .,data = Train[muestra, ])   
}  
(duracion <- Sys.time()-inicio)





# ----------------------------------------------------------------
# Matriz de Confusion
confusionMatrix(tp2_test$Clase,ensamble$predict)

# Accuracy
(accuracy <- sum(diag(conf.mat))/length(test) * 100)