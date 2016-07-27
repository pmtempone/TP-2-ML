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

test_tp <- tp2.work
test_tp$clase_n <- factor(ifelse(test_tp$Clase %in% c("1","2"),"1_2",ifelse(test_tp$Clase %in% c("5","4"),"4_5","3")))

test_tp$clase_n2 <- factor(ifelse(test_tp$Clase %in% c("1","2","3"),"1_2_3","4_5"))

cuatcin <- droplevels(subset(test_tp,test_tp$clase_n2=='4_5'))

#primer modelo rf
set.seed(123)
intrain <- createDataPartition(y=test_tp$clase_n2,p=0.80,list = FALSE)
tp2_train <- test_tp[intrain,]
tp2_test <- test_tp[-intrain,]

model <- clase_n2 ~ tipoprop+sup_tot_m2+sup_cub_m2+sup_cub_dis+cant_amb+cluster
rf.fit <- randomForest(model, data = tp2_train, ntree=200, na.action = na.omit)
rf.fit$confusion

rf.pred <- predict(rf.fit,tp2_test)
confusionMatrix(tp2_test$clase_n2,rf.pred) #Accuracy : 0.6783

'  Sensitivity : 0.7404         
            Specificity : 0.6524         
         Pos Pred Value : 0.4696         
         Neg Pred Value : 0.8581         
             Prevalence : 0.2936         
         Detection Rate : 0.2174         
   Detection Prevalence : 0.4629         
      Balanced Accuracy : 0.6964         
                                         
       'Positive' Class : 1_2_3  '

#modelo j48

tp2.complete$geoname_num <- as.numeric(gsub(",",".",tp2.complete[,c("geoname_num")]))

model_j48 <- Clase ~ tipoprop+sup_tot+cant_amb+cluster
arbol.fit <- J48(model,data=tp2_train)
arbol.pred <- predict(arbol.fit,tp2_test)
confusionMatrix(tp2_test$clase_n2,arbol.pred) #Accuracy : 0.5285 clase 1 0.4715

'   Sensitivity : 0.7683          
            Specificity : 0.4539          
         Pos Pred Value : 0.3046          
         Neg Pred Value : 0.8628          
             Prevalence : 0.2374          
         Detection Rate : 0.1824          
   Detection Prevalence : 0.5989          
      Balanced Accuracy : 0.6111          
                                          
        Class : 1_2_3  '

#modelo SVM

model_svm <- clase_n2 ~ tipoprop+cant_amb+cluster+piso 
modelo.svm    <- svm(model,data=tp2_train)
predict.svm   <- predict(modelo.svm,tp2_test)

confusionMatrix(tp2_test$clase_n2,predict.svm) #Accuracy : 0.3263 (clase 3 39%)

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

set.seed(123)
fit.nnet <- nnet(model,data=tp2_train,size=10,decay=0.0001, maxit=500) 

pr.nn   <- predict(fit.nnet,tp2_test,type="class")

confusionMatrix(tp2_test$clase_n2,factor(pr.nn)) #Accuracy : 0.5801 (clase 2 52%)

'                    ensitivity : 0.91176         
            Specificity : 0.56179         
         Pos Pred Value : 0.10276         
         Neg Pred Value : 0.99143         
             Prevalence : 0.05217         
         Detection Rate : 0.04757         
   Detection Prevalence : 0.46292         
      Balanced Accuracy : 0.73678  '

#bagging
model_bagging <- Clase ~ sup_tot_m2+sup_cub_m2+piso+sup_cub_dis+cluster+sup_tot_dis

fit.bag <- bagging(model,data=tp2_train) 

pr.bag   <- predict(fit.bag,tp2_test)

confusionMatrix(tp2_test$clase_n2,pr.bag) #Accuracy : 0.5615

'  Sensitivity : 0.7169          
            Specificity : 0.4704          
         Pos Pred Value : 0.4427          
         Neg Pred Value : 0.7390          
             Prevalence : 0.3698          
         Detection Rate : 0.2651          
   Detection Prevalence : 0.5989          
      Balanced Accuracy : 0.5936  '

#gbm no funciona
model_bagging <- Clase ~ sup_tot_m2+sup_cub_m2+piso+sup_cub_dis+cluster+sup_tot_dis

fit.gbm <- gbm(model,data=tp2_train,distribution="gaussian",n.trees = 100) 

pr.gbm   <- predict(fit.gbm,tp2_test,n.trees=fit.gbm$n.trees)

confusionMatrix(tp2_test$clase_n,pr.gbm) #Accuracy : 0.41

'                     Class: 1 Class: 2 Class: 3 Class: 4 Class: 5
Sensitivity          0.272727 0.304348  0.53448   0.4346   0.3879
Specificity          0.822128 0.879062  0.86156   0.8562   0.8373
Pos Pred Value       0.008850 0.029915  0.10839   0.6813   0.7347
Neg Pred Value       0.994875 0.990396  0.98327   0.6817   0.5408
Prevalence           0.005789 0.012105  0.03053   0.4142   0.5374
Detection Rate       0.001579 0.003684  0.01632   0.1800   0.2084
Detection Prevalence 0.178421 0.123158  0.15053   0.2642   0.2837
Balanced Accuracy    0.547428 0.591705  0.69802   0.6454   0.6126'


#C5
model_bagging <- Clase ~ sup_tot_m2+sup_cub_m2+piso+sup_cub_dis+cluster+sup_tot_dis

fit.c5 <-  C5.0(model,data=tp2_train)

pr.c5   <- predict(fit.c5,tp2_test)

confusionMatrix(tp2_test$clase_n2,pr.c5) #Accuracy : 0.6689

'    Sensitivity : 0.6863          
            Specificity : 0.6244          
         Pos Pred Value : 0.8235          
         Neg Pred Value : 0.4381          
             Prevalence : 0.7186          
         Detection Rate : 0.4932          
   Detection Prevalence : 0.5989          
      Balanced Accuracy : 0.6554  '

#bayes ingenuo
fit.bayes <- naiveBayes(model,data=tp2_train) 

pr.bayes  <- predict(fit.bayes,tp2_test,na.action=na.omit)

confusionMatrix(tp2_test$clase_n,pr.bayes) #Accuracy : 0.2135

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
ensamble          <- data.frame(rf.pred, pr.nn, pr.c5)
ensamble$

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