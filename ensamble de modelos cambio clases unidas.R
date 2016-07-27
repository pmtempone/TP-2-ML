test_tp <- tp2.work
test_tp$clase_n <- factor(ifelse(test_tp$Clase %in% c("1","2"),"1_2",ifelse(test_tp$Clase %in% c("5","4"),"4_5","3")))
                          
#primer modelo rf
set.seed(123)
intrain <- createDataPartition(y=test_tp$clase_n,p=0.80,list = FALSE)
tp2_train <- test_tp[intrain,]
tp2_test <- test_tp[-intrain,]

model <- clase_n ~ tipoprop+sup_tot_m2+sup_cub_m2+sup_cub_dis+cant_amb+cluster
rf.fit <- randomForest(model, data = tp2_train, ntree=200, na.action = na.omit)
rf.fit$confusion

rf.pred <- predict(rf.fit,tp2_test)
confusionMatrix(tp2_test$clase_n,rf.pred) #Accuracy : 0.6273

'                     Class: 1_2 Class: 3 Class: 4_5
Sensitivity             0.69455  0.56989     0.6190
Specificity             0.74539  0.87440     0.8179
Pos Pred Value          0.30856  0.18467     0.9362
Neg Pred Value          0.93717  0.97603     0.3322
Prevalence              0.14059  0.04755     0.8119
Detection Rate          0.09765  0.02710     0.5026
Detection Prevalence    0.31646  0.14673     0.5368
Balanced Accuracy       0.71997  0.72214     0.7185'

#modelo j48

tp2.complete$geoname_num <- as.numeric(gsub(",",".",tp2.complete[,c("geoname_num")]))

model_j48 <- Clase ~ tipoprop+sup_tot+cant_amb+cluster
arbol.fit <- J48(model,data=tp2_train)
arbol.pred <- predict(arbol.fit,tp2_test)
confusionMatrix(tp2_test$clase_n,arbol.pred) #Accuracy : 0.4154 clase 1 0.4715

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

model_svm <- clase_n ~ tipoprop+cant_amb+cluster+piso 
modelo.svm    <- svm(model_svm,data=tp2_train)
predict.svm   <- predict(modelo.svm,tp2_test)

confusionMatrix(tp2_test$clase_n,predict.svm) #Accuracy : 0.3263 (clase 3 39%)

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

confusionMatrix(tp2_test$clase_n,factor(pr.nn)) #Accuracy : 0.41 (clase 2 52%)

'                    Class: 1_2 Class: 3 Class: 4_5
Sensitivity             0.77215  0.64865     0.5658
Specificity             0.70272  0.86295     0.9224
Pos Pred Value          0.09855  0.08362     0.9914
Neg Pred Value          0.98654  0.99221     0.1181
Prevalence              0.04039  0.01892     0.9407
Detection Rate          0.03119  0.01227     0.5322
Detection Prevalence    0.31646  0.14673     0.5368
Balanced Accuracy       0.73743  0.75580     0.7441'

#bagging
model_bagging <- Clase ~ sup_tot_m2+sup_cub_m2+piso+sup_cub_dis+cluster+sup_tot_dis

fit.bag <- bagging(model,data=tp2_train) 

pr.bag   <- predict(fit.bag,tp2_test)

confusionMatrix(tp2_test$clase_n,pr.bag) #Accuracy : 0.3534

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

confusionMatrix(tp2_test$clase_n,pr.c5) #Accuracy : 0.41

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