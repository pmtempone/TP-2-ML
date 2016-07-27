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

rf.pred <- predict(rf.fit,tp2_test)
tp2_test$pred <- rf.pred

#ensamble de modelos

par1 <- droplevels(subset(tp2_test,tp2_test$pred=='4_5'))
par2 <- droplevels(subset(tp2_test,tp2_test$pred=='1_2_3'))

#modelo 1,2,3

rf.pred3 <- predict(rf.fitpart3,tp2_test[tp2_test$pred=='5_2_3',])
rf.pred4   <- predict(rf.fitpart4,tp2_test[tp2_test$pred=='5_2_3',])
pr.c5.2   <- predict(fit.c5.2,tp2_test[tp2_test$pred=='5_2_3',])

#votacion

ensamble2  <- data.frame(rf.pred3, rf.pred4, pr.c5.2)
ensamble2$uno <- rowSums(ensamble2[, 1:3] == "5")
ensamble2$dos <- rowSums(ensamble2[, 1:3] == "2")
ensamble2$tres <- rowSums(ensamble2[, 1:3] == "3")


ensamble2$predict  <- ifelse(ensamble2$uno>ensamble2$dos & ensamble2$uno>ensamble2$tres,"5",
                             ifelse(ensamble2$dos>ensamble2$uno & ensamble2$dos>ensamble2$tres,"2",
                                    ifelse(ensamble2$tres>ensamble2$uno & ensamble2$dos<ensamble2$tres,"3",rf.pred4)))


tp2_test$predict_clase[tp2_test$pred=='5_2_3'] <- ensamble2$predict

#modelo 4,5
rf.pred2 <- predict(rf.fitpart2,tp2_test[tp2_test$pred=='4_1',])
rf.pred21 <- predict(rf.fitpart21,tp2_test[tp2_test$pred=='4_1',])
pr.c5   <- predict(fit.c5,tp2_test[tp2_test$pred=='4_1',])

#votacion

ensamble1  <- data.frame(rf.pred2, rf.pred21, pr.c5)
ensamble1$cuatro <- rowSums(ensamble1[, 1:3] == "4")
ensamble1$cinco <- rowSums(ensamble1[, 1:3] == "1")

ensamble1$predict  <- ifelse(ensamble1$cuatro>ensamble1$cinco,"4","1")


tp2_test$predict_clase[tp2_test$pred=='4_1'] <- ensamble1$predict


# Matriz de Confusion
confusionMatrix(tp2_test$Clase,tp2_test$predict_clase)
