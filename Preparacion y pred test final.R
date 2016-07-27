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

#preparacion test

tp2.clasificacion <- read.csv("~/Dropbox/TP Aprendizaje Automatico/TP2/Originales/tp2-clasificacion.csv")

test_tp <- tp2.clasificacion
test_tp$cant_amb 
test_tp$cant_amb <- ifelse(test_tp$cant_amb>5,">6",test_tp$cant_amb)
#tp2.work$cant_amb <- ifelse(tp2.work$cant_amb %in% c(">6","6"),">=6",tp2.work$cant_amb)
test_tp$cant_amb <- ifelse(test_tp$cant_amb %in% c("2","3"),"2-3",test_tp$cant_amb)
test_tp$cant_amb <- ifelse(test_tp$cant_amb %in% c("4","5"),"4-5",test_tp$cant_amb)
test_tp$cant_amb <- factor(test_tp$cant_amb)


#primer modelo rf
set.seed(123)

model <- clase_n ~ tipoprop+sup_tot_m2+sup_cub_m2+sup_cub_dis+cant_amb+cluster

rf.pred <- predict(rf.fit,test_tp)

test_tp$clase_n <- rf.pred


#ensamble de modelos

par1 <- droplevels(subset(test_tp,test_tp$clase_n=='4_5'))
par2 <- droplevels(subset(test_tp,test_tp$clase_n=='1_2_3'))



#modelo 1,2,3

#random forest
set.seed(123)

model2 <- Clase ~ tipoprop+sup_tot_m2+sup_cub_m2+sup_cub_dis+cant_amb+cluster
rf.fitpart3 <- randomForest(model2, data = tp2_train3, ntree=200, na.action = na.omit)
rf.fitpart3$confusion

rf.pred3 <- predict(rf.fitpart3,tp2_test3)
confusionMatrix(tp2_test3$Clase,rf.pred3)

#red neuronal

set.seed(123)
fit.nnet2 <- nnet(model2,data=tp2_train3,size=10,decay=0.0001, maxit=500) 

pr.nn2   <- predict(fit.nnet2,tp2_test3,type="class")

confusionMatrix(tp2_test3$Clase,factor(pr.nn2)) #Accuracy : 0.41 (clase 2 52%)

#C5

fit.c5.2 <-  C5.0(model2,data=tp2_train3)

pr.c5.2   <- predict(fit.c5.2,tp2_test3)

confusionMatrix(tp2_test3$Clase,pr.c5.2)


#votacion

ensamble2  <- data.frame(rf.pred3, pr.nn2, pr.c5.2)
ensamble2$uno <- rowSums(ensamble2[, 1:3] == "1")
ensamble2$dos <- rowSums(ensamble2[, 1:3] == "2")
ensamble2$tres <- rowSums(ensamble2[, 1:3] == "3")


ensamble2$predict  <- ifelse(ensamble2$uno>ensamble2$dos & ensamble2$uno>ensamble2$tres,"1",
                             ifelse(ensamble2$dos>ensamble2$uno & ensamble2$dos>ensamble2$tres,"2",
                                    ifelse(ensamble2$tres>ensamble2$uno & ensamble2$dos<ensamble2$tres,"3",rf.pred3)))

# Matriz de Confusion
confusionMatrix(tp2_test3$Clase,ensamble2$predict)


#modelo 4,5

#random forest
set.seed(123)
intrain <- createDataPartition(y=par1$Clase,p=0.80,list = FALSE)
tp2_train2 <- par1[intrain,]
tp2_test2 <- par1[-intrain,]

model2 <- Clase ~ tipoprop+sup_tot_m2+sup_cub_m2+sup_cub_dis+cant_amb+cluster
rf.fitpart2 <- randomForest(model2, data = tp2_train2, ntree=200, na.action = na.omit)
rf.fitpart2$confusion

rf.pred2 <- predict(rf.fitpart2,tp2_test2)
confusionMatrix(tp2_test2$Clase,rf.pred2)

#red neuronal

set.seed(123)
fit.nnet <- nnet(model2,data=tp2_train2,size=10,decay=0.0001, maxit=500) 

pr.nn   <- predict(fit.nnet,tp2_test2,type="class")

confusionMatrix(tp2_test2$Clase,factor(pr.nn)) #Accuracy : 0.41 (clase 2 52%)

#C5

fit.c5 <-  C5.0(model2,data=tp2_train2)

pr.c5   <- predict(fit.c5,tp2_test2)

confusionMatrix(tp2_test2$Clase,pr.c5)


#votacion

ensamble1  <- data.frame(rf.pred2, pr.nn, pr.c5)
ensamble1$cuatro <- rowSums(ensamble1[, 1:3] == "4")
ensamble1$cinco <- rowSums(ensamble1[, 1:3] == "5")

ensamble1$predict  <- ifelse(ensamble1$cuatro>ensamble1$cinco,"4","5")

# Matriz de Confusion
confusionMatrix(tp2_test2$Clase,ensamble1$predict)    