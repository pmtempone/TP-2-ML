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
#test_tp$clase_n <- factor(ifelse(test_tp$Clase %in% c("1","2"),"1_2",ifelse(test_tp$Clase %in% c("5","4"),"4_5","3")))
test_tp$clase_n <- factor(ifelse(test_tp$Clase %in% c("5","2","3"),"5_2_3","4_1"))

test_tp <- work

#primer modelo rf
set.seed(123)
intrain <- createDataPartition(y=test_tp$clase_n,p=0.80,list = FALSE)
tp2_train <- test_tp[intrain,]
tp2_test <- test_tp[-intrain,]

model <- clase_n ~ tipoprop+sup_tot_m2+sup_cub_m2+sup_cub_dis+cant_amb+cluster
model <- clase_n ~ tipoprop+cub_m2+amb+cluster+fl_piscina+fl_solarium+fl_hotel+fl_cochera+fl_solarium


rf.fit <- randomForest(model, data = tp2_train, ntree=200, na.action = na.omit)
rf.fit$confusion

rf.pred <- predict(rf.fit,tp2_test)
confusionMatrix(tp2_test$clase_n,rf.pred)

tp2_test$pred <- rf.pred


#ensamble de modelos

par1 <- droplevels(subset(test_tp,test_tp$clase_n=='4_1'))
par2 <- droplevels(subset(test_tp,test_tp$clase_n=='5_2_3'))



#modelo 1,2,3

  #random forest
set.seed(123)
intrain <- createDataPartition(y=par2$Clase,p=0.80,list = FALSE)
tp2_train3 <- par2[intrain,]
tp2_test3 <- par2[-intrain,]

model2 <- Clase ~ tipoprop+sup_tot_m2+sup_cub_m2+sup_cub_dis+cant_amb+cluster
model2 <- Clase ~ tipoprop+cub_m2+amb+cluster+fl_piscina+fl_solarium+fl_hotel+fl_cochera+fl_solarium


rf.fitpart3 <- randomForest(model2, data = tp2_train3, ntree=200, na.action = na.omit)
rf.fitpart3$confusion

rf.pred3 <- predict(rf.fitpart3,tp2_test3)
confusionMatrix(tp2_test3$Clase,rf.pred3)

#random forest 2
set.seed(123)
intrain <- createDataPartition(y=par2$Clase,p=0.80,list = FALSE)
tp2_train3 <- par2[intrain,]
tp2_test3 <- par2[-intrain,]

model2 <- Clase ~ tipoprop+sup_tot_m2+sup_cub_m2+sup_cub_dis+cant_amb+cluster
model2 <- Clase ~ tipoprop+cub_m2+amb+cluster+fl_piscina+fl_solarium+fl_hotel+fl_cochera+fl_solarium


rf.fitpart4 <- randomForest(model2, data = tp2_train3, ntree=200, proximity=TRUE ,na.action = na.omit)
rf.fitpart4$confusion

rf.pred4 <- predict(rf.fitpart4,tp2_test3)
confusionMatrix(tp2_test3$Clase,rf.pred4)

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

ensamble2  <- data.frame(rf.pred3, rf.pred4, pr.c5.2)
ensamble2$cinco <- rowSums(ensamble2[, 1:3] == "5")
ensamble2$dos <- rowSums(ensamble2[, 1:3] == "2")
ensamble2$tres <- rowSums(ensamble2[, 1:3] == "3")


ensamble2$predict  <- ifelse(ensamble2$cinco>ensamble2$dos & ensamble2$cinco>ensamble2$tres,"5",
                      ifelse(ensamble2$dos>ensamble2$cinco & ensamble2$dos>ensamble2$tres,"2",
                      ifelse(ensamble2$tres>ensamble2$cinco & ensamble2$dos<ensamble2$tres,"3",pr.c5.2)))

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


#random forest 2
set.seed(123)
intrain <- createDataPartition(y=par1$Clase,p=0.80,list = FALSE)
tp2_train2 <- par1[intrain,]
tp2_test2 <- par1[-intrain,]

model2 <- Clase ~ tipoprop+sup_tot_m2+sup_cub_m2+sup_cub_dis+cant_amb+cluster
rf.fitpart21 <- randomForest(model2, data = tp2_train2, ntree=200,proximity=TRUE, na.action = na.omit)
rf.fitpart21$confusion

rf.pred21 <- predict(rf.fitpart21,tp2_test2)
confusionMatrix(tp2_test2$Clase,rf.pred21)
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

ensamble1  <- data.frame(rf.pred2, rf.pred21, pr.c5)
ensamble1$cuatro <- rowSums(ensamble1[, 1:3] == "4")
ensamble1$cinco <- rowSums(ensamble1[, 1:3] == "1")

ensamble1$predict  <- ifelse(ensamble1$cuatro>ensamble1$cinco,"4","1")

# Matriz de Confusion
confusionMatrix(tp2_test2$Clase,ensamble1$predict)                           
