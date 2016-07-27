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


#primer modelo rf


model <- Clase ~ tipoprop+sup_cub_m2+cant_amb2+cant_amb3+cant_amb4+cant_amb5+fl_piscina+fl_solarium+fl_hotel+fl_cochera+lat+lon+piso2+piso3
rf.fit <- randomForest(model, data = train, replace=FALSE,importance=TRUE,ntree=200, na.action = na.omit)
rf.fit$confusion

rf.pred <- predict(rf.fit,test)
confusionMatrix(test$Clase,rf.pred) 

#rf proximity

set.seed(123)
model <- Clase ~ tipoprop+sup_tot_m2+sup_cub_m2+piso+cant_amb+fl_cochera+cant_amb+lat+lon
rf.fit2 <- randomForest(model, data = train,importance=FALSE,replace=TRUE, ntree=200, proximity=TRUE,na.action = na.omit)
rf.fit2$confusion

rf.pred2 <- predict(rf.fit2,test)
confusionMatrix(test$Clase,rf.pred2)

#C5   

fit.c5.2 <-  C5.0(model,data=train,trials=50)

pr.c5.2   <- predict(fit.c5.2,test)

confusionMatrix(test$Clase,pr.c5.2)


#votacion

ensamble  <- data.frame(rf.pred, rf.pred2, pr.c5.2,test$Clase)

ensamble$uno <- rowSums(ensamble[, 1:3] == "1")
ensamble$dos <- rowSums(ensamble[, 1:3] == "2")
ensamble$tres <- rowSums(ensamble[, 1:3] == "3")
ensamble$cuatro <- rowSums(ensamble[, 1:3] == "4")
ensamble$cinco <- rowSums(ensamble[, 1:3] == "5")

#C5 final
set.seed(123)
intrain <- createDataPartition(y=ensamble$test.Clase,p=0.80,list = FALSE)
tp2_train <- ensamble[intrain,]
tp2_test <- ensamble[-intrain,]

model_final <- test.Clase ~ rf.pred+rf.pred2+pr.c5.2+uno+dos+tres+cuatro+cinco
fit.c5.final <-  C5.0(model_final,data=tp2_train,trials=50)

pr.c5.final   <- predict(fit.c5.final,tp2_test)

confusionMatrix(tp2_test$test.Clase,pr.c5.final)

entregapred <- predict(fit.c5.final,ensamble)

entrega <- cbind.data.frame(test$ident,entregapred)

# Matriz de Confusion

confusionMatrix(test$Clase,entregapred)

