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

work$fl_piscina <- factor(work$fl_piscina)
work$fl_solarium <- factor(work$fl_solarium)
work$fl_hotel <- factor(work$fl_hotel)
work$fl_cochera <- factor(work$fl_cochera)

#correr primer modelo

rf.pred <- predict(rf.fit,work)

#segundo modelo

rf.pred2 <- predict(rf.fit2,work)

#tercer modelo

pr.c5.2   <- predict(fit.c5.2,work)

#ensamble

ensamble  <- data.frame(rf.pred, rf.pred2, pr.c5.2)


pr.c5.final   <- predict(fit.c5.final,ensamble)

#entrega
entrega <- cbind.data.frame(ident=work$id,clase=pr.c5.final)
#work$id <- as.numeric(gsub(pattern = '\\.',replacement = '',x = work$id))
write.csv(entrega,file="tp2_VazquezMorales_Stein_Tempone.csv",quote = TRUE,row.names = FALSE)
