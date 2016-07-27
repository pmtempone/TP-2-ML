tp2.work <- read.csv("C:/Users/Pablo/Google Drive/Maestria/Aprendizaje Automático/TP 2/tp2-work.csv", sep=";", stringsAsFactors=FALSE)

library(data.table)
library(mice)
library(VIM)

DT_Train <- as.data.table(tp2.work)
DT_Train$anio <- factor(DT_Train$anio)
sum(!complete.cases(DT_Train)) # 13228 casos incompletos

md.pattern(DT_Train) #Faltantes en sup_cub_m2, cant_amb, sup_tot_m2, piso  

table((DT_Train$Clase)) #5 Clases a predecir

aggr_plot <- aggr(DT_Train, col=c('navyblue','red'), numbers=TRUE, sortVars=TRUE, labels=names(DT_Train), cex.axis=.7, gap=3, ylab=c("Histogram of missing data","Pattern"))
