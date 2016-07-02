install.packages("qdap")
library(qdap)
freqs <- t(wfm(tp2.work$tit, 1:nrow(tp2.work)))
data.frame(tp2.work, freqs, check.names = FALSE)

freqsdes <- t(wfm(tp2.work$des[1:1000], 1:1000))
ords <- rev(sort(colSums(dtm2)))[1:9]
top9 <- freqsdes[, names(ords)]                #grab those columns from freqs  
tp2_train_prueba <- data.frame(tp2.work, top9, check.names = FALSE) #put it together

freqs <- t(wfm(tp2.work$des, 1:nrow(tp2.work)))
ords <- rev(sort(colSums(freqs)))[1:9]      #top 9 words
top9 <- freqs[, names(ords)]                #grab those columns from freqs  
tp2_train_v2 <- data.frame(tp2.work, top9, check.names = FALSE) #put it together
