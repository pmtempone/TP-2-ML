library(dplyr)
library(reshape2)
library(pvclust)


barrios <- select(tp2.work,lugar,Clase) #+group_by(lugar,Clase)
barrios <- group_by(barrios,lugar)
barrios_sum <- melt(barrios,id.vars = 1)
barrios_clus <- dcast(barrios_sum, lugar + variable ~ value)
barrios_clus$variable <- NULL
rownames(barrios_clus) <- barrios_clus$lugar
barrios_clus$lugar <- NULL

d <- dist(as.matrix(barrios_clus))   # find distance matrix 
hc <- hclust(d)                # apply hirarchical clustering 
plot(hc)  
clusterCut <- cutree(hc, 4)
table(clusterCut, tp2.work$lugar)
View(data.frame(clusterCut))


clustercut_DF <- data.frame(vector=clusterCut,lugar=names(clusterCut))
tp2.work$cluster <- clustercut_DF[match(tp2.work$lugar, clustercut_DF$lugar), 'vector']
tp2.work$cluster <- factor(tp2.work$cluster)
