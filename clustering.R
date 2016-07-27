testmice <- mice(tp2.complete,method = "pmm")
tp2.complete <- complete(testmice)

# K-Means Cluster Analysis
fit <- kmeans(tp2.complete[,c("sup_tot_m2","geoname_num")], 4) # 5 cluster solution
# get cluster means
aggregate(tp2.complete,by=list(fit$cluster),FUN=mean)
# append cluster assignment
mydata <- data.frame(tp2.complete, fit$cluster) 
