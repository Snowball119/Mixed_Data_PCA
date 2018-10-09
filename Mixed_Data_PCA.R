# House price prediction
# Xueji Wang 
dataIn = read.csv("/Users/xuejiwang/Documents/3_My_UNC_Charlotte/2018Spring/STAT_7133/Project/Data_HousePrice/train.csv")
str(dataIn)
head(dataIn)
dim(dataIn)
data = dataIn

names<-as.data.frame(names(data))
names
#Process Missing Columns(with no observations at all)
ncol=rep(nrow(data) ,each=ncol(data))
missingdata=as.data.frame(cbind(colnames=names(data),ncol,
                                nmsg=as.integer(as.character(as.vector(apply(data, 2, function(x) length(which(is.na(x)))))))))
# for each colunm how many obeservations are missing
missingdata$nmsg
missingdata$nmsg=as.numeric(levels(missingdata$nmsg))[missingdata$nmsg]
missingdata$nmsg
missingdata=cbind(missingdata,percmissing=as.integer(missingdata$nmsg/ncol*100))
missingdata$percmissing
drops=as.character(subset(missingdata,missingdata$percmissing>0)[,1])
length(drops)
drops = c(drops, "Id")
length(drops)
table(drops)
#--------------------------------------------------------------------------------------------
# Remove columns with missing values from dataset
dim(data)
data=data[,!(names(data) %in% drops)]
dim(data)
head(data)

#--------------------------------------------------------------------------------
# divide price into groups
Y = data$SalePrice
summary(Y)

cutN <- function(X , n = 4){
  cut(
    X ,
    include.lowest = TRUE ,
    breaks = quantile(
      X , 
      probs = (0:n)/n ,
      na.rm = TRUE ))}
NewY = cutN(Y,n=4)
table(NewY)

priceGroup <- function(X , n = 4){cut(
  X,
  breaks = quantile(X, c(0, 0.25, 0.5, 0.75, 1)),
  labels = c("1", "2", "3", "4"),
  right  = FALSE,
  include.lowest = TRUE
)}
priceG = priceGroup(Y,n=4)
table(priceG)
data = cbind(data,priceG)
dim(data)

##-----------------------------------------------------------------------------------------------
# Feature Engineering
TotBath = data$FullBath + data$HalfBath + data$BsmtHalfBath + data$BsmtFullBath
drops = c("FullBath","HalfBath","BsmtHalfBath","BsmtFullBath")
data = data[,!(names(data) %in% drops)]
data = cbind.data.frame(data,TotBath)
dim(data)
str(data)
attach(data)


# Landscape
#data$LotShape
#table(data$LotShape)
#table(data$LandContour)
#Landscape = as.factor(contrasts(data$LotShape)*contrasts(data$LandContour))




#-----------------------------------------------------------------------------------------------
#Check and output the datatype
getNumericColumns<-function(t){
  tn = sapply(t,function(x){is.numeric(x)})
  return(names(tn)[which(tn)])
}
getFactorColumns<-function(t){
  tn = sapply(t,function(x){is.factor(x)})
  return(names(tn)[which(tn)])
}
dim(data[getFactorColumns(data)]) # there is 30 factor predictors
dim(data[getNumericColumns(data)]) # 34 numeric predictors

data.factor = data[getFactorColumns(data)]
str(data.factor)
data.num = data[getNumericColumns(data)]
str(data.num)


##############################################################################################################
# Exploratory data analysis
library(ggplot2)
Y = data$SalePrice



hist(log(Y))
summary(log(Y))
shapiro.test(log(Y))

options(scipen=10000)
ggplot(data=data, aes(SalePrice)) + 
  geom_histogram(breaks=seq(30000, 800000, by = 10000), 
                 col="blue", 
                 fill="blue", 
                 alpha = .2) + 
  labs(title="Histogram for SalePrice") 

# Log SalePrice
options(scipen=10000)
data1 = cbind.data.frame(data,log(SalePrice))
ggplot(data=data1, aes(log(SalePrice))) + 
  geom_histogram(col= 4, 
                 fill= 4, 
                 alpha = .7) + 
  labs(title="Histogram for SalePrice") 


# 95% Confidence interval for Mean of Sale Price
n = length(Y)
Y_bar = mean(Y)
CI.low = Y_bar - qt(0.975,n-1)*sqrt(var(Y))
CI.high = Y_bar + qt(0.975,n-1)*sqrt(var(Y))
CI.low;CI.high


library(ggthemes)
# Boxplot
g <- ggplot(data, aes(Neighborhood, SalePrice))
g + geom_boxplot(aes(fill=factor(Neighborhood))) + 
  theme(axis.text.x = element_text(angle=65, vjust=0.6)) + 
  labs(title="Box plot", 
       subtitle="Sale Price grouped by Neighborhood",
       caption="Source:Kaggle House Price Competition",
       x="Neighborhood",
       y="Sale Price")


g <- ggplot(data, aes(OverallQual, SalePrice))
g + geom_boxplot(aes(fill=factor(OverallQual))) + 
  theme(axis.text.x = element_text(angle=65, vjust=0.6)) + 
  labs(title="Box plot", 
       subtitle="Sale Price grouped by OverallQual",
       caption="Source:Kaggle House Price Competition",
       x="OverallQual",
       y="Sale Price")


g <- ggplot(data, aes(ExterQual, SalePrice))
g + geom_boxplot(aes(fill=factor(ExterQual))) + 
  theme(axis.text.x = element_text(angle=65, vjust=0.6)) + 
  labs(title="Box plot", 
       subtitle="Sale Price grouped by KitchenQual",
       caption="Source:Kaggle House Price Competition",
       x="ExterQual",
       y="Sale Price")


g <- ggplot(data, aes(KitchenQual, SalePrice))
g + geom_boxplot(aes(fill=factor(KitchenQual))) + 
  theme(axis.text.x = element_text(angle=65, vjust=0.6)) + 
  labs(title="Box plot", 
       subtitle="Sale Price grouped by KitchenQual",
       caption="Source:Kaggle House Price Competition",
       x="KitchenQual",
       y="Sale Price")


# Scatter Plot
gg <- ggplot(data, aes(x=GrLivArea, y=SalePrice)) + 
  geom_point(aes(col= ExterQual,shape = KitchenQual)) + 
  geom_smooth(method="loess", se=F) + 
  labs(subtitle="Living Area Vs Population", 
       y="Sale Price", 
       x="Ground Living Area SF", 
       title="Scatterplot", 
       caption = "Source:House Price")
plot(gg)
ExterQual
plot(SalePrice~ExterQual)



gg <- ggplot(data, aes(x=GrLivArea, y=SalePrice)) + 
  geom_point(aes(col= Neighborhood, shape = KitchenQual )) + 
  geom_smooth(method="loess", se=F) + 
  labs(subtitle="Living Area Vs Population", 
       y="Sale Price", 
       x="Ground Living Area SF", 
       title="Scatterplot", 
       caption = "Source:House Price")
plot(gg)


##############################################################################
#Split the dataset into training and test set
train<-floor(0.75*nrow(data))
set.seed(123)
trainIndex<-sample(seq_len(nrow(data)),size=train)
testIndex = seq_len(nrow(data))[-trainIndex]
length(trainIndex)
data_train<-data[trainIndex,]
data_test<-data[testIndex,]
dim(data_train)
dim(data_test)

###########################################################################
# PCA for discrete data
head(data.factor)
dim(data.factor)
mca <- PCAmix(X.quali=data.factor[,-30],rename.level=TRUE)
mca$eig
tmp = mca$quali
class(tmp)
dim1 = tmp$contrib[,1]
sort(dim1,decreasing = T)
plot(mca,choice="ind",main="Scores")
plot(mca,choice="sqload",main="Correlation ratios")
plot(mca,choice="levels",main="Levels")
mca$levels$coord
head(data.num)
dim(data.num)

princomp(data.num)
rho = cor(data.num)
rho


# Not scaled
pca <-PCAmix(X.quanti= data.num[,-32],X.quali = data.factor[,-30], rename.level = TRUE)
#Scaled numeric data
# Not differed from 
pca1 <-PCAmix(X.quanti= scale(data.num[,-32]),X.quali = data.factor[,-30], rename.level = TRUE)

pca$quanti
pca$quali.eta2
summary(pca)
pca$coef

pca$eig
pca1$eig

loading = pca$sqload
loading
loading1 = pca1$sqload
loading1
sort(loading[,1],decreasing = TRUE)[1:20]

sort(loading[,1],decreasing = TRUE)[1:20]
sort(loading[,2],decreasing = TRUE)[1:20]
sort(loading[,3],decreasing = TRUE)[1:20]
sort(loading[,4],decreasing = TRUE)[1:20]
sort(loading[,5],decreasing = TRUE)[1:20]
cat(names(sort(loading[,1],decreasing = TRUE)[1:20]), sep = ', ',"\n")
cat(names(sort(loading[,2],decreasing = TRUE)[1:20]), sep = ', ',"\n")
cat(names(sort(loading[,3],decreasing = TRUE)[1:20]), sep = ', ',"\n")
cat(names(sort(loading[,4],decreasing = TRUE)[1:20]), sep = ', ',"\n")
cat(names(sort(loading[,5],decreasing = TRUE)[1:20]), sep = ', ',"\n")

tmp1 = unique(names(sort(loading[,1],decreasing = TRUE)[1:20]),
       names(sort(loading[,2],decreasing = TRUE)[1:20]))
tmp1
tmp2 = unique(tmp1,names(sort(loading[,3],decreasing = TRUE)[1:20]))
cat(tmp2,sep = ', ',"\n")




# Scree Plot

pc.var = pca$eig[,1]
pve=pc.var/sum(pc.var)
pve
plot(pc.var[1:20], xlab="Principal Component Number", ylab="Variance",
     type="b",main = "Scree Plot",col = 4)

dev.off()

# Non-parametric MANOVA


######################################################
# Classification

# Fisher Discriminant
library(MASS)

#lda.result = lda(priceG ~ GrLivArea+ OverallQual+ Neighborhood+ TotalBsmtSF+ X1stFlrSF+ GarageArea+ YearBuilt+ LotArea+ YearRemodAdd+ TotBath+ GarageCars+ BsmtFinSF1+ ExterQual+ 
                  #X2ndFlrSF+ KitchenQual+ Exterior1st+ BsmtUnfSF+ TotRmsAbvGrd+ Exterior2nd+ OpenPorchSF,data = data_train[,-60])

#lda.result1 = lda(priceG ~ ., data = data_train[,-60])


lda.model = lda(priceG ~ OverallQual+ YearBuilt+Foundation + ExterQual+  TotalBsmtSF+ 
                  GrLivArea  +  X1stFlrSF  + GarageArea  + GarageCars +   X2ndFlrSF  +    BsmtUnfSF+   BsmtFinSF1, data = data_train[,-60])




pred.lda.test = predict(lda.model,data_test[,-60])$class
length(pred.lda.test)
tmp = table(predict(lda.model,data_test)$class,data_test$priceG)
1-sum(diag(tmp)/sum(tmp))
mis.rate(pred.lda.test,data_test$priceG)
my.mis.rate(pred.lda.test,data_test$priceG)

# Classification Tree
library(rpart)
tree.model <- rpart(priceG ~.,
             method="class", data=data_train[,-60])

printcp(tree.model) # display the results 
plotcp(tree.model) # visualize cross-validation results 
summary(tree.model) # detailed summary of splits

# plot tree 
plot(tree.model, uniform=TRUE, 
     main="Classification Tree for Price Group")
text(tree.model, use.n=TRUE, all=TRUE, cex=.8)

# prune the tree 
ptree.model<- prune(tree.model, cp = tree$cptable[which.min(tree$cptable[,"xerror"]),"CP"])

# plot the pruned tree 
plot(ptree.model, uniform=TRUE, 
     main="Pruned Classification Tree for Price Group")
text(ptree.model, use.n=TRUE, all=TRUE, cex=.8)
summary(ptree.model)
imp.tree = ptree.model$variable.importance
tmp = sort(imp.tree,decreasing = TRUE)
class(tmp)
cat(names(tmp), sep = ', ', "\n")

pred.ptree.test = predict(ptree.model,data_test,type = "class")
length(pred.tree)
tmp = table(pred.ptree.test,data_test$priceG)
tmp
1-sum(diag(tmp)/sum(tmp))
mis.rate(pred.ptree.test,data_test$priceG)
my.mis.rate(pred.ptree.test,data_test$priceG)

# Random Forest prediction of data
library(randomForest)
rf.model <- randomForest (priceG ~.,
                    data=data_train[,-60],na.action=na.exclude)

tmp = rf.model$confusion
tmp
1-sum(diag(tmp)/sum(tmp))
tmp = importance(rf)# importance of each predictor
tmp[,1]
class(tmp)
sort(importance(rf)[,1],decreasing = TRUE)

imp = sort(importance(rf)[,1],decreasing = TRUE)
tmp = names(imp[1:20])
print(tmp,sep = ",")
cat(tmp, sep = ', ', "\n")


pred.rf.test = predict(rf.model,data_test)
tmp = table(pred.rf.test,priceG[testIndex])
tmp
1-sum(diag(tmp)/sum(tmp))


my.mis.rate(pred.rf.test,data_test$priceG)


# Support Vector Machine
library(e1071)
svm.model = svm(priceG ~., data = data_train)
dim(na.omit(data_test))
pred.svm.test = predict(svm.model,na.omit(data_test))
pred.svm.test
length(pred.svm.test)
my.mis.rate(pred.svm.test,na.omit(data_test)$priceG)


# Multinomial Logistic regression
library(nnet)
logit.model <- multinom(priceG ~ ., data = na.omit(data_train[,-60]))
# Wald test
z <- summary(logit.model)$coefficients/summary(logit.model)$standard.errors
z
# 2-tailed z test
p <- (1 - pnorm(abs(z), 0, 1)) * 2
p
tmp = names(which(p[1,] <= 0.001))
tmp = names(which(p[1,] > 0.05))
cat(tmp,sep = ',', "\n")
tmp = names(which(p[2,] > 0.05))
cat(tmp,sep = ',', "\n")
tmp = names(which(p[3,] > 0.05))
cat(tmp,sep = ',', "\n")

dim(data_train)
dim(data_test)
pred.logit.test = predict(logit.model,data_test)
length(pred.logit)
tmp = table(pred.logit,priceG[-trainIndex])
1-sum(diag(tmp)/sum(tmp))
my.mis.rate(pred.logit.test, data_test$priceG)






###################
# Compare performance
set.seed(123)
out = matrix(0,100,5)
for(i in 1:100){
  train<-floor(0.75*nrow(data))
  trainIndex = sample(seq_len(nrow(data)),size=train)
  testIndex = seq_len(nrow(data))[-trainIndex]
  data_train = data[trainIndex,]
  data_test<-data[-trainIndex,]
  
  # LDA
  lda.model = lda(priceG ~ OverallQual+ YearBuilt+Foundation + ExterQual+  TotalBsmtSF+ 
                    GrLivArea  +  X1stFlrSF  + GarageArea  + GarageCars +   X2ndFlrSF  +    BsmtUnfSF+   BsmtFinSF1, data = data_train[,-60])
  pred.lda.test = predict(lda.model,data_test[,-60])$class
  
  # DT
  tree.model <- rpart(priceG ~.,
                      method="class", data=data_train[,-60])
  ptree.model<- prune(tree.model, cp= tree$cptable[which.min(tree$cptable[,"xerror"]),"CP"])
  pred.ptree.test = predict(ptree.model,data_test,type = "class")
  
  # Random Forest
  rf.model <- randomForest (priceG ~.,
                            data=data_train[,-60],na.action=na.exclude)
  pred.rf.test = predict(rf.model,data_test)
  
  # SVM
  svm.model = svm(priceG ~., data = data_train)
  pred.svm.test = predict(svm.model,na.omit(data_test))

  # Multinomial Logistic
  logit.model <- multinom(priceG ~ ., data = na.omit(data_train[,-60]))
  pred.logit.test = predict(logit.model,data_test)
  
  
  out[i,1] = my.mis.rate(pred.lda.test,data_test$priceG)
  out[i,2] = my.mis.rate(pred.ptree.test,data_test$priceG)
  out[i,3] = my.mis.rate(pred.rf.test,data_test$priceG)
  out[i,4] = my.mis.rate(pred.svm.test,na.omit(data_test)$priceG)
  out[i,5] = my.mis.rate(pred.logit.test,data_test$priceG)
  
}

out
out1 = as.data.frame(out[1:60,])
names(out1) = c("LDA","Decision Tree","Random Forest", "SVM","Multinomial Logistic")
out1
boxplot(out1,cex.axis = 0.7)

apply(out,2, mean)




# misclassification rate function
my.mis.rate = function(pred, true){
  tmp = table(pred,true)
  out =   1-sum(na.omit(diag(tmp))/sum(na.omit(tmp)))
  return(out)
}











