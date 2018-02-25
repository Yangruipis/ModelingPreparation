library(devtools)
library(wordVectors)
library(showtext)
library(ggplot2)


df <- read.csv("./datafile/word_vector.txt",header = TRUE)

a <- cosineDist(df[1],df[2])
a <- cov(df,df)
r<- matrix(1-a,nrow=40,dimnames=list(colnames(df),colnames(df)))
hc <- hclust(as.dist(r),method = "complete")

pdf(file = 'fujia_7.pdf', width = 12, height = 8)
plot(hc)
dev.off()
## dist
#euclidean                欧几里德距离，就是平方再开方。
#maximum                切比雪夫距离
#manhattan            绝对值距离
#canberra                Lance 距离
#minkowski            明科夫斯基距离，使用时要指定p值
#binary                    定性变量距离.

##method 
#single            最短距离法
#complete        最长距离法
#median        中间距离法
#mcquitty        相似法
#average        类平均法
#centroid        重心法
#ward            离差平方和法

install.packages("ape",repos = 'http://mirrors.ustc.edu.cn/CRAN/')
library(ape)
plot(as.phylo(hc), type = "fan")
plot(as.phylo(hc), type = "fan", tip.color = hsv(runif(15, 0.65, 
                                                       0.95), 1, 1, 0.7), edge.color = hsv(runif(10, 0.65, 0.75), 1, 1, 0.7), edge.width = runif(20, 
                                                                                                                                                 0.5, 3), use.edge.length = TRUE, col = "gray80")
