library(ggplot2)
#install.packages("showtext",repos = 'http://mirrors.ustc.edu.cn/CRAN/')
#install.packages("ggplot2",repos = 'http://mirrors.ustc.edu.cn/CRAN/')
#install.packages("sysfonts",repos = 'http://mirrors.ustc.edu.cn/CRAN/')
#install.packages("gridExtra",repos = 'http://mirrors.ustc.edu.cn/CRAN/')
library(gridExtra)
library(showtext)
library(sysfonts)
library(RColorBrewer)
showtext.auto(enable=T)
font.add('WenQuanYi','/usr/share/fonts/truetype/wqy/wqy-microhei.ttc')
setwd("/home/ray/Documents/suibe/2016/数据分析比赛/python_data/plot")

#### graph1 ####
df<-read.csv('graph1_2.csv',header = TRUE)
a <- reorder(df$cooking_type_upper,rep(1,length(df$cooking_type_upper)),sum)
a <- factor(a, levels=rev(levels(a)))
#df$cooking_style = factor(df$cooking_style, levels=c('面包甜点','快餐简餐','小吃','粤菜','火锅','川菜','西餐','湘菜','咖啡厅','日本料理'))
df$star <- as.factor(df$star)
p <- ggplot(data = df)
p <- p + labs(x="菜品种类",y="菜品店铺数目")  #,title = "\n"
p <- p + geom_bar(aes(x=a,fill = star)) #position = 'fill'
#p <- p+theme(legend.title= element_blank()) remove the legend title 
#p<- p+scale_fill_discrete(name = "star")
p <- p+scale_fill_brewer(palette="Blues")+theme_classic()#+theme(text = element_text(family = 'WenQuanYi'))
p+theme(text = element_text(family = 'WenQuanYi'))
ggsave( file = "latex_graph3.eps", width = 8, height = 6, dpi = 600)



#### graph2 ####
df<-read.csv('graph1.csv',header = TRUE)
df$star <- as.factor(df$star)
df$cooking_style = factor(df$cooking_style, levels=c('日本料理','西餐','火锅','川菜','咖啡厅','粤菜','面包甜点','湘菜','小吃','快餐简餐'))
p <- ggplot(data = df)
p <- p + labs(x="菜品种类",y="比例")  #,title = "\n"
p <- p + geom_bar(aes(x=df$cooking_style,fill = star),position = 'fill') #
p <- p+theme(legend.title= element_blank()) #remove the legend title 
#p<- p+scale_fill_discrete(name = "star")
p <- p+scale_fill_brewer(palette="Blues")+theme_classic()
p
ggsave( file = "latex_graph4.eps", width = 8, height = 6, dpi = 600)


#### graph3 heatmap ####
df<-read.csv('graph3.csv',header = TRUE)
test2<-df[order(df[,1]),]
test2$no<-1:length(test2[,4])
df <- df[-1:-2,]
row.names(df) <- df$None
df <- df[,-1]
#df<-df[,-7]
k <- kmeans (df, 2)
dfc <- cbind (df, Cluster= k$cluster)
dfc <- dfc[order(dfc$Cluster),]
dfc.m <- data.matrix(dfc)
pdf(file = '/home/ray/mydoc/python_data/plot/a.pdf', width = 8, height = 10)
#heatmap(dfc.m, Rowv=NA, Colv=NA, col =rev(brewer.pal(9, "GnBu")[3:8]), revC=TRUE, scale='column')
heatmap(dfc.m, col = rev(brewer.pal(9,"GnBu")[8:1]), revC=TRUE, scale='column')
#RColorBrewer::display.brewer.all()
dev.off()

#### graph4 density ####
df<-read.csv('graph4_2.csv',header = TRUE,sep = ',')
df$菜品 = factor(df$菜品, levels=c('小吃','快餐简餐','面包甜点','咖啡厅','川菜','湘菜','粤菜','火锅','西餐','日本料理','其他'))
p <- ggplot(df)
p <- p+ facet_wrap(~菜品)
#p<- p +geom_density(aes(x = 人均消费,colour = 菜品))
p<- p+ geom_histogram(alpha=0.5,aes(x = 人均消费,color = 菜品),fill = "white")
#p<- p+ geom_density(aes(x =人均消费, colour = 菜品))
p <- p+theme_classic()+theme(legend.position="none")
p
ggsave( file = "latex_graph2.eps", width = 8, height = 8, dpi = 1000)

#### 
df<-read.csv('graph4_2.csv',header = TRUE,sep = ',')
df$菜品 = factor(df$菜品, levels=c('小吃','快餐简餐','面包甜点','咖啡厅','川菜','湘菜','粤菜','火锅','西餐','日本料理','其他'))
p <- ggplot(df)
p <- p+ facet_wrap(~菜品)
#p<- p +geom_density(aes(x = 人均消费,colour = 菜品))
p<- p+ geom_histogram(alpha=0.5,aes(x = 人均消费,color = 菜品),fill = "white")
#p<- p+ geom_density(aes(x =人均消费, colour = 菜品))
p <- p+theme_classic()+theme(legend.position="none")
p
ggsave( file = "latex_graph2.eps", width = 8, height = 8, dpi = 1000)


#### graph4 shangquan ####
df <- read.csv('graph5.csv',header = TRUE)
df$bussines_area = reorder(df$bussines_area,rep(1,length(df$bussines_area)),sum)
#a <- reorder(df$bussines_area,rep(1,length(df$bussines_area)),sum)
df$bussines_area = factor(df$bussines_area,levels =(levels(df$bussines_area))[24:56])
df <- na.omit(df)
#a<- factor(a,levels = rev(levels(a)))[1:24]
#df$star<- as.factor(df$star)
df$consume_level <- as.factor(df$consume_level)
p <- ggplot(df,aes(x=df$bussines_area,fill = consume_level ))
#p <- p+ facet_wrap(~admin_area)
p <- p + labs(x="商圈名称")
p <- p+theme_bw()
p <- p+geom_bar()+coord_polar()
#p <- p+ geom_text(aes(y = reorder(df$bussines_area,rep(1,length(df$bussines_area)),sum),label = reorder(df$bussines_area,rep(1,length(df$bussines_area)),sum)))
p <- p+scale_fill_brewer(palette="Set3")+theme(panel.border = element_blank())+theme(panel.grid =element_blank())
#p <- p+ theme(axis.text.x = element_blank())
p
ggsave( file = "latex_graph9_3.eps", width = 8, height = 8, dpi = 800)


#### competition ####
df <- read.csv('compet_graph1.csv',header = TRUE)
df$商圈 = reorder(df$商圈,df$result)
df$商圈 = factor(df$商圈,levels = rev(levels(df$商圈)))
p <- ggplot(df)
p<- p+labs(y="竞争强度指数")
p <- p + geom_bar(stat = 'identity', aes(x = 商圈,y=result,fill = 行政区))
p <- p+scale_fill_brewer(palette = "Set3")+theme_classic()#+theme(legend.position="none")
p
df2 <- read.csv('compet_graph2.csv',header = TRUE)
df2$行政区 = reorder(df2$行政区,df2$result)
df2$行政区 = factor(df2$行政区,levels=rev(levels(df2$行政区)))
p1 <- ggplot(df2)
p1<- p1+labs(x ="" ,y="")
p1 <- p1 + geom_bar(stat = 'identity', aes(x = df2$行政区,y=df2$result,fill = 行政区))
p1 <- p1+scale_fill_brewer(palette = "Set3")+theme_classic()# set the legend position: +theme(legend.justification=c(1,1), legend.position=c(1,1))
p1

df3 <- read.csv('compet_graph3.csv',header = TRUE)
df3$消费水平 = reorder(df3$消费水平,df3$result)
df3$消费水平 = factor(df3$消费水平,levels=rev(levels(df3$消费水平)))
p2 <- ggplot(df3)
p2<- p2+labs(x ="消费水平" ,y="竞争强度指数")
p2 <- p2 + geom_bar(stat = 'identity', aes(x = df3$消费水平,y=df3$result,fill = "Blues"))
p2 <- p2+scale_fill_brewer(palette = "Set3")+theme_classic()+theme(legend.position="none")
p2

df4 <- read.csv('compet_graph4.csv',header = TRUE)
#df4$X4 = reorder(df4$X4,df4$result)
#df4$X4 = factor(df4$X4,levels=rev(levels(df4$X4)))
p3 <- ggplot(df4)
p3<- p3+labs(x ="星级" ,y="")
p3 <- p3 + geom_bar(stat = 'identity', aes(x =df4$X4,y=df4$result,fill = "blue"))
p3 <- p3+scale_fill_brewer(palette = "Set3")+theme_classic()+theme(legend.position="none")
p3

df5 <- read.csv('compet_graph5.csv',header = TRUE)
df5$X5= reorder(df5$X5,df5$result)
df5$X5 = factor(df5$X5,levels=rev(levels(df5$X5)))
p4 <- ggplot(df5)
p4<- p4+labs(x ="菜品" ,y="竞争强度指数")
p4 <- p4 + geom_bar(stat = 'identity', aes(x =df5$X5,y=df5$result),fill = "salmon")
p4 <- p4+scale_fill_brewer(palette = "Set3")+theme_classic()+theme(legend.position="none")
p4
df6 <- read.csv('compet_graph6.csv',header = TRUE)
p5 <- ggplot(df6)
p5<- p5+labs(x ="竞争强度指数" ,y="概率密度")
p5<-p5+geom_density(aes(x = df6$result),fill = "salmon",color = "salmon")
p5 <- p5+scale_fill_brewer(palette = "Set3")+theme_classic()
p5
pdf(file = 'latex_graph6_1.pdf', width = 15, height = 12)
p_all <- grid.arrange(p,p1,p2,p3,p4,p5, ncol=2, nrow=3, widths=c(1,1), heights=c(1,1,1))
dev.off()
#ggsave( file = "graph_competition.eps",plot = p_all, width = 8, height = 6, dpi = 600)
#??ggsave

#### for_liulu ####
df <- read.csv('to_liulu1.csv',header = TRUE)
p <- ggplot(df)
p <- p+ labs(x ="店铺消费水平" ,y="概率密度")
p <- p+ geom_density(aes(x = df$consume_average),fill = "salmon",color = "salmon")
p <- p+ scale_fill_brewer(palette = "Set3")+theme_classic()
ggsave( file = "latex_graph1.eps", width = 8, height = 6, dpi = 600)

df <- read.csv('to_liulu2_1.csv',header = TRUE)
df$cooking_type_upper= reorder(df$cooking_type_upper,df$X0)
df$cooking_type_upper = factor(df$cooking_type_upper,levels = rev(levels(df$cooking_type_upper)))
p <- ggplot(df)
p <- p+ labs(x ="菜品" ,y="评分")
p <- p+ geom_bar(stat = "identity",aes(x = df$cooking_type_upper,y = df$X0,fill = df$X),position='dodge')
p <- p+ coord_cartesian(ylim=c(6.5,8))
p <- p+ scale_fill_brewer(palette = "Set3")+theme_classic()+theme(legend.title=element_blank())
p

df <- read.csv('to_liulu3_1.csv',header = TRUE)
df$cooking_type_upper= reorder(df$cooking_type_upper,df$X0)
df$cooking_type_upper = factor(df$cooking_type_upper,levels = rev(levels(df$cooking_type_upper)))
p1 <- ggplot(df)
p1 <- p1+ labs(x ="菜品" ,y="评分")
p1 <- p1+ geom_bar(stat = "identity",aes(x = df$cooking_type_upper,y = df$X0,fill = df$X),position='dodge')
p1 <- p1+ coord_cartesian(ylim=c(7,8))
p1 <- p1+ scale_fill_brewer(palette = "Set3")+theme_classic()+theme(legend.title=element_blank())
p1
ggsave( file = "latex_graph5.eps", width = 8, height = 6, dpi = 600)

grid.arrange(p,p1, ncol=2, nrow=1, widths=c(1,1), heights=c(1))

df<- read.csv('to_liulu4_1.csv',header = TRUE)
df$admin_area = factor(df$admin_area, levels=c('天河区','越秀区','海珠区','荔湾区','白云区','番禺区','黄埔区','花都区','增城区','从化区','南沙区','萝岗区'))
p <- ggplot(df)
p <- p+ facet_wrap(~admin_area)
p <- p+labs(x = "人均消费",y = "概率密度")
#p<- p +geom_density(aes(x = 人均消费,colour = 菜品))
#p<- p+ geom_histogram(alpha=0.5,aes(x = 人均消费,fill = 菜品))
p<- p+ geom_histogram(aes(x =consume_average,color = admin_area),fill = "white")
p <- p+theme_classic()
p <- p+theme(legend.title= element_blank())+theme(legend.position="none")
p
ggsave( file = "latex_graph2.eps", width = 8, height = 8, dpi = 1000)


#### liujingfang ####
df<-read.csv("liujingfang1.csv",header = TRUE)

df$cooking_type_upper <- reorder(df$cooking_type_upper,rep(1,length(df$cooking_type_upper)),sum)
p <- ggplot(df)
p <- p + labs(x ="菜品" ,y="店铺数量")
p <- p+ geom_bar(aes(x = df$大类, fill=df$if_liansuo),position = "fill")
p <- p+ scale_fill_brewer(palette = "Set3")+theme_classic()+theme(legend.title=element_blank())
p
ggsave( file = "latex_graph7.eps", width = 8, height = 6, dpi = 600)

df <- read.csv("liujingfang_2_1.csv",header = TRUE)
p <- ggplot(df) + geom_point(aes(x = df$店铺数,y = df$人均消费),color = "salmon")
p <- p+ scale_fill_brewer(palette = "Set3")+theme_classic()+theme(legend.title=element_blank())
p

#### logistic ####
df <- read.csv("logistic_data.csv",header = TRUE)
p <- ggplot(df) + geom_point(aes(x = df$taste,y = df$if_open))
glm.fit <- glm(df$if_open~df$consume_level+df$star_x+df$competition_score+df$consume_average+df$taste+df$serve+df$environment,family = 'binomial')
summary(glm.fit)
step(glm.fit)
glm.probs =predict(glm.fit,type ="response")


#### tangxiang ####
df <- read.csv('to_tangxiang.csv',header = TRUE)
df$cooking_type_upper <- reorder(df$cooking_type_upper,rep(1,length(df$cooking_type_upper)),sum)
df$cooking_type_upper <- factor(df$cooking_type_upper, levels=rev(levels(df$cooking_type_upper))[1:10])
df$star <- as.factor(df$star)
p <- ggplot(df)
p <- p+ facet_wrap(~admin_area)
p<- p+ geom_bar(aes(x = cooking_type_upper,fill = star))
p <- p+ scale_fill_brewer(palette = "Set3")+theme_classic()
p
ggsave( file = "latex_graph8.eps", width = 18, height = 14, dpi = 1500)

df <-read.csv('tangxiang_1_1.csv',header = TRUE)
df= df[-2,]
df = df[c(-3,-6,-7),]
df$admin_area = reorder(df$admin_area,df$consume_average)
df$admin_area = factor(df$admin_area,levels = rev(levels(df$admin_area)))
p<-ggplot(df)
p <- p+labs(x = "行政区",y = "人均消费均值")
p <- p+ geom_bar(aes(x = df$admin_area,y = df$consume_average),stat = "identity",fill = "steelblue")
p <- p+ scale_fill_brewer(palette = "Set3")+theme_classic()
p
ggsave( file = "latex_graph8.eps", width = 18, height = 14, dpi = 1500)


#### if_open ####
df<-read.csv('table.csv',header = TRUE)
df$行政区 = reorder(df$行政区,df$店铺数)
df$行政区 = factor(df$行政区,levels = rev(levels(df$行政区)))
p<-ggplot(df)
p <- p+ geom_bar(aes(x = 行政区,y = 店铺数),stat = "identity",fill = "salmon")
p <- p+ scale_fill_brewer(palette = "Set2")+theme_classic()
p
ggsave( file = "latex_graph12.eps", width = 8, height = 6, dpi = 600)

df<-read.csv('table2.csv',header = TRUE)
df$菜品 = reorder(df$菜品,df$店铺数)
df$菜品 = factor(df$菜品,levels= rev(levels(df$菜品)))
#df$行政区 = reorder(df$行政区,df$店铺数)
#df$行政区 = factor(df$行政区,levels = rev(levels(df$行政区)))
p<-ggplot(df)
p <- p+ geom_bar(aes(x = 菜品,y = 店铺数,fill = 是否营业),stat = "identity",position = "dodge")
p <- p+ scale_fill_brewer(palette = "Set2")+theme_classic()
p
ggsave( file = "latex_graph13.eps", width = 8, height = 6, dpi = 600)

df<-read.csv('newrequest.csv',header = TRUE)
df$real_admin = reorder(df$real_admin,rep(1,length(df$real_admin)),sum)
df$real_admin= factor(df$real_admin,levels = rev(levels(df$real_admin)))
colnames(df)=c("id","是否营业","real_admin")
df$`是否营业` = as.factor(df$`是否营业`)
p<-ggplot(df)
p <- p + labs(x="行政区")
p <- p+ geom_bar(aes(x = df$real_admin,fill = `是否营业`),position = "dodge",stat="count")
p <- p+ scale_fill_brewer(palette = "Set2")+theme_classic()#+theme(legend.title=element_blank())
p
ggsave( file = "latex_graph12_2.eps", width = 7, height = 5, dpi = 600)

#### fujiati ####
df1<-read.csv('fujia.csv',header = TRUE)
df <- na.omit(df1)
p <- ggplot(data = df)
#p <- p + labs(x="菜品种类",y="菜品店铺数目")  #,title = "\n"
p <- p + geom_bar(aes(x=as.factor(df$start_month),fill=df$X3),na.rm =FALSE) #position = 'fill'
#p <- p+theme(legend.title= element_blank()) remove the legend title 
#p<- p+scale_fill_discrete(name = "star")
p <- p+scale_fill_brewer(palette="Set3")+theme_classic()#+theme(text = element_text(family = 'WenQuanYi'))
p+theme(text = element_text(family = 'WenQuanYi'))
p

p <- ggplot(data = df)
p <- p + labs(x="月份")  #,title = "\n"
p <- p + geom_bar(aes(x=as.factor(df$start_month),fill=df$X3),na.rm =FALSE) #position = 'fill'
#p <- p+theme(legend.title= element_blank()) remove the legend title 
#p<- p+scale_fill_discrete(name = "star")
p <- p+scale_fill_brewer(palette="Set3")+theme_classic()+theme(legend.title=element_blank())
p+theme(text = element_text(family = 'WenQuanYi'))
ggsave( file = "fujia_1.eps", width = 8, height = 6, dpi = 600)


p <- ggplot(data = df)
p <- p + labs(x="温度")  #,title = "\n"
p <- p + geom_histogram(aes (x = df$X4,fill=df$X3),na.rm =FALSE) #position = 'fill'
#p <- p+theme(legend.title= element_blank()) remove the legend title 
#p<- p+scale_fill_discrete(name = "star")
p <- p+scale_fill_brewer(palette="Set3")+theme_classic()+theme(legend.title=element_blank())
p+theme(text = element_text(family = 'WenQuanYi'))
ggsave( file = "fujia_2.eps", width = 8, height = 6, dpi = 600)

a <- lm(df$X2~df$start_month)
plot(df$X2,df$start_month)
summary(a)


df1<-read.csv('show.txt',header = FALSE)
a <- reorder(df1$V1,df1$V2)
df1$V1 <- factor(a, levels=rev(levels(a)))
p <- ggplot(data = df1)
p <- p + labs(x="频率",y="地名")  #,title = "\n"
p <- p + geom_bar(aes(x=df1$V1,y = df1$V2),stat = "identity",color = "salmon",fill = "salmon") #position = 'fill'
#p <- p+theme(legend.title= element_blank()) remove the legend title 
#p<- p+scale_fill_discrete(name = "star")
p <- p+scale_fill_brewer(palette="Set3")+theme_classic()#+theme(text = element_text(family = 'WenQuanYi'))
p+theme(text = element_text(family = 'WenQuanYi'))
ggsave( file = "fujia_4.eps", width = 10, height = 8, dpi = 900)

df1<-read.csv('show1.txt',header = FALSE)
a <- reorder(df1$V1,df1$V2)
df1$V1 <- factor(a, levels=rev(levels(a)))
p <- ggplot(data = df1)
p <- p + labs(x="频率",y="地名")  #,title = "\n"
p <- p + geom_bar(aes(x=df1$V1,y = df1$V2),stat = "identity",color = "salmon",fill = "salmon") #position = 'fill'
#p <- p+theme(legend.title= element_blank()) remove the legend title 
#p<- p+scale_fill_discrete(name = "star")
p <- p+scale_fill_brewer(palette="Set3")+theme_classic()#+theme(text = element_text(family = 'WenQuanYi'))
p+theme(text = element_text(family = 'WenQuanYi'))
ggsave( file = "fujia_5.eps", width = 12, height = 8, dpi = 1000)

df1<-read.csv('name_split.csv',header = FALSE)
df2 <- df1[order(df1[,2],decreasing=T),]
df2[2:20,]


setwd("/home/ray/Desktop/")
df<-read.csv('1.csv',header = TRUE)
df$marriage_dummy <- as.factor(df$marriage_dummy)
df$hap = factor(df$hap, levels=c('非常不幸福','不是很幸福','一般幸福','很幸福','非常幸福'))
p <- ggplot(data = df)
p <- p + labs(x="",y="比例")  #,title = "\n"
p <- p + geom_bar(aes(x=df$hap,fill = as.factor(df$marriage_dummy)),position = 'fill') #
p<- p+scale_fill_discrete(name = "star")
p <- p+scale_fill_brewer(palette = "Blues",direction = 1)+theme_classic()
p <- p+theme(legend.title= element_blank()) #remove the legend title 
p
