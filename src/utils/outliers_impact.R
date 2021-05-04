
library(data.table)
library(ggplot2)

data1 <- fread("data/data1.txt")
data2 <- fread("data/data2.txt")

data1[, outlier := FALSE]
data2[, outlier := FALSE]
data2[Row == 21, outlier := TRUE]


(plot1 <- ggplot(data1, aes(x = x, y = y, color = as.factor(outlier))) +
          geom_point() +
          theme_classic() +
          theme(legend.position = "none", 
                panel.grid.major = element_line()))

(plot2 <- ggplot(data1, aes(x = x, y = y, color = as.factor(outlier))) +
            geom_point() +
            geom_smooth(method = 'lm', formula = y~x, color = "gray", 
                        se = FALSE) +
            theme_classic() +
            theme(legend.position = "none", 
                  panel.grid.major = element_line()))

(plot3 <- ggplot(data2, aes(x = x, y = y, color = as.factor(outlier))) +
            geom_point() + 
            theme_classic() +
            theme(legend.position = "none", 
                  panel.grid.major = element_line()))

(plot4 <- ggplot(data2, aes(x = x, y = y, color = as.factor(outlier))) +
            geom_point() + 
            geom_smooth(method = 'lm', formula = y~x, color = "gray", 
                        se = FALSE) +
            theme_classic() +
            theme(legend.position = "none", 
                  panel.grid.major = element_line()))

ggsave("presentation/images/plot1.pdf", plot1)
ggsave("presentation/images/plot2.pdf", plot2)
ggsave("presentation/images/plot3.pdf", plot3)
ggsave("presentation/images/plot4.pdf", plot4)


lm1 <- lm(y~x, data = data1)
sink("presentation/images/lm1.txt")
print(summary(lm1))
sink()

lm2 <- lm(y~x, data = data2)
sink("presentation/images/lm2.txt")
print(summary(lm2))
sink()
