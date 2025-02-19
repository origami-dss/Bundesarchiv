
# Ich wende hier zwei Clusterungstechniken auf den mnist-Datensatz an.
# Ich gehe davon aus, dass die Klassifizierung von Buchstaben sehr ähnlich funktioniert
# Ich versuche im Wesentlichen mit den R-Standardbibliotheken auszukommen
#



# Lade den MNIST Datensatz
if (!require("dslabs")) install.packages("dslabs"); library("dslabs")

mnist <- read_mnist()
dim(mnist$train$images)

# Häufigkeiten einzelner Ziffern
table(mnist$train$labels)


set.seed(123, sample.kind = "Rounding")

# Betrachte einen verkleinerten Datensatz

index <- sample(nrow(mnist$train$images), 10000)
X_train <- mnist$train$images[index,]
y_train <- factor(mnist$train$labels[index])

index <- sample(nrow(mnist$test$images), 1000)
X_test <- mnist$test$images[index,]
y_test <- factor(mnist$test$labels[index])



# Betrachte die durchschnittliche Ziffer
sds <- colSums(X_train)

par(mfrow=c(1,1))
par(mar = c(5, 4, 4, 2) + 0.1)


image(0:28, 0:28, array(sds, dim=c(28,28)), zlim= c(0,max(sds)),
      xlab="x (pixel)", ylab = "y (pixel)", main = "Average Digit of the Trainig Data Set",
      col = gray.colors(16, start = 0, end = 1, gamma = 1))


# Wende K-Means-Clusterung als Beispiel für eine unsupervised Technik an.

colnames(X_train) <- 1:ncol(mnist$train$images)
colnames(X_test) <- colnames(X_train)

k_means_clus <- kmeans(X_train, centers = 10, iter.max = 50, nstart = 50)
cluster_centers <- k_means_clus$centers

# Stelle die die Zentren der 10 Cluster dar

for (i in 1:10)
{
  im <- array(cluster_centers[i,], dim=c(28,28))
  im <- t(apply(im, 1, rev))
  image(0:28, 0:28, im, ann = FALSE, main = "a", xaxt='n', yaxt='n',
        col = gray.colors(16, start = 0, end = 1, gamma = 1))
  text(1, 27,  paste0(i,". CL-Zentrum"), cex=0.65, pos=4, col="red")
}

# Das entspricht leider nicht vollständig den zu erwartenden  Ziffern



for (i in 1:10) image(0:28, 0:28, array(cluster_centers[i,], dim=c(28,28)),
         xlab="x (pixel)", ylab = "y (pixel)", main = paste("Average Digit of the Kmeans Cluster No. ",i),
         col = gray.colors(16, start = 1, end = 0, gamma = 1))


# Versuche deshalb eine supervised Pattern Recognition Technik, #
#  Deshalb passe ich hier ein Gausches Mischmodell an, d.h. ich betrachte jede Ziffer als gemischte Verteilung von Gaussverteilungen
# in einem 784(28*28)-dimensionalen Raum.

if (!require("mclust")) install.packages("mclust"); library("mclust")

# passe ein Modell an, das dauert etwas ...

gmm_model <- MclustDA(X_train, class = y_train,  D=1:2)

par(mfrow=c(2,5))
par(mar=c(0, 0, 2, 0)+0.1)

for (i in 1:10)
{
  im <- array(gmm_model$models[[i]]$parameters$mean[,1], dim=c(28,28))
  im <- t(apply(im, 1, rev))
  image(0:28, 0:28, im, ann = FALSE, main = "a", xaxt='n', yaxt='n',
        col = gray.colors(16, start = 0, end = 1, gamma = 1))
  text(1, 27,  paste0("MW ",i,". Komponente"), cex=0.65, pos=4, col="red")
}


## Sage die Daten des Test-Datensatzes mithilfe des gefitteten Models voraus

pp <- predict( gmm_model, newdata = X_test)


## Vergleiche die vorhergesagten mit den gelabelten Ziffern

result = data.frame(prediction = pp$classification, target = y_test)

tt = table(result)

# 2D Kontingenztafel

print(tt)

#Anzahl der 1000 Ziffern, die korrekt bestimmt sind:

sum(diag(tt))
# Besonders häufig wurde  (10 Mal) die Ziffer 8 als 3 erkannt


# Bessere Vorhersagen können durch höhere Auflösung beim Scannen der Ziffern oder durch
# eine höhere Anzahl der Komponenten des mixed models für jede Ziffer (z.B. G=1:10) erreicht werden.
# Dann ist dringend empfohlen, vorher mit einer PCA zu beginnen, um die Dimensionalität zu reduzieren.
# Desweiteren sollten größere Datensätze zu besseren Mustererkennungen führen.
# siehe
# https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf
