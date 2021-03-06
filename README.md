## ๐ K-Folds CV๋ฅผ ํ์ฉํ ์ต๊ณ ์ฑ๋ฅ ๋ชจํ ํ์
> #### ๊ฐ์ : Heart Failure Prediction Dataset์ ์ด์ง์  ๋ฐ์๋ณ์์ธ HeartDisease์ ์์ธกํ๋ ๋ค์ํ ํต๊ณํ์ต ๋ชจ๋ธ๋ค์ 10-folds CV๋ฅผ ํตํด ๊ตฌํํ๊ณ  ์ ํฉํ์ฌ, ์ด์ ๋ฐ๋ฅธ ์ ํ๋์ AUC๊ฐ์ ๋น๊ตํ๋ค.

#

### 1. ์ ์ ๊ธฐ๊ฐ & ์ฐธ์ฌ ์ธ์
> #### 2021.11 ~ 2021.11
> #### 3์ธ
> #### ์ํ ์ญํ  
>> 10-folds CV๋ฅผ ํ์ฉํ์ฌ Logistic Regression, LDA, QDA, Bagging, Random Forest์ ์ต์  ํ์ดํผ ํ๋ผ๋ฏธํฐ ํ์ ๋ฐ ๋ชจํ์์ฑ  
>> ๋ถ์ ๊ฒฐ๊ณผ ์ทจํฉ ๋ฐ ๋ณด๊ณ ์ ์์ฑ

> #### ๋ฐฐ์ด์ 
>> #### ์ด๋ก ์ ์ผ๋ก ์ดํดํ๊ณ  ์๋ ๋ค์ํ ๋ฐ์ดํฐ ๋ง์ด๋ ๊ธฐ๋ฒ๋ค์ ํ์ฉํ์ฌ, ์ค์  ๋ฐ์ดํฐ์ ํ์ฉํ  ์ ์์๋ ๊ธฐํ์์ต๋๋ค. 
>> #### ๋ํ ํด๋น ํ๋ก์ ํธ๋ฅผ ํตํด ๋จธ์ ๋ฌ๋์ ์ฝ๋ฉ๊ณผ ์ค์ต๋ฅ๋ ฅ์ ํจ์์ํฌ ์ ์์์ต๋๋ค. 

#

### 2. ์ฌ์ฉ ํ๋ก๊ทธ๋จ : R 4.0.0
#### ์ฌ์ฉ ๋ผ์ด๋ธ๋ฌ๋ฆฌ : 
> library(ROCR)  
> library(MASS)  
> library(boot)  
> library(e1071)  
> library(tree)  
> library(gbm)  
> library(randomForest)  

<br/>

### 3. [R์ฝ๋](https://github.com/ChSSolee/002/blob/main/K-Folds%20CV%EB%A5%BC%20%ED%99%9C%EC%9A%A9%ED%95%9C%20%EC%B5%9C%EA%B3%A0%EC%84%B1%EB%8A%A5%20%EB%AA%A8%ED%98%95%20%ED%83%90%EC%83%89.Rmd)

<br/>

## 1. ์๋ก 

<br/>

### 1-1 ๊ฐ์

- Heart Failure Prediction Dataset์ ์ด์ง์  ๋ฐ์๋ณ์์ธ HeartDisease์ ์์ธกํ๋ ๋ค์ํ ํต๊ณํ์ต ๋ชจ๋ธ๋ค์ **10-folds CV**๋ฅผ ํตํด ๊ตฌํํ๊ณ  ์ ํฉํ์ฌ, ์ด์ ๋ฐ๋ฅธ ์ ํ๋์ AUC๊ฐ์ ๋น๊ตํ๋ค.

<br/>

- ๋ผ์ด๋ธ๋ฌ๋ฆฌ ํธ์ถ
```javascript
library(ROCR)
library(MASS)
library(boot)
library(e1071)
library(tree)
library(gbm)
library(randomForest)
```

<br/>

### 1-2 ๋ฐ์ดํฐ ์ดํด

1. ๋ฐ์ดํฐ ์ : โHeart Failure Prediction Datasetโ
2. ๋ฐ์ดํฐ์ ๊ฐ์ : 918๋ช์ ์ฌ์ฅ์งํ์ ์ฌ๋ถ์ ์ฌ์ฅ์งํ๊ณผ ๊ด๋ จ๋ 11๊ฐ์ ๋ณ์๋ค๋ก ๊ตฌ์ฑ. (์ด 12๊ฐ)
3. ๋ณ์๋ณ ์์ฑ

- Age : ํ์ ๋์ด / ์์นํ
- Sex : ํ์ ์ฑ๋ณ / ๋ฒ์ฃผํ (M = ๋จ์ฑ / F = ์ฌ์ฑ) 
- ChestPainType : ํ์ ํ๋ถ ํต์ฆ ์ ํ / ๋ฒ์ฃผํ (TA = ์ ํ์  ํ์ฌ์ฆ / ATA = ๋น์ ํ์  ํ์ฌ์ฆ / NAP = ๋นํ์ฌ์ฆ์ฑ ํต์ฆ / ASY = ๋ฌด์ฆ์)
- RestingBP : ์์  ํ์ / ์์นํ 
- Cholesterol : ํ์ฒญ ์ฝ๋ ์คํ๋กค / ์์นํ
- FastingBS : ๊ณต๋ณต ํ๋น / ๋ฒ์ฃผํ (1 = ๊ณต๋ณต ํ๋น > 120 mg/dl / 0 = ๊ทธ ์ธ)
- RestingECG : ์์ ์ ์ฌ์ ๊ณ ๊ฒฐ๊ณผ / ๋ฒ์ฃผํ (Normal = ์ ์ / ST = ๋น์ ์ ST-T ํ๋ / LVH : ์ข์ฌ์ค ๋น๋)
- MaxHR : ์ต๋ ์ฌ๋ฐ์ / ์์นํ 
- ExerciseAnigma : ์ด๋ ์ ๋ฐ ํ์ฌ์ฆ / ๋ฒ์ฃผํ (Y = ์ฆ์ ์์ / N = ์ฆ์ ์์)
- Oldpeak : ST ๋ถ์  ํ๊ฐ ์ ๋ / ์์นํ
- ST_Slope : ST ๋ถ์  ๊ฒฝ์ฌ / ๋ฒ์ฃผํ (Up = ์ค๋ฅด๋ง / Flat = ํ๋ฉด / Down = ๋ด๋ฆฌ๋ง)
- HeartDisease : ์ฌ์ฅ์งํ / ๋ฒ์ฃผํ (1 = ์ฌ์ฅ์งํ / 0 = ์ ์)

<br/>

4. ๋ณ์๋ณ ์๊ด๊ด๊ณ ์๊ฐํ
<img width="840" height="600" src="https://user-images.githubusercontent.com/100699925/158420885-0e4cdd82-323b-44d1-9cfd-ea3e61569d3c.png">

```javascript
heart <- read.csv("heart.csv", header = T)
plot(heart, panel = panel.smooth)
```

- ์ฐ์ ๋๋ฅผ ํตํด ์ค๋ช๋ณ์ ๊ฐ์ ๋๋ต์ ์ธ ์๊ด๊ด๊ณ๋ฅผ ํ์ํ  ์ ์๋ค. ์๋ฅผ ๋ณด์ ์ค๋ช๋ณ์ RestingBP, MaxHR ๊ฐ์ ์๊ด๊ด๊ณ๊ฐ ์์ ๊ฒ์ผ๋ก ํ๋จ๋๋ฉฐ, ์ดํ ๋ถ์์์๋ ๋ณด๋ค ์ ํํ ํ๋จ์ ์ํด ์ ์ง์ ํ๋ฒ๊ณผ ํ์ง์ ํ๋ฒ์ ์ด์ฉํ๊ธฐ๋ก ํ๋ค.

<br/>

### ๋ฐ์ดํฐ ์ ์ฒ๋ฆฌ (๊ฒฐ์ธก๊ฐ ํ์ธ ๋ฐ ๋ณ์ํํ ์ค์ )
- ๋ถ์์ ์ํด ๋ณ์๋ค์ ํํ๋ฅผ ์์ ํ๋ค.
```javascript
sum(is.na(heart)) 
```
```
## [1] 0
```
```javascript
heart$Sex <- factor(heart$Sex) ; heart$ChestPainType <- factor(heart$ChestPainType)
heart$FastingBS <- factor(heart$FastingBS) ; heart$RestingECG <- factor(heart$RestingECG)
heart$ExerciseAngina <- factor(heart$ExerciseAngina) ; heart$ST_Slope <- factor(heart$ST_Slope)
heart$HeartDisease <- factor(heart$HeartDisease) ; heart$Age <- as.numeric(heart$Age)
heart$RestingBP <- as.numeric(heart$RestingBP) ; heart$Cholesterol <- as.numeric(heart$Cholesterol)
heart$MaxHR <- as.numeric(heart$MaxHR) ;heart$Oldpeak <- as.numeric(heart$Oldpeak)
```

<br/>

- ํ๋ จ ๋ฐ์ดํฐ์ ํ์คํธ ๋ฐ์ดํฐ๋ฅผ 7:3์ ๋น์จ๋ก ๋ถํ .
```javascript
set.seed(1)
train <- sort(sample(1:nrow(heart), round(nrow(heart)*0.7)))
test <- sort(setdiff(1:nrow(heart),train))
train.d <- heart[train,] ; test.d <- heart[test,]
dim(test.d) ; dim(train.d)
```
```
## [1] 275  12
```
```
## [1] 643  12
```

<br/>

## 2. ๋ณธ๋ก 

<br/>

### 2-1 Logistic & LDA & QDA & Naive Bayes
- ์์ ๋ชจํ (๋ชจ๋  ๋ณ์๋ฅผ ์ค๋ช๋ณ์๋ก ๊ฐ์ง๋ ๋ชจํ)๊ณผ ๋น๊ต๊ฐ ๋  ์ ์๋ ๋ชจํ ์ค์ ์ ์ํด, step()ํจ์๋ฅผ ํตํด ์ต์ ์ AIC๊ฐ์ ๊ฐ์ง๋ ๋ชจํ์ ํ์
- ์์ ๋ชจํ, ์๋ชจํ (์์ํญ๋ง์ ์ค๋ช ๋ณ์๋ก ๊ฐ์ง๋ ๋ชจํ)์ ๊ฐ๊ฐ ์ ์ง์ ํ๋ฒ๊ณผ ํ์ง์ ๊ฑฐ๋ฒ์ ์ ์ฉํ์ฌ ๋์จ ์ถ์๋ชจํ์ ๋์ผํ ๊ฒ์ผ๋ก ํ์ธ ๋์์ผ๋ฉฐ, ์๋์ ๊ฐ๋ค. 
- HeartDisease ~ Age + Sex + ChestPainType + Cholesterol + FastingBS + ExerciseAngina + Oldpeak + ST_Slope

```javascript
Fullmod = glm(HeartDisease ~ ., data = train.d, family = binomial(link = "logit")) # ์์ ๋ชจํ
Nullmod = glm(HeartDisease ~ 1, data = train.d, family = binomial(link = "logit")) # ์๋ชจํ
backward <- step(Fullmod, direction = "backward")
forward <- step(Nullmod, scope = list(lower = formula(Nullmod), upper = formula(Fullmod)), direction = "forward")
```
```javascript
formula(backward) ; formula(forward)
```
```
## HeartDisease ~ Age + Sex + ChestPainType + Cholesterol + FastingBS + ExerciseAngina + Oldpeak + ST_Slope
```
```
## HeartDisease ~ Age + Sex + ChestPainType + Cholesterol + FastingBS + ExerciseAngina + Oldpeak + ST_Slope
```

<br/>

```javascript
mod = glm(formula(backward), data = train.d, family = binomial(link = "logit")) # ์ถ์๋ชจํ
formula(mod)
```
```
## HeartDisease ~ Age + Sex + ChestPainType + Cholesterol + FastingBS + ExerciseAngina + Oldpeak + ST_Slope
```


<br/>

#### ๋ก์ง์คํฑ ํ๊ท
- 10-folds CV๋ฅผ ํ์ฉํ์ฌ ๋ก์ง์คํฑ ๋ฐฉ๋ฒ์ ๋ชจํ๋ณ ์ ํ๋์ AUC๊ฐ ๊ณ์ฐ

#### **1. ์์ ๋ชจํ ์ ํฉ**
**์ ํ๋**
```javascript
k = 10 ; list <- 1:k
set.seed(2)
id <- sample(1:k, nrow(heart), replace = T)
glm.prediction <- data.frame()
glm.testset <- data.frame()

for (i in 1:k) {
  train.set = subset(heart, id %in% list[-i])
  test.set = subset(heart, id %in% c(i))
  glm.fit.k <- glm(formula(Fullmod), data = train.set, family = binomial(link="logit"))
  glm.pred.k <- as.data.frame(predict(glm.fit.k, newdata = test.set, type = "response"))
  
  glm.prediction <- rbind(glm.prediction, glm.pred.k)
  glm.testset <- rbind(glm.testset, as.data.frame(test.set[,12]))
}

(LR.Full.acc <- mean(round(glm.prediction)[,1] == glm.testset[,1]))
```
```
## [1] 0.8627451
```

<br/>

**AUC๊ฐ**
```javascript
glm.pr1 <- prediction(glm.prediction[,1], glm.testset[,1])
glm.perf1 <- performance(glm.pr1, measure = "tpr", x.measure = "fpr")
glm.auc1 <- performance(glm.pr1, measure = "auc") ; (LR.Full.auc <- unlist(glm.auc1@y.values))
```
```
## [1] 0.9244383
```


<br/>

#### **2. ์ถ์๋ชจํ ์ ํฉ**
**์ ํ๋**
```javascript
k = 10 ; list <- 1:k
set.seed(2)
id <- sample(1:k, nrow(heart), replace = T)
glm.prediction <- data.frame()
glm.testset <- data.frame()

for (i in 1:k) {
  train.set = subset(heart, id %in% list[-i])
  test.set = subset(heart, id %in% c(i))
  glm.fit.k <- glm(formula(mod), data = train.set, family = binomial(link="logit"))
  glm.pred.k <- as.data.frame(predict(glm.fit.k, newdata = test.set, type = "response"))
  
  glm.prediction <- rbind(glm.prediction, glm.pred.k)
  glm.testset <- rbind(glm.testset, as.data.frame(test.set[,12]))
}

(LR.mod.acc <- mean(round(glm.prediction)[,1] == glm.testset[,1]))
```
```
## [1] 0.8660131
```

<br/>

**AUC๊ฐ**
```javascript
glm.pr2 <- prediction(glm.prediction[,1], glm.testset[,1])
glm.perf2 <- performance(glm.pr2, measure = "tpr", x.measure = "fpr")
glm.auc2 <- performance(glm.pr2, measure = "auc") ; (LR.mod.auc <- unlist(glm.auc2@y.values))
```
```
## [1] 0.9256386
```

<br/>

**๋ก์ง์คํฑ ํ๊ท์์ ์ถ์๋ชจํ์ ์ ํฉํ์์ ๋์ AUC๊ฐ์ด ๋์์ ํ์ธํ  ์ ์์ผ๋ฉฐ, ROC ๊ณก์ ์ ์๋์ ๊ฐ๋ค.**

| | **์์ ๋ชจํ**	| **์ถ์๋ชจํ** |
| ---- | ---- | ---- |  
| ์ ํ๋	| 0.8627451	| 0.8660131 |
| AUC๊ฐ	| 0.9244383	| 0.9256386 |

```javascript
plot(glm.perf2)
```
<img width="840" height="600" src="https://user-images.githubusercontent.com/100699925/158420891-911c3d56-47f2-4f86-8b5b-826afda3ce28.png">



<br/>

#### LDA
- 10-folds CV๋ฅผ ํ์ฉํ์ฌ LDA ๋ฐฉ๋ฒ์ ๋ชจํ๋ณ ์ ํ๋์ AUC๊ฐ ๊ณ์ฐ

<br/>

#### **1. ์์ ๋ชจํ ์ ํฉ**
**์ ํ๋**
```javascript
lda.prediction <- data.frame()
lda.testset <- data.frame()

set.seed(2)
for (i in 1:k) {
  train.set = subset(heart, id %in% list[-i])
  test.set = subset(heart, id %in% c(i))
  lda.fit.k <- lda(formula(Fullmod), data = train.set)
  lda.pred.k <- as.data.frame(predict(lda.fit.k, newdata = test.set))
  
  lda.prediction <- rbind(lda.prediction, lda.pred.k)
  lda.testset <- rbind(lda.testset, as.data.frame(test.set[,12]))
}
(LDA.Full.acc <- mean(lda.prediction$class == lda.testset[,1]))
```
```
## [1] 0.8649237
```

<br/>

**AUC๊ฐ**
```javascript
lda.pr1 <- prediction(lda.prediction$posterior.1, lda.testset[,1])
lda.perf1 <- performance(lda.pr1, measure = "tpr", x.measure = "fpr")
lda.auc1 <- performance(lda.pr1, measure = "auc") ; (LDA.Full.auc <- unlist(lda.auc1@y.values))
```
```
0.9250624
```

<br/>

#### **2. ์ถ์๋ชจํ ์ ํฉ**
**์ ํ๋**
```javascript
lda.prediction <- data.frame()
lda.testset <- data.frame()

set.seed(2)
for (i in 1:k) {
  train.set = subset(heart, id %in% list[-i])
  test.set = subset(heart, id %in% c(i))
  lda.fit.k <- lda(formula(mod), data = train.set)
  lda.pred.k <- as.data.frame(predict(lda.fit.k, newdata = test.set))
  
  lda.prediction <- rbind(lda.prediction, lda.pred.k)
  lda.testset <- rbind(lda.testset, as.data.frame(test.set[,12]))
}
(LDA.mod.acc <- mean(lda.prediction$class == lda.testset[,1]))
```
```
## [1] 0.8714597
```

<br/>

**AUC๊ฐ**
```javascript
lda.pr2 <- prediction(lda.prediction$posterior.1, lda.testset[,1])
lda.perf2 <- performance(lda.pr2, measure = "tpr", x.measure = "fpr")
lda.auc2 <- performance(lda.pr2, measure = "auc") ; (LDA.mod.auc <- unlist(lda.auc2@y.values))
```
```
## [1] 0.9259314
```

<br/>

**LDA์์ ์ถ์๋ชจํ์ ์ ํฉํ์์ ๋์ AUC๊ฐ์ด ๋์์ ํ์ธํ  ์ ์์ผ๋ฉฐ, ROC ๊ณก์ ์ ์๋์ ๊ฐ๋ค.**
| | **์์ ๋ชจํ**	| **์ถ์๋ชจํ** |
| ---- | ---- | ---- |  
| ์ ํ๋	| 0.8649237	| 0.8714597 |
| AUC๊ฐ	| 0.9250624	| 0.9259314 |


```javascript
plot(lda.perf2)
```
<img width="840" height="600" src="https://user-images.githubusercontent.com/100699925/158420895-3b3ce2ce-ac98-4f26-812e-3e13c95ef40a.png">


<br/>

#### QDA
- 10-folds CV๋ฅผ ํ์ฉํ์ฌ QDA ๋ฐฉ๋ฒ์ ๋ชจํ๋ณ ์ ํ๋์ AUC๊ฐ ๊ณ์ฐ

<br/>

#### **1. ์์ ๋ชจํ ์ ํฉ**
**์ ํ๋**
```{r pressure20}
qda.prediction <- data.frame()
qda.testset <- data.frame()

set.seed(2)
for (i in 1:k) {
  train.set = subset(heart, id %in% list[-i])
  test.set = subset(heart, id %in% c(i))
  qda.fit.k <- qda(formula(Fullmod), data = train.set)
  qda.pred.k <- as.data.frame(predict(qda.fit.k, newdata = test.set))
  
  qda.prediction <- rbind(qda.prediction, qda.pred.k)
  qda.testset <- rbind(qda.testset, as.data.frame(test.set[,12]))
}

(qda.Full.acc <- mean(qda.prediction$class == qda.testset[,1]))
```
```
## [1] 0.8453159
```

<br/>

**AUC๊ฐ**
```javascript
qda.pr1 <- prediction(qda.prediction$posterior.1, qda.testset[,1])
qda.perf1 <- performance(qda.pr1, measure = "tpr", x.measure = "fpr")
qda.auc1 <- performance(qda.pr1, measure = "auc") ; (qda.Full.auc <- unlist(qda.auc1@y.values))
```
```
## [1] 0.9116094
```

<br/>

#### **1. ์ถ์๋ชจํ ์ ํฉ**
**์ ํ๋**
```javascript
qda.prediction <- data.frame()
qda.testset <- data.frame()

set.seed(2)
for (i in 1:k) {
  train.set = subset(heart, id %in% list[-i])
  test.set = subset(heart, id %in% c(i))
  qda.fit.k <- qda(formula(mod), data = train.set)
  qda.pred.k <- as.data.frame(predict(qda.fit.k, newdata = test.set))
  
  qda.prediction <- rbind(qda.prediction, qda.pred.k)
  qda.testset <- rbind(qda.testset, as.data.frame(test.set[,12]))
}

(qda.mod.acc <- mean(qda.prediction$class == qda.testset[,1]))
```
```
## [1] 0.8420479
```

<br/>

**AUC๊ฐ**
```javascript
qda.pr2 <- prediction(qda.prediction$posterior.1, qda.testset[,1])
qda.perf2 <- performance(qda.pr2, measure = "tpr", x.measure = "fpr")
qda.auc2 <- performance(qda.pr2, measure = "auc") ; (qda.mod.auc <- unlist(qda.auc2@y.values))
```
```
## [1] 0.9065873
```

<br/>

**QDA์์ ์์ ๋ชจํ์ ์ ํฉํ์์ ๋์ AUC๊ฐ์ด ๋์์ ํ์ธํ  ์ ์์ผ๋ฉฐ, ROC ๊ณก์ ์ ์๋์ ๊ฐ๋ค.**
| | **์์ ๋ชจํ**	| **์ถ์๋ชจํ** |
| ---- | ---- | ---- |  
| ์ ํ๋	| 0.8453159 |	0.8420479 |
| AUC๊ฐ	| 0.9116094	| 0.9065873 |

```javascript
plot(qda.perf1)
```
<img width="840" height="600" src="https://user-images.githubusercontent.com/100699925/158420899-5e6f9fb2-1297-4c96-a06e-0b52fcbd1ace.png">

<br/>

#### Naive Bayes
- 10-folds CV๋ฅผ ํ์ฉํ์ฌ Naive Bayes ๋ฐฉ๋ฒ์ ๋ชจํ๋ณ ์ ํ๋์ AUC๊ฐ ๊ณ์ฐ

<br/>

#### **1. ์์ ๋ชจํ ์ ํฉ**
**์ ํ๋**
```javascript
nb.prediction <- data.frame()
nb.testset <- data.frame()
nb.probability <- data.frame()

set.seed(2)
for (i in 1:k) {
  train.set = subset(heart, id %in% list[-i])
  test.set = subset(heart, id %in% c(i))
  nb.fit.k <- naiveBayes(formula(Fullmod), data = train.set)
  nb.pred.k <- as.data.frame(predict(nb.fit.k, newdata = test.set))
  nb.prob.k <- as.data.frame(predict(nb.fit.k, newdata = test.set, type = "raw"))
  
  nb.prediction <- rbind(nb.prediction, nb.pred.k)
  nb.probability <- rbind(nb.probability, nb.prob.k)
  nb.testset <- rbind(nb.testset, as.data.frame(test.set[,12]))
}
(nb.Full.acc <- mean(nb.prediction[,1] == nb.testset[,1]))
```
```
## [1] 0.8616558
```

<br/>

**AUC๊ฐ**
```javascript
nb.pr1 <- prediction(nb.probability[,2], nb.testset[,1])
nb.perf1 <- performance(nb.pr1, measure = "tpr", x.measure = "fpr")
nb.auc1 <- performance(nb.pr1, measure = "auc") ; (nb.Full.auc <- unlist(nb.auc1@y.values))
```
```
## [1] 0.9205973
```

<br/>

#### **1. ์ถ์๋ชจํ ์ ํฉ**
**์ ํ๋**
```javascript
nb.prediction <- data.frame()
nb.testset <- data.frame()
nb.probability <- data.frame()

set.seed(2)
for (i in 1:k) {
  train.set = subset(heart, id %in% list[-i])
  test.set = subset(heart, id %in% c(i))
  nb.fit.k <- naiveBayes(formula(mod), data = train.set)
  nb.pred.k <- as.data.frame(predict(nb.fit.k, newdata = test.set))
  nb.prob.k <- as.data.frame(predict(nb.fit.k, newdata = test.set, type = "raw"))
  
  nb.prediction <- rbind(nb.prediction, nb.pred.k)
  nb.probability <- rbind(nb.probability, nb.prob.k)
  nb.testset <- rbind(nb.testset, as.data.frame(test.set[,12]))
}
(nb.mod.acc <- mean(nb.prediction[,1] == nb.testset[,1]))
```
```
## [1] 0.8638344
```

<br/>

**AUC๊ฐ**
```javascript
nb.pr2 <- prediction(nb.probability[,2], nb.testset[,1])
nb.perf2 <- performance(nb.pr2, measure = "tpr", x.measure = "fpr")
nb.auc2 <- performance(nb.pr2, measure = "auc") ; (nb.mod.auc <- unlist(nb.auc2@y.values))
```
```
## [1] 0.9235068
```

<br/>

**Naive Bayes์์ ์ถ์๋ชจํ์ ์ ํฉํ์์ ๋์ AUC๊ฐ์ด ๋์์ ํ์ธํ  ์ ์์ผ๋ฉฐ, ROC ๊ณก์ ์ ์๋์ ๊ฐ๋ค.**
| | **์์ ๋ชจํ**	| **์ถ์๋ชจํ** |
| ---- | ---- | ---- |  
| ์ ํ๋	| 0.8616558	| 0.8638344 |
| AUC๊ฐ	| 0.9205973	| 0.9235068 |

```javascript
plot(nb.perf2)
```
<img width="840" height="600" src="https://user-images.githubusercontent.com/100699925/158428213-f9f5764d-18bc-4f55-9e59-3b2d6d4a365f.png">


<br/>

#### ๋ถ๋ฅ ๋ฐฉ๋ฒ๋ณ ๋น๊ต
<img width="840" height="600" src="https://user-images.githubusercontent.com/100699925/158420902-3b7c66ef-3ba7-4ef2-9bed-ec67bca4c634.png">

- ์ ๋ฐฉ๋ฒ๋ค ๋ณ๋ก ๊ฐ์ฅ ๋์ AUC๊ฐ๋ค์ ๊ฐ์ง๋ ๋ชจํ๋ค์ ROC ๊ณก์ ๊ณผ AUC๊ฐ๋ค์ ์์ ๊ทธ๋ฆผ๋ค๊ณผ ๊ฐ๋ค. ๋ฐ๋ผ์, ๋ก์ง์คํฑ, LDA, QDA, Naive Bayes๋ฐฉ๋ฒ๋ค ์ค AUC๊ฐ์ ๊ธฐ์ค์ผ๋ก ์ ์ผ ์ข์ ๋ถ๋ฅ๊ธฐ๋ LDA๋ผ๊ณ  ํ  ์ ์์ผ๋, ๋ก์ง์คํฑ ํ๊ท์ ํฐ ์ฐจ์ด๋ฅผ ๋ณด์ด์ง ์๋๋ค.

<br/>

### 2-2 Tree-Based Method

<br/>

#### Classification tree & Pruning
```javascript
tree.train.f <- tree(formula(Fullmod), data = heart, subset = train)
plot(tree.train.f) ; text(tree.train.f, pretty = 1, cex = 0.8)
```
<img width="840" height="600" src="https://user-images.githubusercontent.com/100699925/158420909-086c9d03-29df-4137-8729-00f242e48fcb.png">

<br/>

**์ ํ๋**
```javascript
tree.pred.f <- predict(tree.train.f, newdata = test.d)
(ctree.acc <- mean(round(tree.pred.f[,2]) == test.d$HeartDisease))
```
```
## [1] 0.8581818
```

<br/>

**AUC๊ฐ**
```javascript
tree.pr <- prediction(tree.pred.f[,2], test.d$HeartDisease)
tree.perf <- performance(tree.pr, measure = "tpr", x.measure = "fpr")
tree.auc <- performance(tree.pr, measure = "auc") ; (ctree.auc <- unlist(tree.auc@y.values))
```
```
## [1] 0.8972507
```

<br/>

#### 10-folds CV๋ฅผ ํ์ฉํ ์ต์ ์ Tree Pruning
```javascript
set.seed(6)
cv.tree.f <- cv.tree(tree.train.f, FUN = prune.misclass)
plot(cv.tree.f$size, cv.tree.f$dev, type = "b")
```
<img width="840" height="600" src="https://user-images.githubusercontent.com/100699925/158420910-c3b5c5f3-b825-4ab0-aa7d-1c7c9288aba0.png">

```javascript
plot(cv.tree.f$k, cv.tree.f$dev, type = "b")
```
<img width="840" height="600" src="https://user-images.githubusercontent.com/100699925/158420914-a7915a1f-1b99-4a8c-ba62-011e35c7b25a.png">

```javascript
cv.tree.f
```
```
## $size
## [1] 12  9  8  4  2  1
## 
## $dev
## [1] 123 123 117 111 118 287
## 
## $k
## [1]  -Inf   0.0   4.0   4.5   8.0 169.0
## 
## $method
## [1] "misclass"
## 
## attr(,"class")
## [1] "prune"         "tree.sequence"
```
**- ์ต์์ dev๊ฐ์ ๊ฐ์ง๋ penalty = 2.333, ๊ฐ์ง์ = 5**

<br/>

**10-folds CV๋ฅผ ํ์ฉํ์ฌ ํ์ํ Tree์ ์ต์  ๊ฐ์ง์์ ๋ง๊ฒ ๊ฐ์ง์น๊ธฐ ์คํ**

```javascript
prune.f <- prune.misclass(tree.train.f, best=5)
plot(prune.f) ; text(prune.f, pretty=0, cex=0.8)
```
<img width="840" height="600" src="https://user-images.githubusercontent.com/100699925/158420917-3b0a10a8-7ac7-47be-a572-207146fe1777.png">


**์ ํ๋**
```javascript
prune.pred <- predict(prune.f, test.d)
(pruned.acc <- mean(round(prune.pred[,2]) == test.d$HeartDisease))
```
```
## [1] 0.8363636
```

<br/>

**AUC๊ฐ**
```javascript
prune.pr <- prediction(prune.pred[,2], test.d$HeartDisease)
prune.perf <- performance(prune.pr, measure = "tpr", x.measure = "fpr")
prune.auc <- performance(prune.pr, measure = "auc") ; (pruned.auc <- unlist(prune.auc@y.values))
```
```
## [1] 0.8725931
```

<br/>

#### 10-folds CV๋ฅผ ํ์ฉํ ์ต์ ์ Bagging ๋ชจํ ํ์
```javascript
k = 10
list <- 1:k

set.seed(1)
id <- sample(1:k, nrow(heart), replace = T)
ntrees <- c(seq(100, 1000, 100),1500,2000)
auc <- c()

set.seed(1)
for (j in 1:length(ntrees)) {
  bag.prediction <- data.frame()
  bag.testset <- data.frame()
  for (i in 1:k) {
    train.set = subset(heart, id %in% list[-i])
    test.set = subset(heart, id %in% c(i))
    bag.fit <- randomForest(formula(Fullmod), data = train.set,
                            mtry=11, importance=T, ntree = ntrees[j])
    bag.prob <- as.data.frame(predict(bag.fit, newdata = test.set, type = "prob"))
  
    bag.prediction <- rbind(bag.prediction, bag.prob)
    bag.testset <- rbind(bag.testset, as.data.frame(test.set[,12]))
    
    bag.pr <- prediction(bag.prediction[,2], bag.testset)
    bag.auc <- performance(bag.pr, measure = "auc")
  }
  auc[j] <- as.numeric(bag.auc@y.values)

}
```

<br/>

**- ntree (์์ฑ ๋๋ฌด ์)๋ฅผ 200์ผ๋ก ์ค์ ํ  ๋, ๊ฐ์ฅ ๋์ ์ฑ๋ฅ์ ๊ฐ์ง์ ์ ์ ์๋ค.**
```javascript
plot(auc ~ ntrees, type = "b")
```
<img width="840" height="600" src="https://user-images.githubusercontent.com/100699925/158420918-5351bb6a-4053-4d2a-9175-47383f0cdc4e.png">


<br/>

#### **- 10-folds CV๋ฅผ ํตํด ํ์ํ ํ์ดํผ ํ๋ผ๋ฏธํฐ์ ๋ง๊ฒ Bagging ๋ชจํ์ ์์ฑํ๊ณ  ํ๋ จ๋ฐ์ดํฐ๋ฅผ ์ ํฉ** 
**์ ํ๋**
```javascript
set.seed(1)
bag.heart1 <- randomForest(formula(Fullmod), data=heart, subset = train,
                           mtry=11, importance=T, ntree=200)
bag.pred <- predict(bag.heart1, newdata = test.d)
(bag.acc <- mean(bag.pred == test.d$HeartDisease))
```
```
## [1] 0.8654545
```

<br/>

**AUC๊ฐ ๋ฐ ROC๊ณก์ **
```javascript
bag.prob <- predict(bag.heart1, newdata = test.d, type = "prob")
bag.pr <- prediction(bag.prob[,2], test.d$HeartDisease)
bag.perf <- performance(bag.pr, measure = "tpr", x.measure = "fpr")
bag.auc <- performance(bag.pr, measure = "auc") ; (bag.auc <- unlist(bag.auc@y.values))
```
```
## [1] 0.9252246
```

```javascript
plot(bag.perf)
```
<img width="840" height="600" src="https://user-images.githubusercontent.com/100699925/158420919-0a19abe7-5380-4ac2-8968-157b3e34375e.png">


<br/>

#### 10-folds CV๋ฅผ ํ์ฉํ ์ต์ ์ RandomForest ๋ชจํ ํ์ 
**m = sqrt(p) ๋ฅผ ํตํด ์ฌ์ฉ ๋ณ์ ์ค์  **

```javascript
set.seed(1)
id <- sample(1:k, nrow(heart), replace = T)
ntrees <- c(seq(100, 1000, 100),1500,2000)
auc3 <- c()

set.seed(1)
for (j in 1:length(ntrees)) {
  rf.prediction <- data.frame()
  rf.testset <- data.frame()
  for (i in 1:k) {
    train.set = subset(heart, id %in% list[-i])
    test.set = subset(heart, id %in% c(i))
    rf.fit <- randomForest(formula(Fullmod), data = train.set,
                            mtry=3, importance=T, ntree = ntrees[j])
    rf.prob <- as.data.frame(predict(rf.fit, newdata = test.set, type = "prob"))
    
    rf.prediction <- rbind(rf.prediction, rf.prob)
    rf.testset <- rbind(rf.testset, as.data.frame(test.set[,12]))
    
    rf.pr <- prediction(rf.prediction[,2], rf.testset)
    rf.auc <- performance(rf.pr, measure = "auc")
  }
  auc3[j] <- as.numeric(rf.auc@y.values)
  
}
```

**-ntree (์์ฑ ๋๋ฌด ์)๋ฅผ 1500์ผ๋ก ์ค์ ํ  ๋, ๊ฐ์ฅ ๋์ ์ฑ๋ฅ์ ๊ฐ์ง์ ์ ์ ์๋ค.**
```javascript
plot(auc3 ~ ntrees, type = "b")
```
<img width="840" height="600" src="https://user-images.githubusercontent.com/100699925/158420922-6655b48e-cd6e-473c-9c08-3113e9c6b2ef.png">

<br/>

#### **- 10-folds CV๋ฅผ ํตํด ํ์ํ ํ์ดํผ ํ๋ผ๋ฏธํฐ์ ๋ง๊ฒ RandomForest ๋ชจํ์ ์์ฑํ๊ณ  ํ๋ จ๋ฐ์ดํฐ๋ฅผ ์ ํฉ** 
**์ ํ๋**
```javascript
set.seed(1)
rf.heart <- randomForest(formula(Fullmod), data= heart, subset = train, 
                         mtry=3, importance=T, ntree=1500)
rf.pred <- predict(rf.heart, newdata = test.d)
(rf.acc <- mean(rf.pred == test.d$HeartDisease))
```
```
## [1] 0.8654545
```

<br/>

**AUC๊ฐ ๋ฐ ROC๊ณก์ **
```javascript
rf.prob <- predict(rf.heart, newdata = test.d, type = "prob")
rf.pr <- prediction(rf.prob[,2], test.d$HeartDisease)
rf.perf <- performance(rf.pr, measure = "tpr", x.measure = "fpr")
rf.auc <- performance(rf.pr, measure = "auc") ; (rf.auc <- unlist(rf.auc@y.values))
```
```
## [1] 0.9388104
```

```javascript
plot(rf.perf)
```
<img width="840" height="600" src="https://user-images.githubusercontent.com/100699925/158420925-1fc0eb4d-8507-47d5-897b-cbce7244453d.png">


|  | **Tree**	| **Pruned Tree**  |	**Bagging**	| **RandomForest** |
| ---- | ---- | ---- | ---- | ---- |
| ์ ํ๋	| 0.8581818	 | 0.8363636	| 0.8654545	| 0.8654545 |
| AUC๊ฐ	| 0.8972507	| 0.8725931	| 0.9252246	| 0.9388104 |

<img width="840" height="600" src="https://user-images.githubusercontent.com/100699925/158420927-2ec52344-07a5-46de-91dc-aebe6942066f.png">




- ์ ๋ฐฉ๋ฒ๋ค ๋ณ๋ก ๊ฐ์ฅ ๋์ AUC๊ฐ๋ค์ ๊ฐ์ง๋ ๋ชจํ๋ค์ ROC ๊ณก์ ๊ณผ AUC๊ฐ๋ค์ ์์ ๊ทธ๋ฆผ๋ค๊ณผ ๊ฐ๋ค. ๋ฐ๋ผ์, Tree๊ธฐ๋ฐ์ method๋ค ์ค AUC๊ฐ์ ๊ธฐ์ค์ผ๋ก ์ ์ผ ์ข์ ๋ถ๋ฅ๊ธฐ๋ RandomForest๋ผ๊ณ  ํ  ์ ์๋ค.

<br/>

### 2-3 Support Vector Machine

<br/>

#### 10-folds CV๋ฅผ ํ์ฉํ ์ต์ ์ Support Vector Machine ๋ชจํ ํ์
```javascript
costs = c(0.1,0.5, 1,10,100,500)
gammas = c(0.5, 1, 2, 3, 4, 5)
set.seed(1)
tune.out <- tune(svm, formula(Fullmod),
                  data = train.d, kernel = "radial",
                  ranges = list(cost = costs,
                                gamma = gammas))
(bestmod <- tune.out$best.model) 
```
```
## 
## Call:
## best.tune(method = svm, train.x = formula(Fullmod), data = train.d, 
##     ranges = list(cost = costs, gamma = gammas), kernel = "radial")
## 
## 
## Parameters:
##    SVM-Type:  C-classification 
##  SVM-Kernel:  radial 
##        cost:  1 
## 
## Number of Support Vectors:  461
```
```javascript
bestmod$cost ; bestmod$gamma
```
```
## [1] 1
## [1] 0.5
```

- margin์ ํฌ๊ธฐ์ ๊ด๋ จ ์๋ cost , fitting ์ ๋์ ๊ด๋ จ ์๋ gamma ์ ๊ฐ์ ์ ์ ํ ์ ์ ํด์ผ ํ๋ค. ์ด๋ฅผ ์ํด 10-fold Cross Validation์ ์ด์ฉํ๋ค. tuning ๊ฒฐ๊ณผ cost=1, gamma=0.5์ ๊ฐ์ด ์ต์ ์ ํ์ดํผ ํ๋ผ๋ฏธํฐ์์ ์ ์ ์๋ค.
- ํ์ํ ํ์ดํผ ํ๋ผ๋ฏธํฐ์ ๋ง๊ฒ Suppor Vector Machine ๋ชจํ์ ์์ฑํ๊ณ  ํ๋ จ๋ฐ์ดํฐ๋ฅผ ์ ํฉ

<br/>

**์ ํ๋**
```javascript
svm.fit <- svm(formula(Fullmod), data = train.d,
                kernel = "radial", cost = 1, gamma = 0.5,
                decision.values = T)
svm.pred <- predict(svm.fit, newdata = test.d, decision.values = T)
(svm.acc <- mean(svm.pred == test.d$HeartDisease))
```
```
## [1] 0.8690909
```

<br/>

**AUC๊ฐ ๋ฐ ROC๊ณก์ **
```javascript
fitted <- attributes(svm.pred)$decision.values
svm.pr <- prediction(-fitted, test.d$HeartDisease)
svm.perf <- performance(svm.pr, measure = "tpr", x.measure = "fpr")
svm.auc <- performance(svm.pr, measure = "auc") ; svm.auc <- unlist(svm.auc@y.values)
```

```javascript
plot(svm.perf)
```
<img width="840" height="600" src="https://user-images.githubusercontent.com/100699925/158420930-cddfc95f-247b-41ba-93b4-421b470d363a.png">


<br/>

## 3. ๊ฒฐ๋ก 

<br/>

- 2-1์์๋ LDA, 2-2์์๋ RandomForest ๋ชจํ์ด ์ต์  ๋ชจ๋ธ๋ก ํ๋จ๋์๋ค.
- ๊ฐ ํํธ์์ ์คํ๋ ์์ธก ๋ชจ๋ธ๋ค์ ์ ํ๋์ AUC ๊ฐ์ ์๋์ ๊ฐ์ผ๋ฉฐ, ๋ชจ๋ธ ์ฑ๋ฅ ํ๋จ ์ฒ๋์ธ AUC ๊ฐ์ด ๊ฐ์ฅ ํฐ LDA์ ์ฑ๋ฅ์ด ๊ฐ์ฅ ์ข๋ค๊ณ  ํ๋จ๋๋ค.

<br/>

**๋ชจํ๋ณ ์ ํ๋**
<img width="840" height="600" src="https://user-images.githubusercontent.com/100699925/158420932-3a7dbebf-3c29-4ef0-bdb0-2abab88bee36.png">

<br/>

**๋ชจํ๋ณ AUC๊ฐ**
<img width="840" height="600" src="https://user-images.githubusercontent.com/100699925/158420934-32c97c9e-310e-423f-b310-997dc4cf5e4e.png">

