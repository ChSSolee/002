## 📌 K-Folds CV를 활용한 최고성능 모형 탐색
> #### 개요 : Heart Failure Prediction Dataset의 이진적 반응변수인 HeartDisease을 예측하는 다양한 통계학습 모델들을 10-folds CV를 통해 구현하고 적합하여, 이에 따른 정확도와 AUC값을 비교한다.

#

### 1. 제작 기간 & 참여 인원
> #### 2021.11 ~ 2021.11
> #### 3인
> #### 수행 역할 
>> 10-folds CV를 활용하여 Logistic Regression, LDA, QDA, Bagging, Random Forest의 최적 하이퍼 파라미터 탐색 및 모형생성  
>> 분석 결과 취합 및 보고서 작성

> #### 배운점
>> #### 이론적으로 이해하고 있던 다양한 데이터 마이닝 기법들을 활용하여, 실제 데이터에 활용할 수 있었던 기회였습니다. 
>> #### 또한 해당 프로젝트를 통해 머신러닝의 코딩과 실습능력을 함양시킬 수 있었습니다. 

#

### 2. 사용 프로그램 : R 4.0.0
#### 사용 라이브러리 : 
> library(ROCR)  
> library(MASS)  
> library(boot)  
> library(e1071)  
> library(tree)  
> library(gbm)  
> library(randomForest)  

<br/>

### 3. [R코드](https://github.com/ChSSolee/002/blob/main/K-Folds%20CV%EB%A5%BC%20%ED%99%9C%EC%9A%A9%ED%95%9C%20%EC%B5%9C%EA%B3%A0%EC%84%B1%EB%8A%A5%20%EB%AA%A8%ED%98%95%20%ED%83%90%EC%83%89.Rmd)

<br/>

## 1. 서론

<br/>

### 1-1 개요

- Heart Failure Prediction Dataset의 이진적 반응변수인 HeartDisease을 예측하는 다양한 통계학습 모델들을 **10-folds CV**를 통해 구현하고 적합하여, 이에 따른 정확도와 AUC값을 비교한다.

<br/>

- 라이브러리 호출
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

### 1-2 데이터 이해

1. 데이터 셋 : “Heart Failure Prediction Dataset”
2. 데이터의 개요 : 918명의 심장질환의 여부와 심장질환과 관련된 11개의 변수들로 구성. (총 12개)
3. 변수별 속성

- Age : 환자 나이 / 수치형
- Sex : 환자 성별 / 범주형 (M = 남성 / F = 여성) 
- ChestPainType : 환자 흉부 통증 유형 / 범주형 (TA = 전형적 협심증 / ATA = 비전형적 협심증 / NAP = 비협심증성 통증 / ASY = 무증상)
- RestingBP : 안정 혈압 / 수치형 
- Cholesterol : 혈청 콜레스테롤 / 수치형
- FastingBS : 공복 혈당 / 범주형 (1 = 공복 혈당 > 120 mg/dl / 0 = 그 외)
- RestingECG : 안정시 심전계 결과 / 범주형 (Normal = 정상 / ST = 비정상 ST-T 파동 / LVH : 좌심실 비대)
- MaxHR : 최대 심박수 / 수치형 
- ExerciseAnigma : 운동 유발 협심증 / 범주형 (Y = 증상 있음 / N = 증상 없음)
- Oldpeak : ST 분절 하강 정도 / 수치형
- ST_Slope : ST 분절 경사 / 범주형 (Up = 오르막 / Flat = 평면 / Down = 내리막)
- HeartDisease : 심장질환 / 범주형 (1 = 심장질환 / 0 = 정상)

<br/>

4. 변수별 상관관계 시각화
<img width="840" height="600" src="https://user-images.githubusercontent.com/100699925/158420885-0e4cdd82-323b-44d1-9cfd-ea3e61569d3c.png">

```javascript
heart <- read.csv("heart.csv", header = T)
plot(heart, panel = panel.smooth)
```

- 산점도를 통해 설명변수 간의 대략적인 상관관계를 파악할 수 있다. 위를 보아 설명변수 RestingBP, MaxHR 간에 상관관계가 있을 것으로 판단되며, 이후 분석에서는 보다 정확한 판단을 위해 전진선택법과 후진선택법을 이용하기로 한다.

<br/>

### 데이터 전처리 (결측값 확인 및 변수형태 설정)
- 분석을 위해 변수들의 형태를 수정한다.
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

- 훈련 데이터와 테스트 데이터를 7:3의 비율로 분할.
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

## 2. 본론

<br/>

### 2-1 Logistic & LDA & QDA & Naive Bayes
- 완전모형 (모든 변수를 설명변수로 가지는 모형)과 비교가 될 수 있는 모형 설정을 위해, step()함수를 통해 최저의 AIC값을 가지는 모형을 탐색
- 완전모형, 영모형 (상수항만을 설명 변수로 가지는 모형)에 각각 전진선택법과 후진제거법을 적용하여 나온 축소모형은 동일한 것으로 확인 되었으며, 아래와 같다. 
- HeartDisease ~ Age + Sex + ChestPainType + Cholesterol + FastingBS + ExerciseAngina + Oldpeak + ST_Slope

```javascript
Fullmod = glm(HeartDisease ~ ., data = train.d, family = binomial(link = "logit")) # 완전모형
Nullmod = glm(HeartDisease ~ 1, data = train.d, family = binomial(link = "logit")) # 영모형
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
mod = glm(formula(backward), data = train.d, family = binomial(link = "logit")) # 축소모형
formula(mod)
```
```
## HeartDisease ~ Age + Sex + ChestPainType + Cholesterol + FastingBS + ExerciseAngina + Oldpeak + ST_Slope
```


<br/>

#### 로지스틱 회귀
- 10-folds CV를 활용하여 로지스틱 방법의 모형별 정확도와 AUC값 계산

#### **1. 완전모형 적합**
**정확도**
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

**AUC값**
```javascript
glm.pr1 <- prediction(glm.prediction[,1], glm.testset[,1])
glm.perf1 <- performance(glm.pr1, measure = "tpr", x.measure = "fpr")
glm.auc1 <- performance(glm.pr1, measure = "auc") ; (LR.Full.auc <- unlist(glm.auc1@y.values))
```
```
## [1] 0.9244383
```


<br/>

#### **2. 축소모형 적합**
**정확도**
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

**AUC값**
```javascript
glm.pr2 <- prediction(glm.prediction[,1], glm.testset[,1])
glm.perf2 <- performance(glm.pr2, measure = "tpr", x.measure = "fpr")
glm.auc2 <- performance(glm.pr2, measure = "auc") ; (LR.mod.auc <- unlist(glm.auc2@y.values))
```
```
## [1] 0.9256386
```

<br/>

**로지스틱 회귀에서 축소모형을 적합하였을 때의 AUC값이 높음을 확인할 수 있으며, ROC 곡선은 아래와 같다.**

| | **완전모형**	| **축소모형** |
| ---- | ---- | ---- |  
| 정확도	| 0.8627451	| 0.8660131 |
| AUC값	| 0.9244383	| 0.9256386 |

```javascript
plot(glm.perf2)
```
<img width="840" height="600" src="https://user-images.githubusercontent.com/100699925/158420891-911c3d56-47f2-4f86-8b5b-826afda3ce28.png">



<br/>

#### LDA
- 10-folds CV를 활용하여 LDA 방법의 모형별 정확도와 AUC값 계산

<br/>

#### **1. 완전모형 적합**
**정확도**
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

**AUC값**
```javascript
lda.pr1 <- prediction(lda.prediction$posterior.1, lda.testset[,1])
lda.perf1 <- performance(lda.pr1, measure = "tpr", x.measure = "fpr")
lda.auc1 <- performance(lda.pr1, measure = "auc") ; (LDA.Full.auc <- unlist(lda.auc1@y.values))

```

<br/>

#### **2. 축소모형 적합**
**정확도**
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

**AUC값**
```javascript
lda.pr2 <- prediction(lda.prediction$posterior.1, lda.testset[,1])
lda.perf2 <- performance(lda.pr2, measure = "tpr", x.measure = "fpr")
lda.auc2 <- performance(lda.pr2, measure = "auc") ; (LDA.mod.auc <- unlist(lda.auc2@y.values))
```
```
## [1] 0.9259314
```

<br/>

**LDA에서 축소모형을 적합하였을 때의 AUC값이 높음을 확인할 수 있으며, ROC 곡선은 아래와 같다.**
| | **완전모형**	| **축소모형** |
| ---- | ---- | ---- |  
| 정확도	| 0.8649237	| 0.8714597 |
| AUC값	| 0.9250624	| 0.9259314 |


```javascript
plot(lda.perf2)
```
<img width="840" height="600" src="https://user-images.githubusercontent.com/100699925/158420895-3b3ce2ce-ac98-4f26-812e-3e13c95ef40a.png">


<br/>

#### QDA
- 10-folds CV를 활용하여 QDA 방법의 모형별 정확도와 AUC값 계산

<br/>

#### **1. 완전모형 적합**
**정확도**
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

**AUC값**
```javascript
qda.pr1 <- prediction(qda.prediction$posterior.1, qda.testset[,1])
qda.perf1 <- performance(qda.pr1, measure = "tpr", x.measure = "fpr")
qda.auc1 <- performance(qda.pr1, measure = "auc") ; (qda.Full.auc <- unlist(qda.auc1@y.values))
```
```
## [1] 0.9116094
```

<br/>

#### **1. 축소모형 적합**
**정확도**
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

**AUC값**
```javascript
qda.pr2 <- prediction(qda.prediction$posterior.1, qda.testset[,1])
qda.perf2 <- performance(qda.pr2, measure = "tpr", x.measure = "fpr")
qda.auc2 <- performance(qda.pr2, measure = "auc") ; (qda.mod.auc <- unlist(qda.auc2@y.values))
```
```
## [1] 0.9065873
```

<br/>

**QDA에서 완전모형을 적합하였을 때의 AUC값이 높음을 확인할 수 있으며, ROC 곡선은 아래와 같다.**
| | **완전모형**	| **축소모형** |
| ---- | ---- | ---- |  
| 정확도	| 0.8453159 |	0.8420479 |
| AUC값	| 0.9116094	| 0.9065873 |

```javascript
plot(qda.perf1)
```
<img width="840" height="600" src="https://user-images.githubusercontent.com/100699925/158420899-5e6f9fb2-1297-4c96-a06e-0b52fcbd1ace.png">

<br/>

#### Naive Bayes
- 10-folds CV를 활용하여 Naive Bayes 방법의 모형별 정확도와 AUC값 계산

<br/>

#### **1. 완전모형 적합**
**정확도**
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

**AUC값**
```javascript
nb.pr1 <- prediction(nb.probability[,2], nb.testset[,1])
nb.perf1 <- performance(nb.pr1, measure = "tpr", x.measure = "fpr")
nb.auc1 <- performance(nb.pr1, measure = "auc") ; (nb.Full.auc <- unlist(nb.auc1@y.values))
```
```
## [1] 0.9205973
```

<br/>

#### **1. 축소모형 적합**
**정확도**
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

**AUC값**
```javascript
nb.pr2 <- prediction(nb.probability[,2], nb.testset[,1])
nb.perf2 <- performance(nb.pr2, measure = "tpr", x.measure = "fpr")
nb.auc2 <- performance(nb.pr2, measure = "auc") ; (nb.mod.auc <- unlist(nb.auc2@y.values))
```
```
## [1] 0.9235068
```

<br/>

**Naive Bayes에서 축소모형을 적합하였을 때의 AUC값이 높음을 확인할 수 있으며, ROC 곡선은 아래와 같다.**
| | **완전모형**	| **축소모형** |
| ---- | ---- | ---- |  
| 정확도	| 0.8616558	| 0.8638344 |
| AUC값	| 0.9205973	| 0.9235068 |

```javascript
plot(nb.perf2)
```
<img width="840" height="600" src="">


<br/>

#### 분류 방법별 비교
<img width="840" height="600" src="https://user-images.githubusercontent.com/100699925/158420902-3b7c66ef-3ba7-4ef2-9bed-ec67bca4c634.png">

- 위 방법들 별로 가장 높은 AUC값들을 가지는 모형들의 ROC 곡선과 AUC값들은 위의 그림들과 같다. 따라서, 로지스틱, LDA, QDA, Naive Bayes방법들 중 AUC값을 기준으로 제일 좋은 분류기는 LDA라고 할 수 있으나, 로지스틱 회귀와 큰 차이를 보이지 않는다.

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

**정확도**
```javascript
tree.pred.f <- predict(tree.train.f, newdata = test.d)
(ctree.acc <- mean(round(tree.pred.f[,2]) == test.d$HeartDisease))
```
```
## [1] 0.8581818
```

<br/>

**AUC값**
```javascript
tree.pr <- prediction(tree.pred.f[,2], test.d$HeartDisease)
tree.perf <- performance(tree.pr, measure = "tpr", x.measure = "fpr")
tree.auc <- performance(tree.pr, measure = "auc") ; (ctree.auc <- unlist(tree.auc@y.values))
```
```
## [1] 0.8972507
```

<br/>

#### 10-folds CV를 활용한 최적의 Tree Pruning
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
**- 최소의 dev값을 가지는 penalty = 2.333, 가지수 = 5**

<br/>

**10-folds CV를 활용하여 탐색한 Tree의 최적 가지수에 맞게 가지치기 실행**

```javascript
prune.f <- prune.misclass(tree.train.f, best=5)
plot(prune.f) ; text(prune.f, pretty=0, cex=0.8)
```
<img width="840" height="600" src="https://user-images.githubusercontent.com/100699925/158420917-3b0a10a8-7ac7-47be-a572-207146fe1777.png">


**정확도**
```javascript
prune.pred <- predict(prune.f, test.d)
(pruned.acc <- mean(round(prune.pred[,2]) == test.d$HeartDisease))
```
```
## [1] 0.8363636
```

<br/>

**AUC값**
```javascript
prune.pr <- prediction(prune.pred[,2], test.d$HeartDisease)
prune.perf <- performance(prune.pr, measure = "tpr", x.measure = "fpr")
prune.auc <- performance(prune.pr, measure = "auc") ; (pruned.auc <- unlist(prune.auc@y.values))
```
```
## [1] 0.8725931
```

<br/>

#### 10-folds CV를 활용한 최적의 Bagging 모형 탐색
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

**-ntree (생성 나무 수)를 200으로 설정할 때, 가장 높은 성능을 가짐을 알 수 있다.**
```javascript
plot(auc ~ ntrees, type = "b")
```
<img width="840" height="600" src="https://user-images.githubusercontent.com/100699925/158420918-5351bb6a-4053-4d2a-9175-47383f0cdc4e.png">


<br/>

#### **- 10-folds CV를 통해 탐색한 하이퍼 파라미터에 맞게 Bagging 모형을 생성하고 훈련데이터를 적합** 
**정확도**
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

**AUC값 및 ROC곡선**
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

#### 10-folds CV를 활용한 최적의 RandomForest 모형 탐색 
**- m \approx \sqrt(p) 를 통해 사용 변수 설정 **

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

**-ntree (생성 나무 수)를 1500으로 설정할 때, 가장 높은 성능을 가짐을 알 수 있다.**
```javascript
plot(auc3 ~ ntrees, type = "b")
```
<img width="840" height="600" src="https://user-images.githubusercontent.com/100699925/158420922-6655b48e-cd6e-473c-9c08-3113e9c6b2ef.png">

<br/>

#### **- 10-folds CV를 통해 탐색한 하이퍼 파라미터에 맞게 RandomForest 모형을 생성하고 훈련데이터를 적합** 
**정확도**
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

**AUC값 및 ROC곡선**
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
| 정확도	| 0.8581818	 | 0.8363636	| 0.8654545	| 0.8654545 |
| AUC값	| 0.8972507	| 0.8725931	| 0.9252246	| 0.9388104 |

<img width="840" height="600" src="https://user-images.githubusercontent.com/100699925/158420927-2ec52344-07a5-46de-91dc-aebe6942066f.png">




- 위 방법들 별로 가장 높은 AUC값들을 가지는 모형들의 ROC 곡선과 AUC값들은 위의 그림들과 같다. 따라서, Tree기반의 method들 중 AUC값을 기준으로 제일 좋은 분류기는 RandomForest라고 할 수 있다.

<br/>

### 2-3 Support Vector Machine

<br/>

#### 10-folds CV를 활용한 최적의 Support Vector Machine 모형 탐색
```javascript
costs = c(0.1,0.5, 1,10,100,500)
gammas = c(0.5, 1, 2, 3, 4, 5)
set.seed(1)
tune.out <- tune(svm, formula(Fullmod),
                  data = train.d, kernel = "radial",
                  ranges = list(cost = costs,
                                gamma = gammas))
(bestmod <- tune.out$best.model) 
bestmod$cost ; bestmod$gamma
```

- margin의 크기와 관련 있는 cost , fitting 정도와 관련 있는 gamma 의 값을 적절히 선정해야 한다. 이를 위해 10-fold Cross Validation을 이용한다. tuning 결과 cost=1, gamma=0.5의 값이 최적의 하이퍼 파라미터임을 알 수 있다.
- 탐색한 하이퍼 파라미터에 맞게 Suppor Vector Machine 모형을 생성하고 훈련데이터를 적합

<br/>

**정확도**
```javascript
svm.fit <- svm(formula(Fullmod), data = train.d,
                kernel = "radial", cost = 1, gamma = 0.5,
                decision.values = T)
svm.pred <- predict(svm.fit, newdata = test.d, decision.values = T)
(svm.acc <- mean(svm.pred == test.d$HeartDisease))
```

<br/>

**AUC값 및 ROC곡선**
```javascript
fitted <- attributes(svm.pred)$decision.values
svm.pr <- prediction(-fitted, test.d$HeartDisease)
svm.perf <- performance(svm.pr, measure = "tpr", x.measure = "fpr")
svm.auc <- performance(svm.pr, measure = "auc") ; svm.auc <- unlist(svm.auc@y.values)
```

```{r pressure50, echo = F}
plot(svm.perf)
```

<br/>

## 3. 결론

<br/>

- 2-1에서는 LDA, 2-2에서는 RandomForest 모형이 최적 모델로 판단되었다.
- 각 파트에서 실행된 예측 모델들의 정확도와 AUC 값은 아래와 같으며, 모델 성능 판단 척도인 AUC 값이 가장 큰 LDA의 성능이 가장 좋다고 판단된다.

<br/>

**모형별 정확도**
```{r echo = FALSE, results = 'axis'}
ACC.1 <- c(LR.mod.acc, LDA.mod.acc, qda.Full.acc, nb.mod.acc, ctree.acc, pruned.acc, bag.acc, rf.acc, svm.acc)
name.1 <- c("로지스틱", "LDA", "QDA", "NB", "Tree", "Pruned", "Bag", "RF", "SVM")
bp <- barplot(ACC.1, names.arg = name.1, col=c(0,21,rep(0,7)), ylim = c(0,1), ylab = "정확도")
text(x=bp, y=ACC.1, labels = round(ACC.1,5), cex = 0.8)
```

<br/>

**모형별 AUC값**
```{r echo = FALSE, results = 'axis'}
AUC.1 <- c(LR.mod.auc, LDA.mod.auc, qda.Full.auc, nb.mod.auc, ctree.auc, pruned.auc, bag.auc, rf.auc, svm.auc)
name.1 <- c("로지스틱", "LDA", "QDA", "NB", "Tree", "Pruned", "Bag", "RF", "SVM")
bp <- barplot(AUC.1, names.arg = name.1, col=c(0,21,rep(0,7)), ylim = c(0,1), ylab = "정확도")
text(x=bp, y=AUC.1, labels = round(ACC.1,5), cex = 0.8)
```
