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

#

### 3. R코드 : https://github.com/ChSSolee/002/blob/main/K-Folds%20CV%EB%A5%BC%20%ED%99%9C%EC%9A%A9%ED%95%9C%20%EC%B5%9C%EA%B3%A0%EC%84%B1%EB%8A%A5%20%EB%AA%A8%ED%98%95%20%ED%83%90%EC%83%89.Rmd
> #### html 파일 : 
