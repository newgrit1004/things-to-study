## Table of Contents

- [Tabular Dataset이란?](#1)
- [Tabular Dataset에서 머신러닝 방법과 딥러닝 방법 성능 비교 실험](#2)

## #1
### Tabular Dataset이란?
* 정의
    * 테이블 형태로 나타낼 수 있는 정형 데이터를 의미
    * 관계형 데이터베이스의 테이블 형태로 표현될 수 있는 데이터

* 특징
    * 딥러닝 모델보다는 트리 기반의 머신러닝 알고리즘(CatBoost, XGBoost, LightGBM) 등의 성능이 일반적으로 더 좋았음

* Tabular Dataset을 활용한 Deep Learning 방법 분류
    * Data Transformation Methods
        * Encoding
    * Specialized Architectures for tabular datasets
        * Hybrid Models
            * machine learning + NN
        * Transformer-based models
    * Regularization Models


## #2
### Tabular Dataset에서 머신러닝 방법과 딥러닝 방법 성능 비교 실험

* 실험 1(Deep Neural Networks and Tabular Data : A Survey)
    * 데이터셋
        * 3개의 binary dataset(HELOC, AdultIncome, HIGGS)
        * 1개의 multi-class dataset(Covertype)
        * 1개의 regression task(California Housing)

    * 실험 결과
        * 성능(Performance on Accuracy and AUC)
            * 5개의 데이터셋 중 4개의 데이터셋에서는 트리기반 모델(CatBoost, XGBoost, LightGBM)의 성능이 가장 좋았으며, 1개의 데이터셋(HIGGS)에서 딥러닝 모델인 SAINT의 성능이 가장 좋았음.
        * 학습 및 추론 속도(Training and Inference speed)
            * 3만개 샘플 데이터(binary)
                * 학습 속도 : Decision Tree < Random Forest < XGBoost < CatBoost < DeepFM < SAINT < TabNet
                * 추론 속도 : Decision Tree < DeepFM < CatBoost < Random Forest < XGBoost < SAINT
            * 1100만개 샘플 데이터(binary)
                * 학습 속도 : Decision Tree < XGBoost < Random Forest < CatBoost < DeepFM < TabNet < SAINT
                * 추론 속도 : Decision Tree < DeepFM < Random Forest < SAINT < TabNet < CatBoost < XGBoost

* 실험 2(TABULAR DATA: DEEP LEARNING IS NOT ALL YOU NEED)
    * 데이터셋
        * 3개의 페이퍼(DNF-Net, NODE, TabNet)에서 사용된 데이터셋 9개 + 새로운 데이터셋 2개(from Kaggle)
    * 실험 결과
        * 1등 개수
            * Deep learning + XGBoost ensemble 모델 7번
            * TabNet 2번(TabNet에서 사용된 데이터셋에서 2번)
            * NODE 1번(Node에서 사용된 데이터셋에서 1번)
            * DNF-Net 1번(DNF-Net에서 사용된 데이터셋에서 1번)

* 실험 3(Why do tree-based models still outperform deep learning on tabular data?)
    * 데이터셋
        * 총 45개의 데이터셋
        * Task
            * Numerical classification
            * Numerical regression
            * Categorical classification
            * Categorical regression

    * 실험 결과
        * numerical classification, regression 비교 실험에서는 XGBoost, Random Forest 등이 딥러닝 모델보다 성능이 좋았다.
        * numerical feature와 categorical feature가 섞인 classification, regression 비교 실험에서는 Random Forest, Gradient Boosting 등 tree-based model의 성능이 딥러닝 모델보다 더 좋았다.
    * 왜 tabular dataset에서는 tree-based model이 딥러닝 모델보다 성능이 좋을까?
        * Tabular Data에는 의미 없는 feature가 존재하고, 이것이 neural network에 큰 영향을 미친다. Tree 계열 모델들은 의미 없는 feature들에 대해 robust하다.
        * Random Forest feature importance가 높은 feature 순서대로 feature를 제거한 뒤 모델간의 성능을 비교하였을 경우에는 tree-based model이나 딥러닝 모델이나 성능이 비슷하게 떨어진다.
        * <b>의미없는 feature을 tabular dataset에 추가했을 경우, 딥러닝 모델은 성능이 크게 떨어진다.</b>

# References
- [[Open DMQA Seminar] Comparison of Machine Learning and Deep Learning for Tabular Datasets](https://www.youtube.com/watch?v=9tQqjO5C-jg)
- [Deep Neural Networks and Tabular Data : A Survey](https://arxiv.org/abs/2110.01889)
- [Tabular Data: Deep Learning is Not All You Need](https://arxiv.org/abs/2106.03253)
- [Why do tree-based models still outperform deep learning on tabular data?](https://arxiv.org/abs/2207.08815)