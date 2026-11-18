# 머신러닝 맛보기 — 실습편

**Python 프로그래밍 특강** · 이수환

가상의 건강검진 데이터 200명분으로 **전처리 → 회귀 → 분류 → 과대적합**까지, scikit-learn을 이용한 머신러닝의 기본 흐름을 직접 따라가 보는 실습입니다.
이론편에서 본 개념(학습·예측, 인코딩, 결측치, 학습/테스트 분할, 과대적합)을 코드로 확인합니다.

## 파일 구성

| 파일 | 설명 |
| --- | --- |
| [`ml_practice.ipynb`](ml_practice.ipynb) | **학생용 노트북**. 빈칸 `___`을 채우면서 진행합니다. |
| [`ml_practice_정답.ipynb`](ml_practice_정답.ipynb) | **정답 노트북** (실행 결과 포함). 막혔을 때 확인하세요. |
| [`health.csv`](health.csv) | 실습 데이터 |
| [`requirements.txt`](requirements.txt) | 필요한 라이브러리 목록 |

> 노트북은 같은 폴더의 `health.csv`를 읽습니다. **세 파일을 같은 폴더에** 두고 실행하세요.

## 시작하기

### 1. 내려받기

```bash
git clone https://github.com/SHwaNy/Introduction-to-Machine-Learning.git
cd Introduction-to-Machine-Learning
```

또는 GitHub 페이지의 **Code → Download ZIP**으로 받은 뒤 압축을 풉니다.

### 2. 라이브러리 설치

필요한 라이브러리는 `pandas`, `scikit-learn`, `matplotlib`입니다. 둘 중 편한 방법을 쓰세요.

- **노트북 안에서**: 노트북의 `0. 환경 준비`에 있는 `%pip install ...` 셀을 실행합니다.
  이미 설치되어 있으면(Anaconda 등) `Requirement already satisfied`가 나오고 넘어갑니다.
- **터미널에서**:
  ```bash
  pip install -r requirements.txt
  ```

### 3. 실행

```bash
jupyter notebook ml_practice.ipynb
```

위에서부터 셀을 하나씩 실행합니다 (`Shift + Enter`). 첫 코드 셀이 Windows · Mac · Linux를 자동으로 구분해 그래프의 한글 폰트를 설정합니다.

> **일부러 에러가 나는 셀이 두 개 있습니다** (실습 2, 실습 4). 그래서 *Run All*로 한 번에 실행하면 중간에 멈춥니다. 셀을 하나씩 실행하면서 에러 메시지의 마지막 줄을 읽어 보세요.

<details>
<summary>Google Colab에서 실행하려면</summary>

1. [Colab](https://colab.research.google.com/)에서 **파일 → 노트북 업로드**로 `ml_practice.ipynb`를 엽니다.
2. 왼쪽 **파일(📁)** 패널에 `health.csv`를 업로드합니다.
3. 위에서부터 셀을 하나씩 실행합니다.

Colab에는 한글 폰트가 없어서 그래프의 한글이 네모(□)로 보일 수 있습니다. 계산 결과에는 영향이 없습니다.
</details>

## 오늘의 데이터: `health.csv`

이론편 Section 2에서 본 건강검진 표를 200명으로 늘린 데이터입니다. 처음 5행은 이론편 13장의 표와 같습니다.

| 열 | 설명 | 데이터 종류 | 실습에서의 역할 |
| --- | --- | --- | --- |
| 키 | cm | 수치형 · 연속형 | X |
| 몸무게 | kg | 수치형 · 연속형 | X |
| 체지방 | %, **빈칸 10개** | 수치형 · 연속형 | X |
| 혈액형 | A · B · O · AB | 범주형 · 명목척도 | 인코딩 연습 |
| 혈압 | mmHg | 수치형 · 연속형 | 회귀의 y |
| 고혈압 | Y · N | 범주형 · 명목척도 | 분류의 y |

> ⚠️ **실습용 가상 데이터**입니다. 실제 사람의 기록이 아닙니다.
> 고혈압은 혈압 125 이상을 Y로 정한 **실습용 기준**이며, 실제 의학 기준과 다릅니다.

## 머신러닝 실습 5단계

오늘 하는 모든 실습은 이 다섯 단계를 따릅니다.

| 단계 | 하는 일 | 주요 코드 | 이론편 |
| :---: | --- | --- | --- |
| 1 | 불러오기 | `read_csv` | p.13 |
| 2 | 전처리 | `fillna` · `map` · `get_dummies` | p.18–19 |
| 3 | X · y 나누기 | `df[[...]]` · `df[...]` | p.16–17 |
| 4 | 학습/테스트 분할 | `train_test_split` | p.20 |
| 5 | 학습 · 예측 · 평가 | `fit` · `predict` · `score` | p.9, p.30 |

scikit-learn은 모델이 수백 가지여도 사용법이 같습니다. 모델 이름만 바꾸면 나머지 코드는 그대로 씁니다.

```python
model = LinearRegression()          # ① 모델 만들기
model.fit(X_train, y_train)         # ② 학습
pred = model.predict(X_test)        # ③ 예측
model.score(X_test, y_test)         # ④ 평가
```

## 실습 구성

노트북의 각 실습 제목에 슬라이드 번호(#)가 함께 적혀 있습니다.

### Part 1. 데이터 준비 — 에러로 배우는 전처리 (슬라이드 #6–13)

| 실습 | 내용 |
| --- | --- |
| 실습 1 | 데이터 살펴보기: `head()`, `shape`, `info()` → 체지방만 Non-Null 190 |
| 실습 2 | 전처리 없이 학습 → ❗ `ValueError: Input X contains NaN.` |
| 실습 3 | 결측치 처리: 체지방 빈칸을 중앙값(18.0)으로 채우기 |
| 실습 4 | 혈액형도 X에 넣기 → ❗ `ValueError: could not convert string to float: 'A'` |
| 실습 5 | 인코딩: 고혈압은 라벨 인코딩(N→0, Y→1), 혈액형은 원-핫 인코딩 |

두 번의 에러가 곧 두 가지 전처리입니다. 빈칸은 **결측치 처리**로, 문자는 **인코딩**으로 해결합니다.

### Part 2. 회귀 — 혈압 예측 (슬라이드 #14–21)

| 실습 | 내용 |
| --- | --- |
| 실습 6 | X(키·몸무게·체지방)와 y(혈압) 나누기, 학습 160명 / 테스트 40명으로 분할 |
| 실습 7 | 선형 회귀로 학습 · 예측 · 평가 (MAE, R²) |
| 실습 8 | 모델이 찾은 규칙(계수) 보기, 혈액형만으로 예측하면? |
| 실습 9 | 내 키·몸무게·체지방으로 혈압 예측하기 |

### Part 3. 분류 — 고혈압 예측 (슬라이드 #22–26)

| 실습 | 내용 |
| --- | --- |
| 실습 10 | X는 그대로, y와 모델만 바꾸기 (`DecisionTreeClassifier`) · 정확도의 함정 |
| 실습 11 | 결정 트리가 찾은 규칙 그려 보기, '나'는 트리의 어디로 갈까? |

> 생각해 보기: 왜 `혈압`은 X에 넣으면 안 될까요? (고혈압은 혈압으로 정해진 값이라, 답을 보고 문제를 푸는 셈이 됩니다.)

### Part 4. 좋은 모델 찾기 — 과대적합과 최적의 깊이 (슬라이드 #27–28)

| 실습 | 내용 |
| --- | --- |
| 실습 12 | `max_depth`를 1–10으로 바꿔 가며 학습/테스트 정확도 비교 → 과대적합 확인 |
| 실습 13 | 테스트 정확도가 가장 높은 깊이로 최종 모델 만들기 |

`max_depth`처럼 사람이 정해 주는 모델 설정값을 **하이퍼파라미터**라고 합니다. 이 값을 고르는 것도 머신러닝의 일부입니다.

<details>
<summary>주요 결과 (스스로 실행한 뒤 확인해 보세요)</summary>

| 항목 | 결과 |
| --- | --- |
| 체지방 중앙값 | 18.0 |
| 선형 회귀 MAE / R² | 4.9 / 0.67 |
| 오차 5 이내 / 10 이내 | 25명 / 34명 (테스트 40명 중) |
| 찾은 규칙 | 혈압 ≈ 72.4 + 0.03×키 + 0.31×몸무게 + 0.98×체지방 |
| 혈액형만 썼을 때 R² | 0.06 |
| 기본 트리 (깊이 14) 학습 / 테스트 정확도 | 1.000 / 0.775 |
| 전부 N으로 찍었을 때 정확도 | 0.775 |
| 가장 좋은 깊이 / 최종 테스트 정확도 | 3 / 0.925 |

</details>

## 정리: 전체 흐름 한눈에

노트북 마지막 셀입니다. 오늘 한 일은 이 몇 줄로 요약됩니다.

```python
df = pd.read_csv('health.csv')                                 # 1. 불러오기
df['체지방'] = df['체지방'].fillna(df['체지방'].median())         # 2. 전처리

X = df[['키', '몸무게', '체지방']]                                 # 3. X, y 나누기
y = df['혈압']

X_train, X_test, y_train, y_test = train_test_split(              # 4. 분할
    X, y, test_size=0.2, random_state=42)

model = LinearRegression()                                        # 5. 학습·예측·평가
model.fit(X_train, y_train)
model.score(X_test, y_test)
```

## 더 해보기

- **모델 바꿔 보기**: `RandomForestClassifier`, `KNeighborsClassifier`로 바꾸고 정확도 비교
- **실제 데이터 도전**: Kaggle [Titanic](https://www.kaggle.com/c/titanic) 생존자 예측
- **공식 튜토리얼**: [scikit-learn.org/stable/tutorial](https://scikit-learn.org/stable/tutorial/)
