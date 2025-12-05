import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. 데이터 로드 (파일 경로를 실제 위치에 맞게 수정하세요)
# 업로드된 파일 이름이 '범죄 3.zip/crime_records.csv' 이므로 압축을 풀거나 경로를 맞춰주세요.
df = pd.read_csv('crime_records.csv')

# 2. 데이터 전처리
# 필요한 컬럼만 선택 (X, Y좌표, 시간, 월)
data = df[['X', 'Y', 'HOUR', 'MONTH']].dropna()

# 3. 학습 데이터 생성 (범죄 발생 vs 비발생)
# 현재 데이터는 모두 '범죄가 발생한' 데이터입니다.
# '범죄가 발생하지 않음'을 학습시키기 위해 임의의 랜덤 데이터(Negative Sample)를 생성합니다.
# 이렇게 해야 AI가 "여기는 위험하다/안전하다"를 판단할 수 있습니다.

# 범죄 데이터 (레이블 1)
crime_data = data.copy()
crime_data['target'] = 1 

# 비범죄 데이터 생성 (레이블 0) - 범죄 데이터의 범위 내에서 랜덤 생성
non_crime_data = pd.DataFrame()
non_crime_data['X'] = np.random.uniform(data['X'].min(), data['X'].max(), len(data))
non_crime_data['Y'] = np.random.uniform(data['Y'].min(), data['Y'].max(), len(data))
non_crime_data['HOUR'] = np.random.randint(0, 24, len(data))
non_crime_data['MONTH'] = np.random.randint(1, 13, len(data))
non_crime_data['target'] = 0

# 데이터 합치기
train_data = pd.concat([crime_data, non_crime_data])

# 4. 모델 학습 (Random Forest)
X = train_data[['X', 'Y', 'HOUR', 'MONTH']]
y = train_data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 정확도 출력
print(f"모델 정확도: {model.score(X_test, y_test):.4f}")

# 5. 모델 저장
joblib.dump(model, 'crime_model.pkl')
print("모델이 'crime_model.pkl'로 저장되었습니다.")