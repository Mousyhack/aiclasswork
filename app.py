from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from pyproj import Proj, transform

app = Flask(__name__)
CORS(app)  # HTML에서 요청 허용

# 저장된 모델 불러오기
try:
    model = joblib.load('crime_model.pkl')
except FileNotFoundError:
    print("오류: 'crime_model.pkl' 파일을 찾을 수 없습니다. train_model.py를 먼저 실행했는지 확인하세요.")
    exit()


# 좌표 변환 설정 (WGS84 위도경도 -> UTM Zone 10N 밴쿠버 지역)
# 데이터의 X, Y 좌표가 밴쿠버 지역 UTM 좌표계라고 가정합니다 (EPSG:26910)
# EPSG:4326 (위도, 경도) -> EPSG:26910 (UTM X, Y)로 변환합니다.
proj_wgs84 = Proj(init='epsg:4326') 
proj_utm = Proj(init='epsg:26910')  

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        lat = float(data['latitude'])
        lng = float(data['longitude'])
        hour = int(data['hour'])
        month = int(data['month'])

        # 1. 위도/경도를 데이터셋과 같은 X, Y 좌표로 변환 (경도, 위도 순서 유의)
        x, y = transform(proj_wgs84, proj_utm, lng, lat)

        # 2. 예측을 위한 데이터 프레임 생성
        input_data = pd.DataFrame([[x, y, hour, month]], columns=['X', 'Y', 'HOUR', 'MONTH'])

        # 3. 예측 (범죄 발생 확률)
        # predict_proba는 [[비범죄확률, 범죄확률]] 을 반환합니다. 우리는 두 번째 값(1, 범죄 발생)을 사용합니다.
        prediction = model.predict_proba(input_data)
        crime_probability = prediction[0][1] * 100  # 퍼센트로 변환

        return jsonify({
            'probability': round(crime_probability, 2),
            'location_x': x,
            'location_y': y
        })
        
    except Exception as e:
        # 오류 발생 시 디버깅을 위해 오류 메시지를 반환합니다.
        return jsonify({'error': str(e)}), 500

# app.py 맨 아래 부분 수정
import os

if __name__ == '__main__':
    # 클라우드 환경에서는 PORT 환경변수를 사용, 없으면 5000번 사용
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)