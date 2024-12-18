from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import requests
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('nuri_amend.html')  # templates/index.html 파일을 서빙


# 1권역 페이지
@app.route('/zone1')
def zone1_page():
    return render_template('zone1.html')

# 2권역 페이지
@app.route('/zone2')
def zone2_page():
    return render_template('zone2.html')

# 홍기님제작
@app.route('/zone3')
def zone3_page():
    return render_template('index.html')


# device 변수 정의
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 클래스 정의 (저장된 모델과 동일한 구조)
class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.lstm = nn.LSTM(input_size=171, hidden_size=256, num_layers=3, batch_first=True)
        self.multioutput_reg = nn.Sequential(
            nn.Linear(in_features=256, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=168),
        )

    def forward(self, x):
        hidden, _ = self.lstm(x)
        output = self.multioutput_reg(hidden[:, -1, :])
        return output


# 실시간 재고 데이터를 가져오는 클래스
BIKE_API_KEY = '787a4c4f41736d6133365464694c56'
station_id_file = 'station_id_list.txt'

class BikeStationChecker:

    def __init__(self, BIKE_API_KEY, station_id_file):
        self.api_key = BIKE_API_KEY
        self.station_id_file = station_id_file
        self.station_id_list = self.load_station_id_list()

    def load_station_id_list(self):
        with open(self.station_id_file, 'r') as fr:
            station_id_list = [line.strip() for line in fr]
        return station_id_list

    def get_bike_stock_api(self):
        # 1000에서 2000까지
        response_data_1 = requests.get(f'http://openapi.seoul.go.kr:8088/{self.api_key}/json/bikeList/1001/2000/')
        bike_data_1 = response_data_1.json()
        bike_infos_1 = bike_data_1['rentBikeStatus']['row']

        # 2000에서 3000까지
        response_data_2 = requests.get(f'http://openapi.seoul.go.kr:8088/{self.api_key}/json/bikeList/2001/3000/')
        bike_data_2 = response_data_2.json()
        bike_infos_2 = bike_data_2['rentBikeStatus']['row']

        # 결합 
        total_bike_infos = bike_infos_1 + bike_infos_2
        return total_bike_infos

@app.route('/process_data', methods=['GET'])
def process_data():
    # 실시간 재고 데이터를 가져오기
    checker = BikeStationChecker(BIKE_API_KEY, station_id_file)
    bike_stock_data = checker.get_bike_stock_api()

    # 필요한 데이터 추출: stationId와 parkingBikeTotCnt
    processed_data = []
    for station in bike_stock_data:
        station_id = station['stationId']
        current_stock = int(station['parkingBikeTotCnt'])  # 현재 재고량을 정수형으로 변환

        # 모델에 필요한 데이터 형식으로 정리 (예: 딕셔너리 리스트)
        processed_data.append({
            'station_id': station_id,
            'current_stock': current_stock
        })

    # 데이터프레임으로 변환
    df = pd.DataFrame(processed_data)
    
    # station_id를 컬럼으로 변환
    df_transformed = df.set_index('station_id').T

    # DataFrame을 JSON으로 변환
    result = df_transformed.to_json()

    # 변환된 데이터를 JSON으로 반환
    return result


# 모델 로드 및 초기화
model = BaseModel()
model.load_state_dict(torch.load('./model/LSTM_Baseline_model.pth', map_location=device))
model.to(device)
model.eval()  # 평가 모드로 전환

# 예측 API 엔드포인트 설정
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # JSON 포맷으로 입력 받음
    input_data = np.array(data['input']).astype(float)
    input_tensor = torch.Tensor(input_data).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(input_tensor)[0]
    
    # 예측 값을 올림 처리 후 반환
    prediction = np.ceil(prediction.cpu().numpy()).astype(int)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
