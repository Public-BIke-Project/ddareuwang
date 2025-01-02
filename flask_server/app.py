import os
from google.cloud import bigquery
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import pandas as pd
from datetime import datetime, timedelta
import toml
import pytz
import pickle
import torch
import torch.nn as nn
from math import ceil
import numpy as np
import csv

# secrets.toml 파일 읽기
secrets = toml.load("./secrets/secrets.toml")

#Flask
app = Flask(__name__)
app.secret_key = secrets['app']['flask_password'] # Flask의 session 사용

# BigQuery 연결 설정
GOOGLE_CREDENTIALS_PATH = secrets['bigquery']['credentials_file']
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_CREDENTIALS_PATH
client = bigquery.Client()

#-- LSTM START -----------------------------------------------------------------------------------------------------------------#
# 모델 클래스 정의
class BidirectionalModel(nn.Module):
    def __init__(self):
        super(BidirectionalModel, self).__init__()
        self.lstm = nn.LSTM(input_size=165, hidden_size=256, num_layers=3, batch_first=True, bidirectional=True)

        self.multioutput_reg = nn.Sequential(
            nn.Linear(in_features=256 * 2, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=161),
        )

    def forward(self, x, hidden=None):
        output, (h_n, c_n) = self.lstm(x, hidden)
        final_output = output[:, -1, :]  # 모든 타임스텝의 출력에서 마지막 타임스텝만 선택
        output = self.multioutput_reg(final_output)
        return output

# 모델 로드 클래스
class LSTM_Bidirectional:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self):
        model = BidirectionalModel()  # 모델 구조 정의
        # CPU에서 모델 로드
        model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        model.eval()  # 평가 모드
        return model

    @staticmethod
    def get_time_series_data(project_id, dataset_id, table_id, before168_datetime, target_datetime):
        before168_datetime_str = before168_datetime.strftime('%Y-%m-%d %H:%M:%S')
        target_datetime_str = target_datetime.strftime('%Y-%m-%d %H:%M:%S')

        print(f"get_time_변환 : {before168_datetime_str},{target_datetime_str}")

        query = f"""    
            SELECT 
                * EXCEPT(time)
            FROM 
                `{project_id}.{dataset_id}.{table_id}`
            WHERE 
                time BETWEEN  '{before168_datetime_str}' AND '{target_datetime_str}'
            ORDER BY 
                year,month,day,hour
        """

        query_job = client.query(query)
        result = query_job.result()
        return result.to_dataframe()

    def predict(self, project_id, dataset_id, table_id, before168_datetime, target_datetime, device):
        # 1. BigQuery에서 데이터 가져오기
        df_168hours = self.get_time_series_data(project_id, dataset_id, table_id, before168_datetime, target_datetime)
        # 데이터 타입 변환
        df_168hours = df_168hours.astype({
            col: 'float32' if df_168hours[col].dtype == 'float64' else 'int32'
            for col in df_168hours.columns
        })
        
        # 3. 모델 입력 준비
        input_data = torch.tensor(df_168hours.values, dtype=torch.float32).unsqueeze(0).to(device)

        # 4. 모델 예측
        with torch.no_grad():
            prediction = self.model(input_data)

        # 5. 결과 반환
        return prediction.cpu().numpy()
#-- LSTM END -----------------------------------------------------------------------------------------------------------------#

# 사용자 날짜 및 시간 입력
def user_input_datetime():
    month = int(request.args.get('month'))
    day = int(request.args.get('day'))
    hour = int(request.args.get('hour'))
    return month, day, hour

# zone별 대여소ID 불러오기
def load_zone_id(zone):
    zone_id_list = []
    with open ('./area1_station_id_list.txt', 'r') as fr:
    # with open (f'./data/{zone}_station_id_list.txt', 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            zone_id_list.append(line.strip())
    return zone_id_list

# 위도 경도 데이터
def load_LatLonName():
    station_LatLonName_dict = {}
    with open('./data/station_latlon.csv', 'r', encoding='utf-8-sig') as fr:
        reader = csv.DictReader(fr)
        for row in reader:
            station_LatLonName_dict[row['Station_ID']] = {
                "Latitude": row['Latitude'],
                "Longitude": row['Longitude'],
                "Station_name": row['Station_name']
            }
        print("\nstation_LatLonName_dict: ", station_LatLonName_dict)
        return station_LatLonName_dict # ★여기 한글 깨지는거 수정해야 함.★

#-- LGBM START -----------------------------------------------------------------------------------------------------------------#
class LGBMRegressor:
    # LGBM모델에 사용되는 input dataframe과 주변시설 정보 불러오기
    @staticmethod
    def load_LGBMfacility():
        LGBM_facility_list = []
        with open ('./data/station_facilities.csv', 'r') as fr:
            reader = csv.DictReader(fr)
            for row in reader:
                LGBM_facility_list.append(row)
        return LGBM_facility_list
        # [('ST-1171', 6, 2, 0, 2, 1, None, None, None), ...] 한 줄은 set, 전체는 list
        # 31개의 대여소에 대해서만 불러온 데이터

    #LGBM모델 예측에 필요한 시간 함수 (1시간 timedelta)
    @staticmethod
    def get_LGBMtime():
        kst = pytz.timezone('Asia/Seoul')
        now_kst = datetime.now(kst)
        kst_1h_timedelta = now_kst + timedelta(hours=1)
        year = kst_1h_timedelta.year
        month, day, hour = user_input_datetime()
        date = datetime(year, month, day, hour) + timedelta(hours=1)
        if date.weekday() < 5:
            weekday = 1
        else:
            weekday = 0
        return month, hour, weekday
  
    @staticmethod
    def merge_LGBM_facility_time():
        facility = LGBMRegressor.load_LGBMfacility()
        columns = ['Rental_Location_ID', 'bus_stop', 'park', 'school', 'subway', 'riverside']
        input_df = pd.DataFrame(facility, columns=columns)
        input_df['Rental_Location_ID'] = input_df['Rental_Location_ID'].astype('category')

        # 사용자 시간 입력 받아오기
        month, hour, weekday = LGBMRegressor.get_LGBMtime()
        input_df['month'] = month
        input_df['hour'] = hour
        input_df['weekday'] = weekday

        # 숫자 데이터 타입으로 변환
        numeric_columns = ['bus_stop', 'park', 'school', 'subway', 'riverside']
        input_df[numeric_columns] = input_df[numeric_columns].apply(pd.to_numeric, errors='coerce').fillna(0)
        
        return input_df
        # 'Rental_Location_ID', 'bus_stop', 'park', 'school', 'subway', 'riverside', 'month', 'hour', 'weekday'
   
    @staticmethod
    # 모델 불러오기
    def load_LGBMmodel():
        with open ('./model/241121_model_ver2.pkl', 'rb') as file:
             LGBM_model = pickle.load(file)
             print(f"Loaded LGBM model: {LGBM_model}")
        return LGBM_model

    @staticmethod
    # 모델 사용해서 대여소별 수요 예측
    def LGBMpredict():
        LGBM_model = LGBMRegressor.load_LGBMmodel()
        model = LGBM_model
        input_df = LGBMRegressor.merge_LGBM_facility_time()
        predictions = model.predict(input_df)
        print("\npredictions: ", predictions)
        return predictions # type : np.ndarray / 소수점 형태

    @staticmethod 
    def load_LGBMstock(zone):
        zone_id_list = load_zone_id(zone)
        zone_id_tuple = tuple(zone_id_list)
        
        # 해당 시간만
        month, day, hour = user_input_datetime()
        input_date = datetime(2023, month, day)
        input_date = str(input_date.strftime('%Y-%m-%d'))
        input_time = int(hour)
        if input_date == None:
            print("input_date == None")
        if input_time == None:
            print("input_time == None")
        else:
            print("\ninput_date: ", input_date)
            print("\ninput_time: ", input_time)
        LGBM_stock_list = []
        query = f"""
        SELECT * 
        FROM `multi-final-project.Final_table_NURI.2023_available_stocks_fin` 
        WHERE Date = '{input_date}'
            AND Time = {input_time} 
            AND Rental_location_ID IN {zone_id_tuple}
        """
        query_job = client.query(query)
        results = query_job.result()
        for row in results:
            LGBM_stock_list.append(dict(row))
        return LGBM_stock_list

#-- LGBM END -----------------------------------------------------------------------------------------------------------------#

def merge_LGBMresult():
    # 1. input data
    input_df = LGBMRegressor.merge_LGBM_facility_time()
    # 2. prediction
    predictions = LGBMRegressor.LGBMpredict()
    predictions_list = np.ceil(predictions).astype(int).tolist() # 올림하여 predictions을 정수로 만듦
    # 앞에 함수 하나 더 만들어서 LSTM과 앙상블
    # 3. stock
    LGBM_stock_list = LGBMRegressor.load_LGBMstock()
                #         [{
                        #     "Date": "Wed, 01 Mar 2023 00:00:00 GMT",
                        #     "Name_of_the_rental_location": "ㅇㄹㅇㄹ",
                        #     "Rental_location_ID": "ST-1577",
                        #     "Time": 12,
                        #     "stock": 0.0
                #       },...]

    # 4-1. stock_dict 생성
    LGBM_dict = {}
    for item in LGBM_stock_list:
        rental_location_id = item["Rental_location_ID"]
        LGBM_dict[rental_location_id] = item

    # 4-2. merge all data
    merged_result = {}    
    for i in range(len(input_df)):  # 관리권역 1, 2 모두 들어있음
        input_row = input_df.iloc[i]
        predicted_value = predictions_list[i]
        stationid = input_row['Rental_Location_ID']

        station_item = LGBM_dict.get(stationid, None)
        if station_item:
            merged_result[stationid] = {
                "predicted_rental": predicted_value,
                "stock": station_item["stock"]
            }
        # else:
        #     print(f"{stationid}: in the other zone")
            
    return merged_result

def find_station_status():
    merged_result = merge_LGBMresult()  # dict 형태 {"ST-1561": {"predicted_rental": 0, "stock": 2.0}, ...}
    for stationid, item in merged_result.items():
        stock = item["stock"]
        predicted_rental = item['predicted_rental']
        status = stock - (predicted_rental + 3) # 예측된 수요량보다 3개 더 많아야 함
        if status <= 0:
            merged_result[stationid]["status"] = "deficient"
        else:
            merged_result[stationid]["status"] = "abundant"
    station_status_dict = merged_result
    return station_status_dict # dict 형태

def load_zone_distance(zone):
    zone_distance = []
    with open (f'./data/{zone}_distance.csv', 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            zone_distance.append(line.strip())
            # 리스트형태의 튜플 (각 row는 tuple, 전체는 list)
            # result = [
            #                 (1, 'Alice', 25),
            #                 (2, 'Bob', 30),
            #                 (3, 'Charlie', 35)
            #             ]
    return zone_distance

def make_supply_list():
        station_status_dict = find_station_status()
        # "ST-1561": {
        #     "predicted_rental": 0,
        #     "status": "deficient",
        #     "stock": 2.0
        #   },
        supply_demand = []
        for station_id, station_info in station_status_dict.items():
            if station_info["status"] == "deficient":
                if station_info["stock"] == 0:
                    supply_demand.append(-3)  # stock이 아예 없는 경우 3개 필요하다고 입력
                else:
                    supply_demand.append(-station_info["predicted_rental"] -2)  # 예상 수요 +2 만큼 demand로 설정
            elif station_info["status"] == "abundant": # abundant = 예상 수요보다 3개 이상의 stock을 가진 경우
                supply_demand.append(station_info["stock"]) # abundant한 경우: stock 그대로 넣기
            else:
                print(f"ERROR! {station_id} : no status info")
        if sum(supply_demand) < 0:
            supply_demand.append((-1) * sum(supply_demand))  # supply_demand에 center 추가
            print("def make_supply_list로 supply list에 Center 추가!")
        month, day, hour = user_input_datetime()
        print(f"{hour}시 supply_demand: ", supply_demand)
        return supply_demand

def station_names(zone):
        zone_distance = MakeRoute.load_zone_distance(zone)
        station_names = {}
        for i, row in enumerate(zone_distance):
            station_names[i] = row[0] # row[0] = 대여소 ID
            if i == len(zone_distance) - 1:
                break

        supply_demand = MakeRoute.make_supply_list()
        if sum(supply_demand[:-1]) < 0: # 대여가능수량이 부족해서 Center에서 출발할 때 자전거를 적재해야 하는 경우
            station_names[len(zone_distance)] = "center"
            print("\ndef station_names()로 station_names에 Center 추가!")
            print("station_names[len(zone_distance)]: ", station_names[len(zone_distance)])
        return station_names

def Bike_Redistribution(zone):
    supply_demand = MakeRoute.make_supply_list()
    zone_distance = MakeRoute.load_zone_distance(zone)
    # 데이터 정의
    supply = supply_demand
    num_stations = len(supply)
    cost = zone_distance
    # 디버깅 코드
    if num_stations == len(cost):
        print("\nlen(num_stations)랑 len(cost) 일치!")
    else:
        print("\nERROR: len(num_stations)랑 len(cost) 불일치!")

    # 문제 정의
    problem = pulp.LpProblem("Bike_Redistribution", pulp.LpMinimize)

    # 변수 정의: x_ij는 i에서 j로 이동하는 자전거 수
    x = pulp.LpVariable.dicts("x", ((i, j) for i in range(num_stations) for j in range(num_stations)),
                            lowBound=0, cat="Integer")

    # 목표 함수: 총 이동 비용 최소화
    problem += pulp.lpSum(cost[i][j] * x[i, j] for i in range(num_stations) for j in range(num_stations))

    # 여유 대여소에서 자전거 이동량 제한
    for i in range(num_stations):
        if supply[i] > 0:  # 여유 대여소
            problem += pulp.lpSum(x[i, j] for j in range(num_stations) if i != j) <= supply[i]

    # 부족 대여소의 수요 충족
    for j in range(num_stations):
        if supply[j] < 0:  # 부족 대여소
            problem += pulp.lpSum(x[i, j] for i in range(num_stations) if i != j) >= -supply[j]
                
    # 부족 대여소에서 자전거 이동 금지 조건 추가 (새로 추가되는 조건)
    for i in range(num_stations):
        if supply[i] < 0:  # 부족 대여소
            problem += pulp.lpSum(x[i, j] for j in range(num_stations) if i != j) == 0

    # 재고 부족인 경우에만 center(start_station)에서 출발
    if sum(supply) < 0:
        start_station = len(num_stations) - 1 #center의 인덱스
        problem += pulp.lpSum(x[start_station, j] for j in range(num_stations) if j != start_station) >= 1
        
    # 문제 해결
    Bike_Redistribution = problem.solve()
    solve_status = pulp.LpStatus[problem.status]
    print("\nStatus:", solve_status)

    return Bike_Redistribution, x, solve_status

# 메인 페이지
@app.route('/')
def index():
    return render_template('nuri_amend.html')

@app.route('/zone1')
def zone1_page():
    tmap_api_key = secrets['api_keys']['tmap_api_key']    # 기본값 설정
    zone = None
    month = None
    day = None
    hour = None
    buttons_visible = False    

    if request.args:  # 사용자가 폼을 제출했을 때
        zone = 'zone1'
        month = request.args.get('month') # 'month' 입력 필드의 값
        day = request.args.get('day')    # 'day' 입력 필드의 값
        hour = request.args.get('hour')   # 'hour' 입력 필드의 값
        month, day , hour = user_input_datetime()
        zone_id_list = load_zone_id(zone) # ⭐️

        print(f"사용자 입력값 - month: {month}, day: {day}, hour: {hour}")

        # LGBM 클래스 메서드 호출
        try:
            # 사용자 입력에 따라 데이터 처리
            LGBM_time = LGBMRegressor.get_LGBMtime()  # 시간 정보 가져오기
            print(f"LGBM Time Info: {LGBM_time}")
            
            # 데이터 병합 및 예측
            input_df = LGBMRegressor.merge_LGBM_facility_time()
            print(f"Input DataFrame:\n{input_df.head()}")
            
            predictions = LGBMRegressor.LGBMpredict()
            print(f"LGBM Predictions: {predictions}")
            
            # BigQuery 데이터 가져오기
            print(f"Loading stocks for zone: {zone}")
            stocks = LGBMRegressor.load_LGBMstock(zone)
            print(f"Stocks loaded: {stocks}")

            zone_distances = LGBMRegressor.load_zone_distance(zone)
            print(f"zone_distances: {zone_distances}")

            # 결과값 추가 가공 메서드 호출
            processed_data = find_station_status()  # 상태 계산
            print(f"Processed Station Status:\n{processed_data}")

            # 가공된 결과 확인
            supply_demand = make_supply_list()  # 수요-공급 리스트 생성
            print(f"Supply Demand List:\n{supply_demand}")

            # Zone Names 생성
            station_name_data = station_names(zone)
            print(f"Station Names:\n{station_name_data}")


            
            buttons_visible = True  # 버튼 활성화
        except Exception as e:
            print(f"Error during LGBM processing: {str(e)}")
            predictions = []
            stocks = []
        
        if month and day and hour:
            # 폼이 제출되면 버튼을 보이도록 설정
            buttons_visible = True
            month = str(month).zfill(2)
            day = str(day).zfill(2)
            hour = str(hour).zfill(2)

    # GET 요청 시 HTML 폼 렌더링
    return render_template('zone1.html',buttons_visible = buttons_visible, tmap_api_key = tmap_api_key, month=month, day=day, hour=hour)

@app.route('/zone2')
def zone2_page():
    tmap_api_key = secrets['api_keys']['tmap_api_key']    # 기본값 설정
    zone = None # ⭐️
    month = None
    day = None
    hour = None
    buttons_visible = False
    if request.args:  # 사용자가 폼을 제출했을 때
        zone = 'zone2'
        month = request.args.get('month')  # 'month' 입력 필드의 값
        day = request.args.get('day')      # 'day' 입력 필드의 값
        hour = request.args.get('hour')    # 'hour' 입력 필드의 값
        month, day , hour = user_input_datetime()        
        zone_id_list = load_zone_id(zone) # ⭐️

      
        # 168시간 이전 데이터 계산
        target_datetime = datetime(2024, int(month), int(day), int(hour))  # 예시 연도
        before168_datetime = target_datetime - timedelta(hours=168)
        
        print(f"입력값: {target_datetime}, 168시간 이전: {before168_datetime}")

        # 빅쿼리에서 데이터 조회
        project_id = "multi-final-project"
        dataset_id = "Final_table_NURI"
        table_id = "LSTM_data_for_forecast_cloudsql"
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        time_series_data = LSTM_Bidirectional.get_time_series_data(
            project_id, dataset_id, table_id, before168_datetime, target_datetime
        )
        print(f"불러온 데이터:\n{time_series_data}")
        
        # LSTM 모델 초기화
        lstm_model = LSTM_Bidirectional(model_path='./model/LSTM_Bidirectional_model_1202.pth')
        print("LSTM Bidirectional model loaded.")

        # 예측 수행
        predictions = lstm_model.predict(
        project_id, dataset_id, table_id, before168_datetime, target_datetime, device
        )
        print(predictions)
        session['predictions'] = predictions.tolist()

        if month and day and hour:
            # 폼이 제출되면 버튼을 보이도록 설정
            buttons_visible = True
            month = str(month).zfill(2)
            day = str(day).zfill(2)
            hour = str(hour).zfill(2)

    # GET 요청 시 HTML 폼 렌더링
    return render_template('zone2.html',buttons_visible = buttons_visible, tmap_api_key = tmap_api_key, month=month, day=day, hour=hour)

@app.route('/test', methods=['GET'])
def test():
    print("클라이언트에서 /test 요청 도착") 

    if 'predictions' not in session:
        return jsonify({"error": "No predictions available"}), 400
    return jsonify({"predictions": session['predictions']})

@app.route('/moves', methods=['GET'])
def get_simple_moves():
    print("클라이언트에서 /moves 요청 도착") 

    simple_moves = [
        {
            "visit_index": 1,
            "visit_station_id": "ST-963",
            "visit_station_name": "언주역4번출구",
            "station_visit_count": 1,
            "latitude": 37.501228,
            "longitude": 127.050362,
            "status": "abundant",
            "current_stock": 17,
            "move_bikes": 5
        },
        {
            "visit_index": 2,
            "visit_station_id": "ST-3208",
            "visit_station_name": "강남역2번출구",
            "station_visit_count": 1,
            "latitude": 37.512810,
            "longitude": 127.026367,
            "status": "deficient",
            "current_stock": 0,
            "move_bikes": 8
        },
        {
            "visit_index": 3,
            "visit_station_id": "ST-963",
            "visit_station_name": "언주역4번출구",
            "station_visit_count": 1,
            "latitude": 37.501228,
            "longitude": 127.050362,
            "status": "abundant",
            "current_stock": 12,
            "move_bikes": 4
        },
        {
            "visit_index": 4,
            "visit_station_id": "ST-961",
            "visit_station_name": "강남역2번출구",
            "station_visit_count": 1,
            "latitude": 37.518639,
            "longitude": 127.035400,
            "status": "deficient",
            "current_stock": 0,
            "move_bikes": 8
        },
        {
            "visit_index": 5,
            "visit_station_id": "ST-784",
            "visit_station_name": "강남역2번출구",
            "station_visit_count": 1,
            "latitude": 37.515888,
            "longitude": 127.066200,
            "status": "deficient",
            "current_stock": 0,
            "move_bikes": 8
        },
        {
            "visit_index": 6,
            "visit_station_id": "ST-786",
            "visit_station_name": "강남역2번출구",
            "station_visit_count": 1,
            "latitude": 37.517773,
            "longitude": 127.043022,
            "status": "deficient",
            "current_stock": 0,
            "move_bikes": 8
        },
        {
            "visit_index": 7,
            "visit_station_id": "ST-1366",
            "visit_station_name": "강남역2번출구",
            "station_visit_count": 1,
            "latitude": 37.509586,
            "longitude": 127.040909,
            "status": "deficient",
            "current_stock": 0,
            "move_bikes": 8
        },
        {
            "visit_index": 8,
            "visit_station_id": "ST-2882",
            "visit_station_name": "강남역2번출구",
            "station_visit_count": 1,
            "latitude": 37.509785,
            "longitude": 127.042770,
            "status": "deficient",
            "current_stock": 0,
            "move_bikes": 8
        },
        {
            "visit_index": 9,
            "visit_station_id": "ST-1246",
            "visit_station_name": "강남역2번출구",
            "station_visit_count": 1,
            "latitude": 37.506367,
            "longitude": 127.034523,
            "status": "deficient",
            "current_stock": 0,
            "move_bikes": 8
        },
        {
            "visit_index": 10,
            "visit_station_id": "ST-3108",
            "visit_station_name": "강남역2번출구",
            "station_visit_count": 1,
            "latitude": 37.505703,
            "longitude": 127.029198,
            "status": "deficient",
            "current_stock": 0,
            "move_bikes": 8
        }

    ]
    return jsonify(simple_moves)

if __name__ == "__main__":
    app.run(debug=True)





