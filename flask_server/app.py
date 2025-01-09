import os
from google.cloud import bigquery
from flask import Flask, render_template, request, session, jsonify
import pandas as pd
from datetime import datetime, timedelta
import toml
import pytz
import pickle
import torch
import torch.nn as nn
import numpy as np
import csv
import pulp

# secrets.toml 파일 읽기
secrets = toml.load("./secrets/secrets.toml")

#Flask
app = Flask(__name__)
app.secret_key = secrets['app']['flask_password'] # Flask의 session 사용

# BigQuery 연결 설정
GOOGLE_CREDENTIALS_PATH = secrets['bigquery']['credentials_file']
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_CREDENTIALS_PATH
client = bigquery.Client()

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
            # LGBM 데이터 병합 및 예측
            LGBM_time = LGBMRegressor.get_LGBMtime()  # 시간 정보 가져오기           
            input_df = LGBMRegressor.merge_LGBM_facility_time()    
            LGBM_pred = LGBMRegressor.LGBMpredict()  

            # LSTM 168시간 이전 데이터 계산
            target_DT = datetime(2024, int(month), int(day), int(hour))  # 예시 연도
            before168_DT = target_DT - timedelta(hours=168)
            
            # 빅쿼리에서 데이터 조회
            project_id = "multi-final-project"
            dataset_id = "Final_table_NURI"
            table_id = "LSTM_data_for_forecast_cloudsql"
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            time_series_data = LSTM_Bidirectional.get_time_series_data(project_id, dataset_id, table_id, before168_DT, target_DT)
            
            # LSTM 모델 초기화
            lstm_model = LSTM_Bidirectional(model_path='./model/LSTM_Bidirectional_model_1202.pth')
            print("LSTM Bidirectional model loaded.")

            # 예측 수행
            LSTM_pred_fin = lstm_model.predict(project_id, dataset_id, table_id, before168_DT, target_DT, device)

            LGBM_pred_fin = LGBM_pred[np.newaxis, :]

            # BigQuery 데이터 가져오기
            stocks = load_stock(zone)
            merged_result = merge_result(zone,LSTM_pred_fin)

            # session['predictions'] = predictions.tolist()


            # 결과값 추가 가공 메서드 호출
            zone_distances = load_zone_distance(zone)
            processed_data = find_station_status(zone,LSTM_pred_fin)  # 상태 계산
            # 가공된 결과 확인
            supply_demand = make_supply_list(zone,LSTM_pred_fin)  # 수요-공급 리스트 생성
            # Zone Names 생성
            station_name_data = station_names(zone,LSTM_pred_fin)
            #----------
            Bike_Redistribution_result, x, solve_status = Bike_Redistribution(zone, LSTM_pred_fin)
            print(f"Bike Redistribution Result: {Bike_Redistribution_result}")
            print(f"Solve Status: {solve_status}")
            # results_dict = save_result(zone) 나한테 없는 코드
            # simplified_moves = simplify_movements(zone, x, station_name_data)
            # simple_moves = final_route(x, station_name_data)
            # final_simple_moves = get_simple_moves(zone)
            
            buttons_visible = True  # 버튼 활성화
        except Exception as e:
            print(f"Error during LGBM processing: {str(e)}")
            LGBM_pred = []
            stocks = []
        
        if month and day and hour:
            # 폼이 제출되면 버튼을 보이도록 설정
            buttons_visible = True
            month = str(month).zfill(2)
            day = str(day).zfill(2)
            hour = str(hour).zfill(2)

    # GET 요청 시 HTML 폼 렌더링
    return render_template('zone1.html',buttons_visible = buttons_visible, tmap_api_key = tmap_api_key, month=month, day=day, hour=hour)

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
    def get_time_series_data(project_id, dataset_id, table_id, before168_DT, target_DT):
        before168_DT_str = before168_DT.strftime('%Y-%m-%d %H:%M:%S')
        target_DT_str = target_DT.strftime('%Y-%m-%d %H:%M:%S')

        print(f"get_time_변환 : {before168_DT_str},{target_DT_str}")

        query = f"""    
            SELECT 
                * EXCEPT(time)
            FROM 
                `{project_id}.{dataset_id}.{table_id}`
            WHERE 
                time BETWEEN  '{before168_DT_str}' AND '{target_DT_str}'
            ORDER BY 
                year,month,day,hour
        """

        query_job = client.query(query)
        result = query_job.result()
        return result.to_dataframe()

    def predict(self, project_id, dataset_id, table_id, before168_DT, target_DT, device):
        # 1. BigQuery에서 데이터 가져오기
        df_168hours = self.get_time_series_data(project_id, dataset_id, table_id, before168_DT, target_DT)
            # 데이터 타입 변환
        df_168hours = df_168hours.astype({
            col: 'float32' if df_168hours[col].dtype == 'float64' else 'int32'
            for col in df_168hours.columns
        })
        
        # 2. 모델 입력 준비 : 텐서 값으로 입력
        input_data = torch.tensor(df_168hours.values, dtype=torch.float32).unsqueeze(0).to(device)

        # 3. 모델 예측
        with torch.no_grad():
            LSTM_pred = self.model(input_data) # tensor 형태

        # 4. 결과 반환
        LSTM_pred_fin = LSTM_pred.cpu().numpy()
        return LSTM_pred_fin

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
    with open (f'./data/{zone}_station_id_list.txt', 'r') as fr:
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
        return station_LatLonName_dict # ★여기 Flask에서 한글 깨지는거 수정해야 함.★

#-- LGBM START -----------------------------------------------------------------------------------------------------------------#
class LGBMRegressor:
    # LGBM모델에 사용되는 input dataframe과 주변시설 정보 불러오기
    @staticmethod
    def load_LGBMfacility():
        LGBM_facility_list = []
        with open ('./data/station_facilities.csv', 'r') as fr:
            reader = csv.reader(fr)
            next(reader)
            for row in reader:
                LGBM_facility_list.append(tuple(row))
        # print("\nLGBM_facility_list: ", LGBM_facility_list)
        return LGBM_facility_list
        # {'Rental_Location_ID': 'ST-2675', 'bus_stop': '11', 'park': '5', 'school': '1', 'subway': '0', 'riverside': '0', 'month': '', 'hour': '', 'weekday': ''}
        # [('ST-1171', 6, 2, 0, 2, 1, None, None, None), ...] 한 줄은 tuple, 전체는 list

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
        # facility data 불러오기 
        facility = LGBMRegressor.load_LGBMfacility()
        columns = ['Rental_Location_ID', 'bus_stop', 'park', 'school', 'subway', 'riverside']
        input_df = pd.DataFrame(facility, columns=columns)

        # 사용자 시간 입력 받아오기
        month, hour, weekday = LGBMRegressor.get_LGBMtime()
        input_df['month'] = month
        input_df['hour'] = hour
        input_df['weekday'] = weekday

        input_columns = ['Rental_Location_ID', 'bus_stop', 'park', 'school', 'subway', 'riverside', 'month', 'hour', 'weekday']
        numeric_columns = ['bus_stop', 'park', 'school', 'subway', 'month', 'hour'] 
        categorical_columns = ['Rental_Location_ID', 'riverside', 'weekday']
        for col in input_columns:
            if col in categorical_columns: # 카테고리형 컬럼
                input_df[col] = input_df[col].astype('category')
            elif col in numeric_columns: # 정수형 컬럼
                input_df[col] = input_df[col].astype('int')
            else:
                print(f"{col} is not categorical nor numeric")

        # 디버깅 출력
        print("\nDataFrame after processing:")
        print(input_df.dtypes)
        print(input_df.head())
        
        return input_df
        # 'Rental_Location_ID', 'bus_stop', 'park', 'school', 'subway', 'riverside', 'month', 'hour', 'weekday'
   
    @staticmethod
    # 모델 불러오기
    def load_LGBMmodel():
        with open ('./model/250109_NEW_LGBMmodel.pkl', 'rb') as file:
             LGBM_model = pickle.load(file)
        return LGBM_model

    @staticmethod
    # 모델 사용해서 대여소별 수요 예측
    def LGBMpredict():
        LGBM_model = LGBMRegressor.load_LGBMmodel()
        model = LGBM_model
        input_df = LGBMRegressor.merge_LGBM_facility_time()
        LGBM_pred = model.predict(input_df)
        return LGBM_pred # type : np.ndarray / 소수점 형태


#-- LGBM END -----------------------------------------------------------------------------------------------------------------#

@staticmethod 
def load_stock(zone):
    zone_id_list = load_zone_id(zone)
    zone_id_tuple = tuple(zone_id_list)
    
    # user input 시간만 stock 불러옴
    month, day, hour = user_input_datetime()
    input_date = datetime(2023, month, day)
    input_date = str(input_date.strftime('%Y-%m-%d'))
    input_time = int(hour)

    stock_list = []
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
        stock_list.append(dict(row))
    return stock_list
    # stock_list = [{'Date': datetime.date(2023, 4, 2), 
    #               'Time': 6, 'stock': 3.0, 'Rental_location_ID': 'ST-3164', 
    #               'Name_of_the_rental_location': '청담역 1번출구'}, ... ]

def merge_result(zone, LSTM_pred_fin):
    """
    1) LGBM 예측
    2) 앙상블: (LGBM_pred + LSTM_pred) / 2
    3) 재고(stock)와 병합
    4) 대여소별로 최종 딕셔너리 반환
    """
    # 1. LGBM input_data & 예측
    LGBM_pred = LGBMRegressor.LGBMpredict()             # shape (N,) : 1차원 배열 ndarray [1 2 3 4]
    LGBM_pred_fin = LGBM_pred[np.newaxis, :]            # shape (1, N) : 2차원 배열 [[1 2 3 4]]

    # 2. 앙상블
    ensemble_array = np.rint((LGBM_pred_fin + LSTM_pred_fin) / 2).astype(int)  # shape (1, N)    # rint : 가장 가까운 정수로 반올림
    ensemble_list = ensemble_array[0].tolist()
    # print("\nensemble_list: ", ensemble_list, "type(ensemble_list): ", type(ensemble_list))

    # (선택) LGBM 예측을 올림하기 원한다면, np.ceil 또는 np.round 후 list 변환
    # LGBM_pred_list = np.ceil(LGBM_pred_fin)[0].astype(int).tolist()

    # 3. stock 병합
    stock_list = load_stock(zone)
        #         [{
            #     "Date": "Wed, 01 Mar 2023 00:00:00 GMT",
            #     "Name_of_the_rental_location": "ㅇㄹㅇㄹ",
            #     "Rental_location_ID": "ST-1577",
            #     "Time": 12,
            #     "stock": 1 (int)
    #       },...]

    selected_zone_dict = {}
    for item in stock_list:
        rental_location_id = item["Rental_location_ID"]
        selected_zone_dict[rental_location_id] = item  # { 'ST-784': {...}, ...}

    # 4. 대여소별 결과 dict 작성
    input_df = LGBMRegressor.merge_LGBM_facility_time()
    merged_result = {}
    for i in range(len(input_df)):  # N개의 대여소
        stationid = input_df.iloc[i]['Rental_Location_ID']
        ens_val  = ensemble_list[i] # Ensemble 예측
        station_item = selected_zone_dict.get(stationid, None)
        if station_item:
            merged_result[stationid] = {
                "predicted_rental": ens_val,
                "stock": station_item["stock"]
            }
        # else:
        #     print(f"{stationid}: not in {zone} (다른 관리권역에 있거나 31개에 포함되지 않음)")

    return merged_result

def find_station_status(zone,LSTM_pred_fin):
    merged_result = merge_result(zone,LSTM_pred_fin)  # dict 형태 {"ST-1561": {"predicted_rental": 0, "stock": 2.0}, ...}
    for stationid, item in merged_result.items():
        stock = item["stock"]
        predicted_rental = item['predicted_rental']
        status = stock - (predicted_rental + 3) # 예측된 수요량보다 3개 더 많아야 함
        if status < 0:
            merged_result[stationid]["status"] = "deficient"
        else:
            merged_result[stationid]["status"] = "abundant"
    station_status_dict = merged_result
    return station_status_dict # dict 형태

def load_zone_distance(zone):
    zone_distance = []
    with open(f'./data/{zone}_distance.csv', 'r') as fr:
        lines = fr.readlines()
        for line in lines[1:]:  # header 건너뛰기
            values = str(line).split(",")            # ['ST-786', '0', '2.83', '1.78', '2.18']
            distance_values = values[1:]             # 맨 앞에 ST-..는 건너뛰기 -> ['0', '2.83', '1.78', '2.18']
            row = list(map(float, distance_values))  # [0.0, 2.83, 1.78, 2.18]
            zone_distance.append(row)
    return zone_distance

def make_supply_list(zone,LSTM_pred_fin):
        station_status_dict = find_station_status(zone,LSTM_pred_fin)
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
                    supply_demand.append(int(-station_info["predicted_rental"]) -2)  # 예상 수요 +2 만큼 demand로 설정
            elif station_info["status"] == "abundant": # abundant = 예상 수요보다 3개 이상의 stock을 가진 경우
                supply_demand.append(int(station_info["stock"])) # abundant한 경우: stock 그대로 넣기
            else:
                print(f"ERROR! {station_id} : no status info")
        if sum(supply_demand) < 0:
            supply_demand.append(int((-1) * sum(supply_demand)))  # supply_demand에 center 추가
            print("def make_supply_list로 supply list에 Center 적재량 추가!")
        return supply_demand

def station_names(zone, LSTM_pred_fin):
    station_names_data = {}
    with open(f'./data/{zone}_distance.csv', 'r') as fr: 
        first_line = fr.readline()
        first_line_list = first_line.strip().split(',')
        for i in range(1, len(first_line_list)):
            station_name = first_line_list[i]
            station_names_data[i] = station_name  # -> 일단 center까지 모두 추가

    # Center 처리 조건 (station_names_data)
    supply_demand = make_supply_list(zone, LSTM_pred_fin)
    if sum(supply_demand[:-1]) > 0:  # 공급 부족이 아닌 경우 center 제거
        last_key = max(station_names_data.keys())
        removed_value = station_names_data.pop(last_key)  # 마지막 항목(center) 제거
        print(f"\n[INFO] Station '{removed_value}'가 station_names_data에서 제거되었습니다!")
    else:  # 공급 부족인 경우 center 유지
        print(f"\n[INFO] supply_demand가 부족하여 station_names_data에서 Center 정보 유지.")

    # # 디버깅: center 처리 후 station_names 출력
    # for index, station_name in station_names_data.items():
    #     print(f"center 처리 후 - Index: {index}, Station Name: {station_name}")
    return station_names_data

def Bike_Redistribution(zone, LSTM_pred_fin):
    supply_demand = make_supply_list(zone, LSTM_pred_fin) # list 형태
    zone_distance = load_zone_distance(zone) # csv 파일을 (각 row는 tuple, 전체는 list) 형태로 변환
    
    # Center 처리 조건 (zone_distance)
    if sum(supply_demand[:-1]) > 0:  # 공급 부족이 아닌 경우 zone_distance에서 center 제거
        zone_distance.pop()  # 마지막 항목(center) 제거
        print(f"\n[INFO] Center가 zone_distance에서 제거되었습니다!")
    else:  # 공급 부족인 경우 center 유지
        print(f"\n[INFO] 공급이 부족하여 zone_distance에서 Center 정보 유지.")

    # 데이터 정의
    supply = supply_demand # center가 처리된 상태
    num_stations = len(supply)
    cost = zone_distance

    # 디버깅 코드
    if num_stations == len(cost):
        print("\nnum_stations랑 len(cost) 일치!")
    else:
        print("\nERROR: len(num_stations)랑 len(cost) 불일치!")

    # 문제 정의
    problem = pulp.LpProblem("Bike_Redistribution", pulp.LpMinimize)

    # 변수 정의: x_ij는 i에서 j로 이동하는 자전거 수
    x = pulp.LpVariable.dicts("x", ((i, j) for i in range(num_stations) for j in range(num_stations)), lowBound=0, cat="Integer")

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
        start_station = num_stations - 1 # center의 인덱스
        problem += pulp.lpSum(x[start_station, j] for j in range(num_stations) if j != start_station) >= 1
        
    # 문제 해결
    Bike_Redistribution_result = problem.solve()
    solve_status = pulp.LpStatus[problem.status]
    print("\nStatus:", solve_status)

    return Bike_Redistribution_result, x, solve_status

@staticmethod
def save_result(zone, LSTM_pred_fin):
    supply_demand = make_supply_list(zone,LSTM_pred_fin)
    station_names_data = station_names(zone)
    num_stations = len(supply_demand)
    problem, x, solve_status = Bike_Redistribution(zone, LSTM_pred_fin)

    station_status_dict = find_station_status(zone)
    results_dict = {"status": solve_status, "moves": []}

    for i in range(num_stations):
        for j in range(num_stations):
            if x[i, j].varValue is not None and x[i, j].varValue > 0:
                from_name = station_names_data[i]
                to_name = station_names_data[j]
                cur_station_dict = station_status_dict.get(to_name)
                if not cur_station_dict: # 디버깅 코드
                    print("\n", to_name, "no visit!")
                else:
                    stock = cur_station_dict["stock"]
                    results_dict["moves"].append({
                        "from_station": from_name,
                        "from_index": i,
                        "to_station": to_name,
                        "to_index": j,
                        "bikes_moved": x[i, j].varValue,
                        "stock": stock  # 가져온 stock 값 
                    })
                print(f"후처리 전- From {from_name}({i}) to {to_name}({j}), move bikes: {x[i, j].varValue}")
    print("\nresults_dict: ", results_dict)
    return results_dict

# #후처리 함수
# @staticmethod
# def simplify_movements(zone, x, station_names):
#     supply_demand = make_supply_list(zone)
#     zone_distance = load_zone_distance(zone)
#     station_names = station_names(zone)
#     problem, x, solve_status = Bike_Redistribution(supply_demand, zone_distance, station_names)

#     simplified_moves = {}
#     simplified_flag = False  # 간소화 여부를 확인하기 위한 플래그
#     for (i, j), var in x.items():
#         if var.varValue is not None and var.varValue > 0:
#             # 현재 이동량 추가
#             move_amount = var.varValue
#             if (j, i) in simplified_moves: #이미 저장된 반대 방향 이동 (j, i)가 존재하는지 확인
#                 reverse_amount = simplified_moves.pop((j, i)) #반대 방향 이동량 가져오기 및 삭제
#                 net_amount = move_amount - reverse_amount #상쇄 결과 계산(간소화된 move결과)
                
#                 #간소화된 move를 simplified_moves에 추가(양수인지 음수인지 고려해서)
#                 if net_amount > 0: 
#                     simplified_moves[(i, j)] = net_amount
#                 elif net_amount < 0:
#                     simplified_moves[(j, i)] = -net_amount

#                 simplified_flag = True
#             else:
#                 simplified_moves[(i, j)] = move_amount
    
#     # 결과 출력
#     if simplified_flag:
#         for (i, j), amount in simplified_moves.items():
#             from_name = station_names[i]
#             to_name = station_names[j]
#             print(f" 후처리 후- Move {amount} bikes from {from_name}({i}) to {to_name}({j})")
#         print("후처리 진행됨!")
#     else:
#         print("후처리 불필요")
#     return simplified_moves

# # 인덱스 추가 및 파라미터 정리 함수 #POST 대상
# def final_route(x, station_names):
#     # 1. 후처리된 결과
#     simplified_moves = simplify_movements(x, station_names)
#     # simplified_moves 예시 : {(1, 0): 5.0, (1, 2): 4.0, (1, 6): 3.0, (4, 12): 5.0, (5, 3): 3.0, (10, 15): 6.0, (13, 14): 5.0}

#     # 2. 대여소 상태 dict
#     results_dict = save_result()
#     stock_and_status = results_dict["moves"]
#         # results_dict["moves"].append({
#         #     "from_station": from_name,
#         #     "from_index": i,
#         #     "to_station": to_name,
#         #     "to_index": j,
#         #     "bikes_moved": x[i, j].varValue,
#         # })

#     # 3. station_latlon
#     station_LatLonName_dict = load_LatLonName()

#     # 4. 경로 결과값 출력
#     previous_from_station = None
#     simple_moves = []
#     for i, (from_station, to_station), move in enumerate(simplified_moves.items()):
#         # visit station name
#         key = (from_station, to_station)
#         to_station_id= key[1]
#         visit_station_name = station_names[to_station_id]

#         # station_visit_count
#         station_visit_count_list = [key[1] for key in simplified_moves.keys()]

#         # lat lon
#         to_station_lat = station_LatLonName_dict[to_station_id]["latitude"]
#         to_station_lon = station_LatLonName_dict[to_station_id]["longitude"]

#         visit_count_dict = {}
#         if previous_from_station != from_station:
#             visit_count_dict[to_station_id] = visit_count_dict.get(to_station_id, 0) + 1
#             simple_moves.append({
#                 "visit_index": i,
#                 "visit_station_id": to_station,
#                 "visit_station_name": visit_station_name,
#                 "station_visit_count": visit_count_dict[to_station_id],
#                 "latitude": to_station_lat, 
#                 "longitude": to_station_lon,
#                 "status": stock_and_status["status"],
#                 "current_stock": stock_and_status["stock"],
#                 "move_bikes": move
#         })        
#         previous_from_station = from_station
#     print(simple_moves)
#     return jsonify(simple_moves)

# @app.route('/moves', methods=['GET'])
# def get_simple_moves(zone):
#     supply_demand = make_supply_list(zone)
#     zone_distance = load_zone_distance(zone)
#     station_names = station_names(zone)
#     problem, x, solve_status = Bike_Redistribution(supply_demand, zone_distance, station_names)
#     simple_moves = final_route(x, station_names)
#     final_simple_moves = jsonify(simple_moves)
#     return final_simple_moves





# ----- FLASK ------------------------------------------------------------------------------------------------------------------------#


@app.route('/zone2')
def zone2_page():
    tmap_api_key = secrets['api_keys']['tmap_api_key']    # 기본값 설정
    zone = None # ⭐️
    month = None
    day = None
    hour = None
    buttons_visible = False

    # GET 요청 시 HTML 폼 렌더링
    return render_template('zone2.html',buttons_visible = buttons_visible, tmap_api_key = tmap_api_key, month=month, day=day, hour=hour)

@app.route('/test', methods=['GET'])
def test():
    print("클라이언트에서 /test 요청 도착") 

    if 'predictions' not in session:
        return jsonify({"error": "No predictions available"}), 400
    return jsonify({"predictions": session['predictions']})



if __name__ == "__main__":
    app.run(debug=True)