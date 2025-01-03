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
from math import ceil
import numpy as np
import csv
import pulp
from collections import Counter

# secrets.toml 파일 읽기
secrets = toml.load("./secrets/secrets.toml")

#Flask
app = Flask(__name__)
app.secret_key = secrets['app']['flask_password'] # Flask의 session 사용

# BigQuery 연결 설정
GOOGLE_CREDENTIALS_PATH = secrets['bigquery']['credentials_file']
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_CREDENTIALS_PATH
client = bigquery.Client()


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

    if request.args:
        zone = 'zone1'
        month = request.args.get('month')
        day = request.args.get('day')   
        hour = request.args.get('hour')  
        month, day , hour = user_input_datetime()
        zone_id_list = load_zone_id(zone) # ⭐️

        print(f"사용자 입력값 - month: {month}, day: {day}, hour: {hour}")

        # zone1_page 렌더링
        try:
            LGBM_time = LGBMRegressor.get_LGBMtime()  # 시간 정보 가져오기
            input_df = LGBMRegressor.merge_LGBM_facility_time()
            predictions = LGBMRegressor.LGBMpredict()
            stocks = LGBMRegressor.load_LGBMstock(zone)
            merged_result = merge_result(zone)
            zone_distances = load_zone_distance(zone)
            processed_data = find_station_status(zone)
            supply_demand = make_supply_list(zone)
            station_name_data = station_names(zone)
            Bike_Redistribution, x, solve_status = Bike_Redistribution(zone)
            results_dict = save_result(zone)
            simplified_moves = simplify_movements(zone, x, station_name_data)
            simple_moves = final_route(x, station_name_data)
            final_simple_moves = get_simple_moves(zone)

            # 버튼 활성화
            buttons_visible = True  
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
    zone = None
    month = None
    day = None
    hour = None
    buttons_visible = False    

    if request.args:
        zone = 'zone2'
        month = request.args.get('month')
        day = request.args.get('day')   
        hour = request.args.get('hour')  
        month, day , hour = user_input_datetime()
        zone_id_list = load_zone_id(zone) # ⭐️

        # zone2_page 렌더링
        try:
            LGBM_time = LGBMRegressor.get_LGBMtime()  # 시간 정보 가져오기
            input_df = LGBMRegressor.merge_LGBM_facility_time()
            predictions = LGBMRegressor.LGBMpredict()
            stocks = LGBMRegressor.load_LGBMstock(zone)
            merged_result = merge_result(zone)
            zone_distances = load_zone_distance(zone)
            processed_data = find_station_status(zone)
            supply_demand = make_supply_list(zone)
            station_name_data = station_names(zone)
            Bike_Redistribution, x, solve_status = Bike_Redistribution(zone)
            results_dict = save_result(zone)
            simplified_moves = simplify_movements(zone, x, station_name_data)
            simple_moves = final_route(x, station_name_data)
            final_simple_moves = get_simple_moves(zone)

            # 버튼 활성화
            buttons_visible = True  
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
    return render_template('zone2.html',buttons_visible = buttons_visible, tmap_api_key = tmap_api_key, month=month, day=day, hour=hour)



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
            predictions = self.model(input_data)

        # 5. 결과 반환
        return predictions.cpu().numpy()


def filter_target_stations(predictions: np.ndarray, all_stations: list[str], target_stations: list[str]) -> np.ndarray:
    for st_id in target_stations:
        if st_id not in all_stations:
            raise ValueError(f"[filter_target_stations] 대여소 ID '{st_id}'가 all_stations 리스트에 없습니다.")

    # 1) station_id -> 인덱스 매핑
    target_indices = [all_stations.index(st) for st in target_stations]

    # 2) 예측에서 해당 인덱스만 추출
    LSTM_predictions = predictions[:, target_indices]  # shape: (1, len(target_indices))

    # print(f"Filtered prediction shape: {filtered_pred.shape}")
    return LSTM_predictions
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
        return LGBM_model

    @staticmethod
    # 모델 사용해서 대여소별 수요 예측
    def LGBMpredict():
        LGBM_model = LGBMRegressor.load_LGBMmodel()
        model = LGBM_model
        input_df = LGBMRegressor.merge_LGBM_facility_time()
        LGBM_predictions = model.predict(input_df)
        return LGBM_predictions # type : np.ndarray / 소수점 형태
    
#-- LGBM END -----------------------------------------------------------------------------------------------------------------#
    
def ensemble(pred, all_stations, target_stations):
    LSTM_predictions = filter_target_stations(pred, all_stations, target_stations)
    LGBM_predictions = LGBMRegressor.LGBMpredict()

    ensembled_result = (LSTM_predictions + LGBM_predictions) / 2
    print("\nensembled_result: ", ensembled_result)
    return ensembled_result

@staticmethod 
def load_stock(zone):
    # 해당 zone id list 불러오기
    zone_id_list = load_zone_id(zone)
    zone_id_tuple = tuple(zone_id_list)
    
    # 해당 시간만
    month, day, hour = user_input_datetime()
    input_date = datetime(2023, month, day)
    input_date = str(input_date.strftime('%Y-%m-%d'))
    input_time = int(hour)

    query = f"""
    SELECT * 
    FROM `multi-final-project.Final_table_NURI.2023_available_stocks_fin` 
    WHERE Date = '{input_date}'
        AND Time = {input_time} 
        AND Rental_location_ID IN {zone_id_tuple}
    """
    query_job = client.query(query)
    results = query_job.result()
    
    stock_list = []
    for row in results:
        stock_list.append(dict(row))
    return stock_list

def merge_result(zone, pred, all_stations, target_stations): # stock과 pred를 합치는 함수
    # 1. input data
    input_df = LGBMRegressor.merge_LGBM_facility_time()
    
    # 2. predicted data
    ensembled_result = ensemble(pred, all_stations, target_stations)
    ensembled_predictions = np.ceil(ensembled_result).astype(int).tolist()# 올림하여 ensembled_result을 정수로 만듦 # list 형태

    # 3. stock 
    stock_list = load_stock(zone)
    #         [{
            #     "Date": "Wed, 01 Mar 2023 00:00:00 GMT",
            #     "Name_of_the_rental_location": "ㅇㄹㅇㄹ",
            #     "Rental_location_ID": "ST-1577",
            #     "Time": 12,
            #     "stock": 0.0
    #       },...]

    # 4-1. selected_zone_dict 생성
    selected_zone_dict = {}
    for item in stock_list:
        rental_location_id = item["Rental_location_ID"]
        selected_zone_dict[rental_location_id] = item

    # 4-2. merge all data
    merged_result = {}    
    for i in range(len(input_df)):  # 관리권역 1, 2 모두 들어있음
        input_row = input_df.iloc[i]
        predicted_value = ensembled_predictions[i]
        stationid = input_row['Rental_Location_ID']

        station_item = selected_zone_dict.get(stationid, None)
        if station_item:
            merged_result[stationid] = {
                "predicted_rental": predicted_value,
                "stock": station_item["stock"]
            }
        # else:
        #     print(f"{stationid}: in the other zone")
            
    return merged_result

def find_station_status(zone):
    merged_result = merge_result(zone)  # dict 형태 {"ST-1561": {"predicted_rental": 0, "stock": 2.0}, ...}
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
    return zone_distance # 리스트형태의 튜플 (각 row는 tuple, 전체는 list)

def make_supply_list(zone):
        station_status_dict = find_station_status(zone)
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
        # print(f"{hour}시 supply_demand: ", supply_demand)
        return supply_demand

def station_names(zone):
        zone_distance = load_zone_distance(zone)
        station_names = {}
        for i, row in enumerate(zone_distance):
            station_names[i] = row[0] # row[0] = 대여소 ID
            if i == len(zone_distance) - 1:
                break

        supply_demand = make_supply_list(zone)
        if sum(supply_demand[:-1]) < 0: # 대여가능수량이 부족해서 Center에서 출발할 때 자전거를 적재해야 하는 경우
            station_names[len(zone_distance)] = "center"
            print("\ndef station_names()로 station_names에 Center 추가!")
            print("station_names[len(zone_distance)]: ", station_names[len(zone_distance)])
        print(station_names)
        return station_names

def Bike_Redistribution(zone):
    supply_demand = make_supply_list(zone)
    zone_distance = load_zone_distance(zone)

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

@staticmethod
def save_result(zone):
    supply_demand = make_supply_list(zone)
    zone_distance = load_zone_distance(zone)
    station_names = station_names(zone)
    num_stations = len(supply_demand)
    problem, x, solve_status = Bike_Redistribution(supply_demand, zone_distance, station_names)

    station_status_dict = find_station_status(zone)
    results_dict = {"status": solve_status, "moves": []}

    for i in range(num_stations):
        for j in range(num_stations):
            if x[i, j].varValue is not None and x[i, j].varValue > 0:
                from_name = station_names[i]
                to_name = station_names[j]
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
    return results_dict

#후처리 함수
@staticmethod
def simplify_movements(zone, x, station_names):
    supply_demand = make_supply_list(zone)
    zone_distance = load_zone_distance(zone)
    station_names = station_names(zone)
    problem, x, solve_status = Bike_Redistribution(supply_demand, zone_distance, station_names)

    simplified_moves = {}
    simplified_flag = False  # 간소화 여부를 확인하기 위한 플래그
    for (i, j), var in x.items():
        if var.varValue is not None and var.varValue > 0:
            # 현재 이동량 추가
            move_amount = var.varValue
            if (j, i) in simplified_moves: #이미 저장된 반대 방향 이동 (j, i)가 존재하는지 확인
                reverse_amount = simplified_moves.pop((j, i)) #반대 방향 이동량 가져오기 및 삭제
                net_amount = move_amount - reverse_amount #상쇄 결과 계산(간소화된 move결과)
                
                #간소화된 move를 simplified_moves에 추가(양수인지 음수인지 고려해서)
                if net_amount > 0: 
                    simplified_moves[(i, j)] = net_amount
                elif net_amount < 0:
                    simplified_moves[(j, i)] = -net_amount

                simplified_flag = True
            else:
                simplified_moves[(i, j)] = move_amount
    
    # 결과 출력
    if simplified_flag:
        for (i, j), amount in simplified_moves.items():
            from_name = station_names[i]
            to_name = station_names[j]
            print(f" 후처리 후- Move {amount} bikes from {from_name}({i}) to {to_name}({j})")
        print("후처리 진행됨!")
    else:
        print("후처리 불필요")
    return simplified_moves


# 인덱스 추가 및 파라미터 정리 함수 #POST 대상
def final_route(x, station_names):
    # 1. 후처리된 결과
    simplified_moves = simplify_movements(x, station_names)
    # simplified_moves 예시 : {(1, 0): 5.0, (1, 2): 4.0, (1, 6): 3.0, (4, 12): 5.0, (5, 3): 3.0, (10, 15): 6.0, (13, 14): 5.0}

    # 2. 대여소 상태 dict
    results_dict = save_result()
    stock_and_status = results_dict["moves"]
        # results_dict["moves"].append({
        #     "from_station": from_name,
        #     "from_index": i,
        #     "to_station": to_name,
        #     "to_index": j,
        #     "bikes_moved": x[i, j].varValue,
        # })

    # 3. station_latlon
    station_LatLonName_dict = load_LatLonName()

    # 4. 경로 결과값 출력
    previous_from_station = None
    simple_moves = []
    for i, (from_station, to_station), move in enumerate(simplified_moves.items()):
        # visit station name
        key = (from_station, to_station)
        to_station_id= key[1]
        visit_station_name = station_names[to_station_id]

        # station_visit_count
        station_visit_count_list = [key[1] for key in simplified_moves.keys()]

        # lat lon
        to_station_lat = station_LatLonName_dict[to_station_id]["latitude"]
        to_station_lon = station_LatLonName_dict[to_station_id]["longitude"]

        visit_count_dict = {}
        if previous_from_station != from_station:
            visit_count_dict[to_station_id] = visit_count_dict.get(to_station_id, 0) + 1
            simple_moves.append({
                "visit_index": i,
                "visit_station_id": to_station,
                "visit_station_name": visit_station_name,
                "station_visit_count": visit_count_dict[to_station_id],
                "latitude": to_station_lat, 
                "longitude": to_station_lon,
                "status": stock_and_status["status"],
                "current_stock": stock_and_status["stock"],
                "move_bikes": move
        })        
        previous_from_station = from_station
    print(simple_moves)
    return simple_moves

@app.route('/moves', methods=['GET'])
def get_simple_moves(zone):
    Bike_Redistribution, x, solve_status = Bike_Redistribution(zone)
    simple_moves = final_route(x, station_names)
    final_simple_moves = jsonify(simple_moves)
    return final_simple_moves

if __name__ == "__main__":
    app.run(debug=True)