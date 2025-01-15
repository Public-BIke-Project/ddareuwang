import os
import toml
from google.cloud import bigquery
from flask import Flask, render_template, request, jsonify
from google.cloud import bigquery
import pandas as pd
from datetime import datetime, timedelta
import pytz
import pickle
import torch
import torch.nn as nn
import numpy as np
import csv
import pulp

# TOML 파일 및 상대 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))           # 상대 경로 (깃허브 연결 시 필요)
secrets_path = os.path.join(BASE_DIR, "secrets/secrets.toml")
secrets = toml.load(secrets_path)                               # TOML 파일 읽기

# Flask
app = Flask(__name__)
app.secret_key = secrets['app']['flask_password']               # Flask의 session 사용

# model path
LGBM_model_path = os.path.join(BASE_DIR, "model/250109_NEW_LGBMmodel.pkl")
LSTM_model_path = os.path.join(BASE_DIR, "model/LSTM_Bidirectional_model_1202.pth")

# BigQuery 연결 설정
project_id = secrets['bigquery']['project_id']
GOOGLE_CREDENTIALS_FILE = os.path.join(BASE_DIR, secrets['bigquery']['credentials_file'])
table = secrets['bigquery']['table']

# 환경 변수로 인증 정보 설정
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_CREDENTIALS_FILE
client = bigquery.Client()
client._use_bqstorage_api = False  # BigQuery Storage API 비활성화

# ---------------- ZONE PAGE START--------------------------------------------------------------------------#

# 메인 페이지
@app.route('/')
def index():
    return render_template('nuri_amend.html')

@app.route('/zone1')
def zone1_page():
    tmap_api_key = secrets['api_keys']['tmap_api_key']
    # 기본 값 설정
    zone, month, day, hour = None, None, None, None
    buttons_visible = False    

    if request.args:  # 사용자가 폼을 제출했을 때
        zone = 'zone1'
        month = request.args.get('month', default=None)
        day = request.args.get('day', default=None)
        hour = request.args.get('hour', default=None)
        month, day , hour = user_input_datetime()
        zone_id_list = load_zone_id(zone)

        print(f"사용자 입력값 - month: {month}, day: {day}, hour: {hour}")
        # LGBM 클래스 메서드 호출
        try:
            # ----------------- LGBM 예측 ----------------- #
            LGBM_facility_list = LGBMRegressor.load_LGBMfacility()
            m, h, w = LGBMRegressor.get_LGBMtime(month, day, hour)
            input_df = LGBMRegressor.merge_LGBM_facility_time(LGBM_facility_list, m, h, w)
            LGBM_pred_fin = LGBMRegressor.LGBMpredict(input_df)

             # ----------------- LSTM 예측 ----------------- #
            target_DT = datetime(2024, month, day, hour)  # ⭐ int로 안 바꿔도 되는지 확인!
            before168_DT = target_DT - timedelta(hours=168)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            time_series_data = LSTM_Bidirectional.get_time_series_data(
                project_id="multi-final-project",
                dataset_id="Final_table_NURI",
                table_id="LSTM_data_for_forecast_cloudsql",
                before168_DT=before168_DT,
                target_DT=target_DT
            )

            lstm_model = LSTM_Bidirectional(model_path=LSTM_model_path)
            LSTM_pred_fin = lstm_model.predict(
                project_id="multi-final-project",
                dataset_id="Final_table_NURI",
                table_id="LSTM_data_for_forecast_cloudsql",
                before168_DT=before168_DT,
                target_DT=target_DT,
                device=device
            )
            print("LSTM Bidirectional model loaded.")


            # BigQuery 데이터 가져오기
            stock_list = load_stock(zone, month, day, hour)
            merged_result = merge_result(LGBM_pred_fin, LSTM_pred_fin, stock_list, input_df)

            # 자전거 재배치 데이터 형성
            # 1. 수요 부족/충분
            station_status_dict = find_station_status(merged_result)
            # (D-2) supply_demand 계산
            supply_demand = make_supply_list(zone, station_status_dict)
            # (D-3) 거리 정보
            zone_distance = load_zone_distance(zone)
            # (D-4) 최적화
            results_dict = Bike_Redistribution(supply_demand, zone_distance, station_status_dict)

            # 완료 시 버튼 활성화
            buttons_visible = True

        except Exception as e:
            print(f"[ERROR] {str(e)}")
            # 예외 발생 시에도 최소한의 정보만 넘김
            stocks = []
            results_dict = {}
            pass

        # 최종적으로 month/day/hour를 2자리 문자열로 변환해서 렌더링에 전달
        month_str = str(month).zfill(2)
        day_str = str(day).zfill(2)
        hour_str = str(hour).zfill(2)

        return render_template(
            'zone1.html',
            buttons_visible=buttons_visible,
            tmap_api_key=tmap_api_key,
            month=month_str,
            day=day_str,
            hour=hour_str
        )
    
    # 사용자가 아무 것도 선택 안 했을 때(단순 GET)
    return render_template(
        'zone1.html',
        buttons_visible=buttons_visible,
        tmap_api_key=tmap_api_key
    )
@app.route('/zone2')
def zone2_page():
    tmap_api_key = secrets['api_keys']['tmap_api_key']
    zone = None # ⭐️
    month = None
    day = None
    hour = None
    buttons_visible = False
    return render_template('zone2.html',buttons_visible = buttons_visible, tmap_api_key = tmap_api_key, month=month, day=day, hour=hour)
# ---------------- ZONE PAGE END --------------------------------------------------------------------------#

#------------------ 모델 예측 준비 함수 START -------------------------------------------------------------------------#
# 사용자 날짜 및 시간 입력
def user_input_datetime():
    month = int(request.args.get('month'))
    day = int(request.args.get('day'))
    hour = int(request.args.get('hour'))
    return month, day, hour

# zone별 대여소ID 불러오기
def load_zone_id(zone):
    zone_id_list = []
    zone_id_path = os.path.join(BASE_DIR, f'data/{zone}_station_id_list.txt')
    with open(zone_id_path, 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            zone_id_list.append(line.strip())
    return zone_id_list

# 위도 경도 데이터
def load_LatLonName():
    station_LatLonName_dict = {}
    station_LatLonName_path = os.path.join(BASE_DIR, './data/station_latlon.csv')
    with open(station_LatLonName_path, 'r', encoding='utf-8-sig') as fr:
        reader = csv.DictReader(fr)
        for row in reader:
            stationID = row['Station_ID']
            station_LatLonName_dict[stationID] = {
                "Latitude": row['Latitude'],
                "Longitude": row['Longitude'],
                "Station_name": row['Station_name']
            }
        return station_LatLonName_dict # ★여기 Flask에서 한글 깨지는거 수정해야 함.★
    
#------------------ 모델 예측 준비 함수 END ------------------------------------------------------------------#


#------------------ LSTM START -----------------------------------------------------------------------------#
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

#-------------------- LSTM END -----------------------------------------------------------------------------#


#-------------------- LGBM START ---------------------------------------------------------------------------#
class LGBMRegressor:
    # LGBM모델에 사용되는 input dataframe과 주변시설 정보 불러오기
    @staticmethod
    def load_LGBMfacility():
        LGBM_facility_list = []
        LGBM_facility_path = os.path.join(BASE_DIR,'./data/station_facilities.csv')
        with open (LGBM_facility_path, 'r') as fr:
            reader = csv.reader(fr)
            next(reader)
            for row in reader:
                LGBM_facility_list.append(tuple(row))
        return LGBM_facility_list
        # [('ST-1171', 6, 2, 0, 2, 1, None, None, None), ...] 한 줄은 tuple, 전체는 list

    #LGBM모델 예측에 필요한 시간 함수 (1시간 timedelta)
    @staticmethod
    def get_LGBMtime(month, day, hour):
        kst = pytz.timezone('Asia/Seoul')
        now_kst = datetime.now(kst)
        kst_1h_timedelta = now_kst + timedelta(hours=1)
        year = kst_1h_timedelta.year
        date = datetime(year, month, day, hour) + timedelta(hours=1)
        if date.weekday() < 5:
            weekday = 1
        else:
            weekday = 0
        return month, hour, weekday
  
    @staticmethod
    def merge_LGBM_facility_time(LGBM_facility_list, month, hour, weekday):
        # facility data 불러와서 dataframe화 
        columns = ['Rental_Location_ID', 'bus_stop', 'park', 'school', 'subway', 'riverside']
        input_df = pd.DataFrame(LGBM_facility_list, columns=columns)

        # 사용자 시간 입력 받아오기
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
                print(f"ERROR: {col} is not categorical nor numeric")
        
        # 디버깅 코드
        # print("input_df")
        # for index, row in input_df.iterrows():
        #     if index > 5:
        #         break
        #     print(row)

        return input_df
        # 'Rental_Location_ID', 'bus_stop', 'park', 'school', 'subway', 'riverside', 'month', 'hour', 'weekday'
   
    @staticmethod
    # 모델 불러오기
    def load_LGBMmodel():
        LGBM_model_path = os.path.join(BASE_DIR, './model/250109_NEW_LGBMmodel.pkl')
        with open (LGBM_model_path, 'rb') as file:
             LGBM_model = pickle.load(file)
        return LGBM_model

    @staticmethod
    # 모델 사용해서 대여소별 수요 예측
    def LGBMpredict(input_df):
        LGBM_model = LGBMRegressor.load_LGBMmodel()
        model = LGBM_model
        LGBM_pred = model.predict(input_df)
        LGBM_pred_fin = LGBM_pred[np.newaxis, :]  
        return LGBM_pred_fin # type : np.ndarray / 소수점 형태


#-- LGBM END -----------------------------------------------------------------------------------------------------------------#

def load_stock(zone, month, day, hour):
    zone_id_list = load_zone_id(zone)
    zone_id_tuple = tuple(zone_id_list)
    
    # user input 시간만 stock 불러옴
    input_date = datetime(2023, month, day)
    input_date = str(input_date.strftime('%Y-%m-%d'))
    input_time = int(hour)

    # Bigquery에서 해당 기간 stock 내역 불러옴
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
    # print("\nlen(stock_list): ", len(stock_list))
    return stock_list
    # stock_list = [{
    #               'Date': datetime.date(2023, 4, 2), 
    #               'Time': 6, 
    #               'stock': 3.0, 
    #               'Rental_location_ID': 'ST-3164', 
    #               'Name_of_the_rental_location': '청담역 1번출구'},  
    #                ... ]

def merge_result(LGBM_pred_fin, LSTM_pred_fin, stock_list, input_df):
    # 1. 양상블
    ensemble_array = np.rint((LGBM_pred_fin + LSTM_pred_fin) / 2).astype(int) # shape (1, N)  # rint : 가장 가까운 정수로 반올림
    ensemble_list = ensemble_array[0].tolist()

    # 2. stock과 prediction 병합
    # stock_list = 
        #         [{
            #     "Date": "Wed, 01 Mar 2023 00:00:00 GMT",
            #     "Name_of_the_rental_location": "언주역 3번 출구",
            #     "Rental_location_ID": "ST-1577",
            #     "Time": 12,
            #     "stock": 1 (int)
    #       },...]
    selectedzone_stock_dict = {}
    for i in range(len(stock_list)):
        rental_location_id = stock_list[i]["Rental_location_ID"]
        stock = stock_list[i]["stock"]
        selectedzone_stock_dict[rental_location_id] = stock
    
    # 디버깅 출력
    # print("\nselectedzone_stock_dict")
    # for key, value in selectedzone_stock_dict.items():
    #     print(f"{key}: {value}")

    # 4. 대여소별 결과 dict 작성
    merged_result = {}
    for i, row in input_df.iterrows():  # 161개의 대여소
        stationid = row['Rental_Location_ID']
        ens_val  = ensemble_list[i] # Ensemble 예측
        station_item = selectedzone_stock_dict.get(stationid, None) # zone에 맞는 대여소 선별
        if station_item is not None:
            merged_result[stationid] = {
                "predicted_rental": ens_val,
                "stock": station_item
            }
        # else:
        #     print(f"{stationid}: not in {zone} (다른 관리권역에 있거나 31개에 포함되지 않음)")

    # 디버깅 출력
    print("\nmerged_result")
    for key, value in merged_result.items():
        print(f"{key}: {value}")
    return merged_result
            # {
            #   "ST-1577": { "predicted_rental": 3, "stock": 1 },
            #   "ST-784": { "predicted_rental": 5, "stock": 4 }, ...
            # }

def find_station_status(merged_result): #abundant, deficient labeling 하기
    # 1. 순서대로 status 구하기
    for stationid, item in merged_result.items():
        stock = item["stock"]
        predicted_rental = item['predicted_rental']
        status = stock - (predicted_rental + 3) # 예측된 수요량보다 3개 더 많아야 함
        if status < 0:
            merged_result[stationid]["status"] = "deficient"
        else:
            merged_result[stationid]["status"] = "abundant"

    # 2. 임의로 center 정보 추가
    merged_result["center"] = {
            "predicted_rental": 0,
            "stock": 0,
            "status": "abundant"                
        }
    station_status_dict = merged_result
    return station_status_dict # center 까지 포함 (필수)

center_flag = False # Center 처리를 위해 글로벌 변수 사용

def make_supply_list(zone, station_status_dict):
    global center_flag
    # station_status_dict = 
    # "ST-1561": {
    #     "predicted_rental": 0,
    #     "status": "deficient",
    #     "stock": 2.0
    #   }, ... center 포함

    # 1. 일단 모두 추가 (Center 값: 0)
    zone_id_list = load_zone_id(zone)

    supply_demand = []
    for station_id in zone_id_list:
        station_info = station_status_dict.get(station_id, None)

        if station_info is None:
            print(f"ERROR! {station_id} : no data in station_status_dict")
            continue

        # 1. deficient = 예상 수요보다 stock이 부족한 경우
        if station_info["status"] == "deficient":
            # 1-1. stock이 아예 없는 경우 3개 필요하다고 입력
            if station_info["stock"] == 0:
                supply_demand.append(-3)
            # 1-2. 예상 수요 +2 만큼 demand로 설정
            else:
                supply_demand.append(int(-station_info["predicted_rental"]) -2)

        # 2. abundant = 예상 수요보다 3개 이상의 stock을 가진 경우
        elif station_info["status"] == "abundant":
            supply_demand.append(int(station_info["stock"])) # abundant한 경우: stock 그대로 넣기 (Center도 그대로 들어감)
        
        # 디버깅    
        else:
            print(f"ERROR! {station_id} : no status info")
        # 디버깅 코드 : print(f"{station_id} {station_info["status"]} ", append_value)
    
    # 2. supply_demand 총합에 알맞는 center 처리
    # 총합 >= 0 인 경우 : 위의 abundant로 이미 center 값 '0'으로 처리
    # 총합 < 0 인 경우 : 아래의 코드를 실행하여 center 값을 'supply_sum'으로 처리
    if sum(supply_demand) < 0:
        center_flag = True
        supply_demand.pop()
        supply_sum = sum(supply_demand)
        supply_demand.append(-supply_sum)
        print("\n[INFO]공급 부족으로 center에 양수 처리 시행!")

    print("\n[center_flag]: ", center_flag)
    print("[supply_demand]: ", supply_demand)
    print("[sum(supply_demand)]: ", sum(supply_demand), "<- Center flag True일 때는 0")
    return supply_demand # 길이는 Center 포함까지 (필수)

def load_zone_distance(zone):
    zone_distance = []
    zone_distance_path = os.path.join(BASE_DIR, f'./data/{zone}_distance.csv')
    # print("\nload_zone_distance(zone) 실행!")
    with open (zone_distance_path, 'r') as fr:
        lines = fr.readlines()
        for line in lines[1:]:
            values = line.strip().split(",")         # ['ST-786', '0', '2.83', '1.78', '2.18']
            distance_values = values[1:]             # 맨 앞에 ST-..는 건너뛰기 -> ['0', '2.83', '1.78', '2.18']
            row = list(map(float, distance_values))  # [0.0, 2.83, 1.78, 2.18]
            zone_distance.append(row)
            # print(values)   
    return zone_distance # center까지 모두 추가(필수)

def station_index(supply_demand):
    station_index_data = {}
    for i, supply in enumerate(supply_demand):
        station_index_data[i] = supply # {'0': -3, '1': 34, '2': -7 ... } Center까지 모두 있음
    print("\n[station_index_data]: ", station_index_data)
    return station_index_data

def Bike_Redistribution(supply_demand, zone_distance, station_status_dict):
    # 0. 인덱스(station_index)와 대여소 이름 매칭
    station_names_dict = {}
    for i, id in enumerate(station_status_dict.keys()):
        station_names_dict[i] = id
    print(f"\n station_names_dict: {station_names_dict}")
    
    # -------------------- Bike Redistribution 시작--------------------------------- #
    # 1. 데이터 정의
    supply = supply_demand # center가 처리된 상태
    num_stations = len(supply)
    cost = zone_distance

    ## 디버깅 코드
    if num_stations == len(cost):
        print("\nnum_stations랑 len(cost) 일치!")
    else:
        print("\nERROR: len(num_stations)랑 len(cost) 불일치!")

    # 2. 문제 해결 시작
    problem = pulp.LpProblem("Bike_Redistribution", pulp.LpMinimize)

        ## 변수 정의: x_ij는 i에서 j로 이동하는 자전거 수
    x = pulp.LpVariable.dicts("x", ((i, j) for i in range(num_stations) for j in range(num_stations)), lowBound=0, cat="Integer")

        ## 목표 함수: 총 이동 비용 최소화
    problem += pulp.lpSum(cost[i][j] * x[i, j] for i in range(num_stations) for j in range(num_stations))

        ## 여유 대여소에서 자전거 이동량 제한
    for i in range(num_stations):
        if supply[i] > 0:  # 여유 대여소
            problem += pulp.lpSum(x[i, j] for j in range(num_stations) if i != j) <= supply[i]

        ## 부족 대여소의 수요 충족
    for j in range(num_stations):
        if supply[j] < 0:  # 부족 대여소
            problem += pulp.lpSum(x[i, j] for i in range(num_stations) if i != j) >= -supply[j]
                
        ## 부족 대여소에서 자전거 이동 금지 조건 추가
    for i in range(num_stations):
        if supply[i] < 0:  # 부족 대여소
            problem += pulp.lpSum(x[i, j] for j in range(num_stations) if i != j) == 0

    ## 제약 조건 : 반드시 center(start_station)에서 출발
    start_station = num_stations - 1 # center의 인덱스
    problem += pulp.lpSum(x[start_station, j] for j in range(num_stations) if j != start_station) >= 1
        
    # 3. 문제 해결
    Distibution_result = problem.solve()

    # 4. 결과 출력
    solve_status = pulp.LpStatus[problem.status]
    print("\nStatus:", solve_status)
    
    # 5. 결과 dict로 저장
    results_dict = {"status": solve_status, "moves": []}
    for i in range(num_stations):
        for j in range(num_stations):
            if x[i, j].varValue is not None and x[i, j].varValue > 0:
                # print(f"1. x[{i}, {j}] = {x[i, j].varValue}") # 디버깅 코드
                from_name = station_names_dict[i]
                to_name = station_names_dict[j]
                print(f"2. From {from_name}({i}) to {to_name}({j}), move bikes: {x[i, j].varValue}") # 디버깅
                cur_station_dict = station_status_dict.get(to_name)
                if not cur_station_dict: # 해당 대여소는 방문하지 않음
                    print(f"\n{to_name}: no visit!")
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
                # print(f"3. result_dict- From {from_name}({i}) to {to_name}({j}), move bikes: {x[i, j].varValue}")
    # 디버깅 코드
    print("\n[result_check] 0115 기준 디버깅 필요. 일단 return으로 반환은 됨.")
    result_check = results_dict.get("moves")
    for move_value in result_check:  # 리스트의 각 원소를 반복
        print(move_value)
    return results_dict

# # #후처리 함수 ---> 수정 필요!!
# def simplify_movements(supply_demand, zone_distance, station_index_data):
#     simplified_moves = {}
#     simplified_flag = False # 후처리 여부 확인
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
#             from_name = station_index[i]
#             to_name = station_index[j]
#             print(f" 후처리 후- Move {amount} bikes from {from_name}({i}) to {to_name}({j})")
#         print("후처리 진행됨!")
#     else:
#         print("후처리 불필요")
#     return simplified_moves

# # 인덱스 추가 및 파라미터 정리 함수 #POST 대상
# def final_route(simplified_moves, results_dict, stock_and_status, station_LatLonName_dict):
#     # 후처리된 결과
#     # simplified_moves 예시 : {(1, 0): 5.0, (1, 2): 4.0, (1, 6): 3.0, (4, 12): 5.0, (5, 3): 3.0, (10, 15): 6.0, (13, 14): 5.0}

#     # 1. 대여소 상태 dict
#         # results_dict["moves"].append({
#         #     "from_station": from_name,
#         #     "from_index": i,
#         #     "to_station": to_name,
#         #     "to_index": j,
#         #     "bikes_moved": x[i, j].varValue,
#         # })

#     # 3. station_latlon
#      = load_LatLonName()

#     # 4. 경로 결과값 출력
#     previous_from_station = None
#     simple_moves = []
#     for i, (from_station, to_station), move in enumerate(simplified_moves.items()):
#         # visit station name
#         key = (from_station, to_station)
#         to_station_id= key[1]
#         visit_station_name = station_index[to_station_id]

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
#     station_index_data = station_index(zone)
#     problem, x, solve_status = Bike_Redistribution(supply_demand, zone_distance, station_index)
#     simple_moves = final_route(x, station_index)
#     final_simple_moves = jsonify(simple_moves)
#     return final_simple_moves





# ---------------------------------------------------------------
# @app.route('/test', methods=['GET'])
# def test():
#     print("클라이언트에서 /test 요청 도착") 

#     if 'predictions' not in session:
#         return jsonify({"error": "No predictions available"}), 400
#     return jsonify({"predictions": session['predictions']})


if __name__ == "__main__":
    app.run(debug=True)