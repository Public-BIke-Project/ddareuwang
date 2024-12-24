from flask import Flask, jsonify, render_template, request, redirect, url_for
from sqlalchemy import create_engine, text
import json
import pandas as pd
from datetime import datetime, timedelta
import toml
import pytz
import pickle
from google.cloud import storage
from math import ceil
import numpy as np

# TOML
with open('./secrets/secrets.toml', 'r') as fr:
    secrets = toml.load(fr)

#Flask
app = Flask(__name__)
app.secret_key = secrets['app']['flask_password'] # Flask의 session 기능 사용

# CloudSQL 연결
USER = secrets['database']['user']
PASSWORD = secrets['database']['password']
HOST = secrets['database']['host']
PORT = secrets['database']['port']
NAME = secrets['database']['name']
engine = create_engine(f"mysql+pymysql://{USER}:{PASSWORD}@{HOST}:{PORT}/{NAME}")

# 메인 페이지
@app.route('/')
def index():
    return render_template('nuri_amend.html')

@app.route('/zone1')
def zone1_page():
    tmap_api_key = secrets['api_keys']['tmap_api_key']    # 기본값 설정
    zone = 'zone1' # ⭐️
    year = None
    month = None
    day = None
    hour = None
    buttons_visible = False
    if request.args:  # 사용자가 폼을 제출했을 때
        kst = pytz.timezone('Asia/Seoul')
        now_kst = datetime.now(kst)
        year = now_kst.year
        month = request.args.get('month')  # 'month' 입력 필드의 값
        day = request.args.get('day')      # 'day' 입력 필드의 값
        hour = request.args.get('hour')    # 'hour' 입력 필드의 값
        month, day , hour = user_input_datetime()
        if month and day and hour:
            # 폼이 제출되면 버튼을 보이도록 설정
            buttons_visible = True
            year = str(year)
            month = str(month).zfill(2)
            day = str(day).zfill(2)
            hour = str(hour).zfill(2)
    
    zone_id_list = load_zone_id(zone) # ⭐️
    zone_distance = MakeRoute.load_zone_distance(zone) # ⭐️

    # GET 요청 시 HTML 폼 렌더링
    return render_template('zone1.html',buttons_visible = buttons_visible, tmap_api_key = tmap_api_key, month=month, day=day, hour=hour)

@app.route('/zone2')
def zone2_page():
    tmap_api_key = secrets['api_keys']['tmap_api_key']    # 기본값 설정
    zone = 'zone2' # ⭐️
    year = None
    month = None
    day = None
    hour = None
    buttons_visible = False
    if request.args:  # 사용자가 폼을 제출했을 때
        kst = pytz.timezone('Asia/Seoul')
        now_kst = datetime.now(kst)
        year = now_kst.year
        month = request.args.get('month')  # 'month' 입력 필드의 값
        day = request.args.get('day')      # 'day' 입력 필드의 값
        hour = request.args.get('hour')    # 'hour' 입력 필드의 값
        month, day , hour = user_input_datetime()
        if month and day and hour:
            # 폼이 제출되면 버튼을 보이도록 설정
            buttons_visible = True
            year = str(year)
            month = str(month).zfill(2)
            day = str(day).zfill(2)
            hour = str(hour).zfill(2)
    
    zone_id_list = load_zone_id(zone) # ⭐️
    zone_distance = MakeRoute.load_zone_distance(zone) # ⭐️
    # GET 요청 시 HTML 폼 렌더링
    return render_template('zone2.html',buttons_visible = buttons_visible, tmap_api_key = tmap_api_key, month=month, day=day, hour=hour)

# 사용자 날짜 및 시간 입력
def user_input_datetime():
    month = request.args.get('month')
    day = request.args.get('day')
    hour = request.args.get('hour')
    return month, day, hour

#zone별 대여소ID 불러오기
@app.route('/selectzone')
def load_zone_id(zone): 
    table_name = f"zone{zone[-1]}_id_list"
    query = text(f"SELECT * FROM {table_name};")
    with engine.connect() as connection:
        result = connection.execute(query)
        zone_id_list = result.fetchall()
    return zone_id_list

# 위도 경도 데이터 
def load_LatLonName():
    query = text("SELECT * FROM bike_stations;")  # 총 32row (관리권역1,2 대여소 32개만 있음)
    station_LatLonName_dict = {}
    with engine.connect() as connection:
        result = connection.execute(query)
        for row in result.mappings():
            station_LatLonName_dict[row['Station_ID']] = {
                "Latitude": row['Latitude'],
                "Longitude": row['Longitude'],
                "Station_name": row['Station_name']
            }
    return station_LatLonName_dict # ★여기 한글 깨지는거 수정해야 함.★

class LGBMRegressor:
    # LGBM모델에 사용되는 input dataframe과 주변시설 정보 불러오기
    @staticmethod
    def load_LGBMfacility():
        query = text("SELECT * FROM station_facilities")
        with engine.connect() as connection:
            result = connection.execute(query)
            LGBM_facility_list = result.fetchall()
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
    @app.route('/input')
    def merge_LGBM_facility_time():
        # facility CloudSQL에서 불러오기
        facility = LGBMRegressor.load_LGBMfacility()
        columns = ['Rental_Location_ID', 'bus_stop', 'park', 'school', 'subway', 'riverside']
        input_df = pd.DataFrame(facility, columns=columns)
        input_df['Rental_Location_ID'] = input_df['Rental_Location_ID'].astype('category')

        # 사용자 시간 입력 받아오기
        month, hour, weekday = LGBMRegressor.get_LGBMtime() 
        time_values = {
            "month": np.full(len(input_df), month), 
            "hour": np.full(len(input_df), hour), 
            "is_weekday": np.full(len(input_df), weekday)
            }
        input_df[['month', 'hour', 'weekday']] = pd.DataFrame(time_values)
        return input_df # 'Rental_Location_ID', 'bus_stop', 'park', 'school', 'subway', 'riverside', 'month', 'hour', 'weekday'
   
    @staticmethod
    # 모델 불러오기
    def load_LGBMmodel_from_gcs(
            bucket_name='bike_data_for_service', 
            source_blob_name='model/241121_model_ver2.pkl'
            ):
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)    
        file_content = blob.download_as_bytes()
        LGBM_model = pickle.loads(file_content)
        return LGBM_model

    @staticmethod
    @app.route('/predict')
    # 모델 사용해서 대여소별 수요 예측
    def LGBMpredict():
        model = LGBMRegressor.load_LGBMmodel_from_gcs(
            bucket_name='bike_data_for_service',
            source_blob_name='model/241121_model_ver2.pkl'
        )
        input_df = LGBMRegressor.merge_LGBM_facility_time()
        predictions = model.predict(input_df)
        print("\npredictions: ", predictions)
        return predictions # type : np.ndarray / 소수점 형태

    @staticmethod
    @app.route('/stock')
    def load_LGBMstock():
        # 해당 관리권역만
        zone = 2
        if zone is None:
            return jsonify({"error": "Zone parameter is required"}), 400
        zone_id_list = load_zone_id(zone)
        zone_id_tuple = tuple(row[0] for row in zone_id_list)
        
        # 해당 시간만
        month, day, hour = user_input_datetime()
        input_date = datetime(2023, month, day)
        input_time = datetime.strptime(f"{hour}:00:00", "%H:%M:%S").time()

        # 해당 관리권역 및 시간만 불러와서 append
        # 2023_available stocks :[{Date, Time, stock, Rental_location_ID, Name_of_the_rental_location}, {...}, ... ]
        LGBM_stock_list = []
        try:
            query = text("""
                        SELECT * 
                        FROM 2023_available_stocks 
                        WHERE Rental_location_ID IN :ids
                            AND Date = :date
                            AND Time = :time
                        """)
            with engine.connect() as connection:
                result = connection.execute(query, {"ids": zone_id_tuple, "date": input_date, "time": input_time})
                for row in result.mappings():
                    LGBM_stock_list.append(dict(row))
            return LGBM_stock_list

        except Exception as e:
            return jsonify({"error": str(e)}), 500
    

@staticmethod        
@app.route('/merge')                                      
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

@staticmethod        
@app.route('/status')                                      
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
    
class MakeRoute:
    @staticmethod
    @app.route('/distance')                                      
    def load_zone_distance(zone):
        # ★밑의 코드가 찐★
        # zone = request.args.get('zone', type=int)
        #     if zone is None:
        #         return jsonify({"error": "Zone parameter is required"}), 400
        # 요청 예시 http = GET /stock?zone=2 
        table_name = f"zone{zone[-1]}_id_list"
        query = text(f"SELECT * FROM {table_name};")
        with engine.connect() as connection:
            result = connection.execute(query)
            zone_distance = result.fetchall() # 리스트형태의 튜플 (각 row는 tuple, 전체는 list)
            # result = [
            #                 (1, 'Alice', 25),
            #                 (2, 'Bob', 30),
            #                 (3, 'Charlie', 35)
            #             ]
        return zone_distance
    
    @staticmethod
    @app.route('/supply')
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
                    supply_demand.append(-3)  # stock이 아예 없는 경우 5개 필요하다고 입력
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
    
    @staticmethod
    @app.route('/names')
    def station_names():
        zone = 2
        # ★밑의 코드가 찐★
        # zone = request.args.get('zone', type=int)
        #     if zone is None:
        #         return jsonify({"error": "Zone parameter is required"}), 400
        zone_distance = MakeRoute.load_zone_distance(zone)

        station_names = {}
        for i, row in enumerate(zone_distance):
            station_names[i] = row[0] # row[0] = 대여소 ID
            if i == len(zone_distance) - 1:
                break

        supply_demand = MakeRoute.make_supply_list()
        if sum(supply_demand[:-1]) < 0: # 대여가능수량이 부족해서 Center에서 출발할 때 자전거를 적재해야 하는 경우
            station_names[len(zone_distance)] = "center"
            print("def station_names()로 station_names에 Center 추가!")
            print("\nstation_names[len(zone_distance)]: ", station_names[len(zone_distance)])
        return station_names


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
    LGBMRegressor_model = LGBMRegressor.load_LGBMmodel_from_gcs(
            bucket_name='bike_data_for_service', 
            source_blob_name='model/241121_model_ver2.pkl'
            )
    app.run(debug=True)





