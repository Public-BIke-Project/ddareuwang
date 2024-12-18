from flask import Flask, jsonify, render_template, request
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
app.secret_key = secrets['app']['flask_password'] # Flask의 session 사용

# CloudSQL 연결
USER = secrets['database']['user']
PASSWORD = secrets['database']['password']
HOST = secrets['database']['host']
PORT = secrets['database']['port']
NAME = secrets['database']['name']
engine = create_engine(f"mysql+pymysql://{USER}:{PASSWORD}@{HOST}:{PORT}/{NAME}")


@app.route('/')
def index():
    return render_template('nuri_amend.html')

@app.route('/zone1')
def zone1_page():
    return render_template('zone1.html')

@app.route('/zone2')
def zone2_page():
    return render_template('zone2.html')

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
   
# 관리권역 설정
def load_zone_id(zone): # ★Flask에서 누른 권역에 따라 달라지도록 변경해야 함.★
    zone = 2 # 임시
    # ★밑의 코드가 찐★
    # zone = request.args.get('zone', type=int)
    #     if zone is None:
    #         return jsonify({"error": "Zone parameter is required"}), 400
    # 요청 예시 http = GET /stock?zone=2
    table_name = f"zone{zone}_id_list"
    query = text(f"SELECT * FROM {table_name};")
    with engine.connect() as connection:
        result = connection.execute(query)
        zone_id_list = result.fetchall()
    return zone_id_list

# 입력한 DateTime 불러오기
def user_input_datetime():
    # month = request.args.get('month')
    # day = request.args.get('day')
    # hour = request.args.get('hour')
    month = 3
    day = 1
    hour = 12   
    return month, day, hour

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

    #LGBM모델 예측에 필요한 시간 함수
    @staticmethod
    def get_LGBMtime():
        kst = pytz.timezone('Asia/Seoul')
        current_time = datetime.now(kst)
        year = current_time.year # 현재 년도로 예측
        month, day, hour = user_input_datetime()    
        date = datetime(year, month, day)
        if date.weekday() < 5:
            weekday = 1
        else:
            weekday = 0
        return month, hour, weekday
    
    @staticmethod
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
    # 모델 사용해서 대여소별 수요 예측
    def LGBMpredict():
        model = LGBMRegressor.load_LGBMmodel_from_gcs(
            bucket_name='bike_data_for_service',
            source_blob_name='model/241121_model_ver2.pkl'
        )
        input_df = LGBMRegressor.merge_LGBM_facility_time()
        
        predictions = model.predict(input_df)
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
        merged_result = []
        for i in range(len(input_df)): # 관리권역 1, 2 모두 들어있음
            input_row = input_df.iloc[i]
            predicted_value = predictions_list[i]
            stationid = input_row['Rental_Location_ID']

            station_item = LGBM_dict.get(stationid, None)
            if station_item:
                merged_result.append({
                    "station_id": stationid,
                    "predicted_rental": predicted_value,
                    "stock": station_item["stock"]
                })
            else:
                print(f"{stationid}: in the other zone")
        return merged_result

if __name__ == "__main__":
    LGBMRegressor_model = LGBMRegressor.load_LGBMmodel_from_gcs(
            bucket_name='bike_data_for_service', 
            source_blob_name='model/241121_model_ver2.pkl'
            )
    app.run(debug=True)