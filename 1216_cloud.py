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
        facility = LGBMRegressor.load_LGBMfacility()
        columns = ['Rental_Location_ID', 'bus_stop', 'park', 'school', 'subway', 'riverside', 'month', 'hour', 'is_weekday']
        input_df = pd.DataFrame(facility, columns=columns)

        month, hour, weekday = LGBMRegressor.get_LGBMtime()
        fill_values = {"month": int(month), "hour": int(hour), "is_weekday": int(weekday)}
        input_df.fillna(value=fill_values, inplace=True)
        input_df['Rental_Location_ID'] = input_df['Rental_Location_ID'].astype('category')
        print("input data length", len(input_df))
        return input_df
   
    @staticmethod
    def load_LGBMmodel_from_gcs(bucket_name='bike_data_for_service', source_blob_name='model/241121_model_ver2.pkl'):
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)    
        file_content = blob.download_as_bytes()
        LGBM_model = pickle.loads(file_content)
        return LGBM_model

@app.route('/merge') 
def LGBMpredict():
    model = LGBMRegressor.load_LGBMmodel_from_gcs(
        bucket_name='bike_data_for_service',
        source_blob_name='model/241121_model_ver2.pkl'
    )
    input_df = LGBMRegressor.merge_LGBM_facility_time()
    
    predictions = model.predict(input_df)
    return predictions # type : np.ndarray
    
# @app.route('/merge')                                               
# def merge_LGBMresult():
#     # 1. input data
#     input_df = LGBMRegressor.merge_LGBM_facility_time()
#     # 2. prediction
#     predictions = LGBMRegressor.LGBMpredict()
#     predictions = int(ceil(predictions)) # 올림하여 predictions을 정수로 만듦
#     # 3. stock
#     merged_result = []
#     for i in range(len(input_df)):
#         row = input_df.iloc[i]
#         predicted_value = predict_bike_response[i]
#         current_stock = station_current_stock[i]
#         merged_result.append({
#             "station_id": row["Rental_Location_ID"],
#             "predicted_rental": predicted_value,
#             "stock" : current_stock['stock']
#         })

#     query = text("SELECT * FROM 2023_available_stocks;")
#     merged_result = []
#     with engine.connect() as connection:
#         result = connection.execute(query)
#         for i in range(len(input_df)):
#         row = input_df.iloc[i]
#         predicted_value = predictions[i]
#         current_stock = station_current_stock[i]
#         merged_result.append({
#             "station_id": row["Rental_Location_ID"],
#             "predicted_rental": predicted_value,
#             "stock" : current_stock['stock']
#         })

#         for row in result.mappings():
#             station_LatLonName_dict[row['Station_ID']] = {
#                 "Latitude": row['Latitude'],
#                 "Longitude": row['Longitude'],
#                 "Station_name": row['Station_name']
#             }




#     current_stock = ModelInput.load_currentstock_latlon(api_key=BIKE_API_KEY, station_file_path=station_file_path)
    
#     merged_result = []
#     for i in range(len(input_dataframe)):
#         row = input_dataframe.iloc[i]
#         predicted_value = predict_bike_response[i]
#         current_stock = station_current_stock[i]
#         merged_result.append({
#             "station_id": row["Rental_Location_ID"],
#             "predicted_rental": predicted_value,
#             "stock" : current_stock['stock']
#         })
#     return merged_result




if __name__ == "__main__":
    LGBMRegressor_model = LGBMRegressor.load_LGBMmodel_from_gcs(
            bucket_name='bike_data_for_service', 
            source_blob_name='model/241121_model_ver2.pkl'
            )

    app.run(debug=True)

    
    # Flask에 표시할 stock 데이터
    # def load_LGBMstock():
    #     month, day, hour = user_input_datetime()
    #     inputDate = datetime(2023, month, day)
    #     inputTime = hour

    #     query = text("SELECT * FROM 2023_available_stocks WHERE Date=:inputDate AND Time=:inputTime")
    #     stock = []
    #     with engine.connect() as connection:
    #         result = connection.execute(query, {"inputDate": inputDate, "inputTime": inputTime})
    #         stock = result.fetchall()

    #     # 여기서 필요한 zone id list만 가져올 것임

    #     @app.route('/')
    # def load_stock(dt):
    #     datetime_data = datetime(dt) #datetime형태로 바꾸기
    #     input_date = datetime_data.date
    #     input_time = datetime_data.timestamp
    #     input_stock = stock

    #     engine = create_sqlalchemy_engine()
    #     query = text("SELECT * FROM 2023_available_stocks WHERE Date='input_date' AND Time='input_time';)
    #     with engine.connect() as connection:
    #         result = connection.execute(query)
    #         stock_df = pd.DataFrame(result.fetchall(), columns=result.keys())
    #     return jsonify(stock_df)

    #     return stock