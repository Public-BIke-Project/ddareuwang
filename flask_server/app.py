from flask import Flask, jsonify, render_template, request, session
from sqlalchemy import create_engine, text
import csv
import json
import pandas as pd
from datetime import datetime, timedelta
import toml
import logging
import pickle

# TOML
with open('./secrets/secrets.toml', 'r') as fr:
    secrets = toml.load(fr)

#Flask
app = Flask(__name__)
app.secret_key = secrets['app']['flask_password'] # Flask의 session 사용
logging.basicConfig(level=logging.DEBUG)

# CloudSQL 연결
USER = secrets['database']['user']
PASSWORD = secrets['database']['password']
HOST = secrets['database']['host']
PORT = secrets['database']['port']
NAME = secrets['database']['name']
engine = create_engine(f"mysql+pymysql://{USER}:{PASSWORD}@{HOST}:{PORT}/{NAME}")


# # 엔진 생성
# engine = init_connection_pool()

@app.route('/')
def home():
    return render_template('nuri_amend.html')  # templates/index.html 파일을 서빙

# 1권역 페이지
@app.route('/zone1')
def zone1_page():
    tmap_api_key = secrets['api_keys']['tmap_api_key']
    return render_template('zone1.html',tmap_api_key = tmap_api_key)

@app.route('/zone2', methods=['GET'])
def zone2_page():
    tmap_api_key = secrets['api_keys']['tmap_api_key']
    # 기본값 설정
    month = None
    day = None
    hour = None
    buttons_visible = False    
    if request.args:  # 사용자가 폼을 제출했을 때
        month = request.args.get('month')  # 'month' 입력 필드의 값
        day = request.args.get('day')      # 'day' 입력 필드의 값
        hour = request.args.get('hour')    # 'hour' 입력 필드의 값
        
        if month and day and hour:
            # 폼이 제출되면 버튼을 보이도록 설정
            buttons_visible = True
        
    # GET 요청 시 HTML 폼 렌더링
    return render_template('zone2.html',buttons_visible = buttons_visible, tmap_api_key = tmap_api_key, month=month, day=day, hour=hour)

        # # 정규화
        # normalized_month = month / 12
        # normalized_day = day / 31
        # normalized_hour = hour / 24

        # # 동적으로 대여소 ID 조회
        # try:
        #     with engine.connect() as connection:
        #         # 대여소 ID 조회
        #         column_query = text("""
        #             SHOW COLUMNS
        #             FROM LSTM_data_for_forecast
        #             WHERE Field LIKE 'ST-%'
        #         """)
        #         columns_result = connection.execute(column_query)
        #         station_columns = [row['Field'] for row in columns_result]

        #         # 동적으로 SELECT 쿼리 작성
        #         select_query = f"""
        #             SELECT {', '.join(station_columns)}
        #             FROM LSTM_data_for_forecast
        #             WHERE month = :normalized_month
        #               AND day = :normalized_day
        #               AND hour = :normalized_hour
        #         """

        #         # 쿼리 실행
        #         result = connection.execute(text(select_query), {
        #             'normalized_month': normalized_month,
        #             'normalized_day': normalized_day,
        #             'normalized_hour': normalized_hour
        #         })

        #         # 결과를 JSON으로 반환
        #         data = [dict(row) for row in result]

        #         if data:
        #             print("\nsuccess!")
        #             print("\ndata!",data)
        #             return jsonify(data)
        #         else:
        #             return "No data found for the given inputs.", 404

        # except Exception as e:
        #     print(f"Database error: {e}")
        #     return "Error connecting to the database.", 500

    

BIKE_API_KEY = '787a4c4f41736d6133365464694c56'
station_id_file = './data/final_stations_109.txt'

class BikeStationChecker:
    def __init__(self, BIKE_API_KEY, station_file_path):
        self.api_key = BIKE_API_KEY
        self.station_file = station_file_path
        self.station_id_list = self.load_station_id_list()

    def load_station_id_list(self):
        """파일에서 대여소 ID 목록을 읽어옴"""
        with open(self.station_file, 'r') as fr:
            return [line.strip() for line in fr]

    def get_bike_api(self):
        """서울시 공공 자전거 API 호출"""
        response_data_1 = requests.get(f'http://openapi.seoul.go.kr:8088/{self.api_key}/json/bikeList/1001/2000/')
        bike_data_1 = response_data_1.json()
        bike_infos_1 = bike_data_1['rentBikeStatus']['row']

        response_data_2 = requests.get(f'http://openapi.seoul.go.kr:8088/{self.api_key}/json/bikeList/2001/3000/')
        bike_data_2 = response_data_2.json()
        bike_infos_2 = bike_data_2['rentBikeStatus']['row']

        total_bike_infos = bike_infos_1 + bike_infos_2
        return total_bike_infos

# @app.route('/input', methods=['GET', 'POST'])
class ModelInput:
    def load_station_facility():
        station_facility_list = []
        with open('./data/station_input_1121_ver2.csv', 'r') as fr:
            reader = csv.DictReader(fr)
            for row in reader:
                station_facility_list.append(row)
        return station_facility_list

    def load_station_currentstock(api_key=BIKE_API_KEY, station_id_file=station_id_file):
        checker = BikeStationChecker(api_key, station_id_file)
        total_bike_infos = checker.get_bike_api()

        processed_data = []
        for station in total_bike_infos:
            station_id = station['stationId']
            current_stock = int(station['parkingBikeTotCnt'])

            processed_data.append({
                'station_id': station_id,
                'stock': current_stock
            })    
        return processed_data
    
    def get_time():
        kst = pytz.timezone('Asia/Seoul')
        timedelta_1h_kst = datetime.now(kst) + timedelta(hours=1)
        
        # month = timedelta_1h_kst.month
        # hour = timedelta_1h_kst.strftime('%H')
        # weekday = timedelta_1h_kst.weekday()
        month = 11
        hour = 6
        weekday = 1

        return (month, hour, weekday)

    @app.route('/input', methods=['GET', 'POST'])
    def join_facility_stock_time(api_key=BIKE_API_KEY, station_id_file=station_id_file):
        station_facility_list = ModelInput.load_station_facility()
        station_current_stock = ModelInput.load_station_currentstock(api_key, station_id_file)
        
        month, hour, weekday = ModelInput.get_time()

        facility_dict = {facility['Rental_Location_ID']: facility for facility in station_facility_list}

        # Merge data based on station_id
        merged_data = []
        for stock in station_current_stock:
            station_id = stock['station_id']
            if station_id in facility_dict:
                merged_data.append({
                    **facility_dict[station_id],
                    'month': month,           
                    'hour': hour,           
                    'is_weekday': 1 if weekday < 5 else 0  
                })
            # else:
            #     print(station_id)
        return merged_data

    def convert_to_dataframe(input_data):
      input_data = ModelInput.join_facility_stock_time(BIKE_API_KEY, station_id_file)
      df = pd.DataFrame(input_data)
      df['Rental_Location_ID'] = df['Rental_Location_ID'].astype('category')
      
      for col in df.columns:
          if col != 'Rental_Location_ID':
              df[col] = pd.to_numeric(df[col])

      return df


with open('./model/241121_model_ver2.pkl', 'rb') as fr: 
    model = pickle.load(fr)


class prediction:
    def predict_bike_demand():    
        input_data = ModelInput.join_facility_stock_time(api_key=BIKE_API_KEY, station_id_file=station_id_file)
        input_dataframe = ModelInput.convert_to_dataframe(input_data)

        if input_dataframe.ndim == 1:
            input_array = input_array.reshape(1, -1)

        predictions = model.predict(input_dataframe)
        return predictions

    @app.route('/predict', methods=['GET'])                                               
    def merge_input_prediction_stock():
        # 1. input data
        input_data = ModelInput.join_facility_stock_time(api_key=BIKE_API_KEY, station_id_file=station_id_file)
        input_dataframe = ModelInput.convert_to_dataframe(input_data)

        # 2. prediction
        predict_bike_response = prediction.predict_bike_demand()

        # 3. stock
        station_current_stock = ModelInput.load_station_currentstock(api_key=BIKE_API_KEY, station_id_file=station_id_file)
        merged_result = []
        for i in range(len(input_dataframe)):
            row = input_dataframe.iloc[i]
            predicted_value = predict_bike_response[i]
            current_stock = station_current_stock[i]
            merged_result.append({
                "station_id": row["Rental_Location_ID"],
                "predicted_rental": predicted_value,
                "stock" : current_stock['stock']
            })

        return merged_result
    
    # 2. 각 station 별 예상 수요(def predict_bike_demand()의 predictions)
    @app.route('/show', methods=['GET']) 
    def find_abundant_deficient_stations():
        merged_result = prediction.merge_input_prediction_stock()

        station_demands = []
        for i in range(len(merged_result)):
            row = merged_result[i]
            predicted_rental = ceil(row['predicted_rental'])
            stock = row['stock']
            status = ceil(stock - predicted_rental - 3) # 예측된 수요량보다 3개 더 많아야 함

            if status <= 0:
                station_demands.append({
                    "station_id": row['station_id'],
                    "stock": stock,
                    "predicted_rental": predicted_rental,
                    "status": "deficient"
                })
            else:
                station_demands.append({
                    "station_id": row['station_id'],
                    "stock": stock,
                    "predicted_rental": predicted_rental,
                    "status": "abundant"
                })
        return station_demands

zone1_id_list =[
'ST-786',
'ST-3108',
'ST-963',
'ST-953',
'ST-1568',
'ST-2682',
'ST-2882',
'ST-789',
'ST-962',
'ST-784',
'ST-1366',
'ST-3164',
'ST-961',
'ST-1883',
'ST-1246',
'ST-3208']

class make_route:
    @app.route('/distance', methods=['GET']) 
    def get_zone_distance():
        # 관리권역1 파일 불러오기 & 인덱스 설정
        zone_df = pd.read_csv("./data/zone1_distance.csv")
        zone_df = zone_df.set_index("Rental_location_ID")

        station_demand = prediction.find_abundant_deficient_stations()
        station_status_dict = {}

        for station in station_demand:
            station_check = station["station_id"]
            if station_check in zone1_id_list:
                if station["status"] == "deficient":
                    station_status_dict[station["station_id"]] = {
                        "status": "deficient",
                        "stock": station['stock'],
                        "predicted_rental": station['predicted_rental'],
                    }
                if station["status"] == "abundant":
                    if station['stock'] >= 0:
                        station_status_dict[station["station_id"]] = {
                            "status": "abundant",
                            "stock": station['stock'],
                            "predicted_rental": station['predicted_rental']
                        }
                    else:
                        station_status_dict[station["station_id"]] = {
                            "status": "abundant",
                            "stock": station['stock'],
                            "predicted_rental": station['predicted_rental']
                        }
        return station_status_dict
    
    @app.route('/supplydemand', methods=['GET']) 
    def supply_demand():
        # station_status_dict = make_route.get_zone_distance()
        station_demand = prediction.find_abundant_deficient_stations()
        supply_demand = []  # 결과를 저장할 리스트

        for station in station_demand:
            station_check = station["station_id"]
            if station_check in zone1_id_list:
                if station["status"] == "deficient":
                    supply_demand.append(-station["predicted_rental"]-5)  # 음수 처리
                if station["status"] == "abundant":
                    supply_demand.append(5)  # 양수 처리

        supply_demand.append((-1) * sum(supply_demand))
        print("supply_demand", supply_demand)
        return supply_demand

# zone1 station names
station_names = {
    0: "ST-786",
    1: "ST-3108",
    2: "ST-963",
    3: "ST-953",
    4: "ST-1568",
    5: "ST-2682",
    6: "ST-2882",
    7: "ST-789",
    8: "ST-962",
    9: "ST-784",
    10: "ST-1366",
    11: "ST-3164",
    12: "ST-961",
    13: "ST-1883",
    14: "ST-1246",
    15: "ST-3208",
    16: "center",
}

@app.route('/route', methods=['GET']) 
def find_route_order():
    # num_stations
    zone_df = pd.read_csv("./data/zone1_distance.csv")
    zone_df = zone_df.set_index("Rental_location_ID")
    # supply
    supply_demand = make_route.supply_demand()
    # cost
    distance_list = zone_df.values.tolist()

    # 데이터 정의
    num_stations = len(zone_df) - 1  # 16개 = 정비센터 1개, 경유지 15개
    supply = supply_demand  # 오전 6시
    cost = distance_list

    # 문제 정의
    problem = pulp.LpProblem("Bike_Redistribution", pulp.LpMinimize)

    # 변수 정의: x_ij는 i에서 j로 이동하는 자전거 수
    x = pulp.LpVariable.dicts("x", ((i, j) for i in range(num_stations) for j in range(num_stations)),
                              lowBound=0, cat="Integer")

    # 목표 함수: 총 이동 비용 최소화
    problem += pulp.lpSum(cost[i][j] * x[i, j] for i in range(num_stations) for j in range(num_stations))

    # 제약 조건
    # 특정 스테이션에서만 자전거 출발 가능하도록 설정
    start_station = len(zone_df) - 1  # center
    for i in range(num_stations):
        if i != start_station:
            # 다른 스테이션에서 자전거가 나갈 수 없도록 설정
            problem += pulp.lpSum(x[i, j] for j in range(num_stations) if i != j) == 0

    # 여유 대여소에서 자전거 이동량 제한
    for i in range(num_stations):
        if supply[i] > 0:  # 여유 대여소
            problem += pulp.lpSum(x[i, j] for j in range(num_stations) if i != j) <= supply[i]

    # 부족 대여소의 수요 충족
    for j in range(num_stations):
        if supply[j] < 0:  # 부족 대여소
            problem += pulp.lpSum(x[i, j] for i in range(num_stations) if i != j) >= -supply[j]

    # 문제 해결
    problem.solve()

    station_status_dict = make_route.get_zone_distance()
    results_list = []

    for i in range(num_stations):
        for j in range(num_stations):
            # None이 아닌 경우에만 비교
            if x[i, j].varValue is not None and x[i, j].varValue > 0:
                from_name =station_names[i]
                to_name = station_names[j]
                cur_station_dict = station_status_dict.get(to_name)
                predicted_rental = station_status_dict.get(to_name)["predicted_rental"]
                if not cur_station_dict:
                    print(to_name, "no information!")
                else:
                    stock = cur_station_dict["stock"]
                    results_list.append({
                        "from_station": from_name,
                        "predicted_rental": predicted_rental,
                        "from_index": i,
                        "to_station": to_name,
                        "to_index": j,
                        "bikes_moved": x[i, j].varValue,
                        "stock": stock  # 가져온 stock 값
                    })

    return results_list


if __name__ == "__main__":

    # Flask 애플리케이션 실행
    app.run(debug=True)