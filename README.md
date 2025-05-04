![Image](https://github.com/user-attachments/assets/4263fc23-10c8-4f5f-bbba-3225c386f7bb)

# 🚲 따르왕 | 서울시 공공자전거 재배치 경로 및 스케줄 최적화 시스템
서울시 공공자전거 따릉이의 자전거 재고 불균형 문제를 해결하기 위한 수요 기반 재배치 경로 최적화 서비스입니다. </br>
배송 인력에게 시간대별 권역 내 최적 동선을 안내하여 재배치 효율성과 따릉이 이용 품질 향상을 목표로 합니다.

## 📺 [서비스 데모 영상](https://drive.google.com/drive/u/0/folders/14jFlKT7UzQliQTWusFUJ5ij-KbwD2vYE)

## 🧠 핵심 기술 스택 및 기능 구성

### ⛏ Backend
담당자: [👾YUNA AN](https://github.com/pompom33) & [👾NURI PARK](https://github.com/Hello-Nuri) </br>
언어: Python [📁app.py](https://github.com/Public-BIke-Project/ddareuwang/blob/main/flask_server/app.py) </br>
주요 기능:  </br>
* **데이터 연동**: Google BigQuery와 연동하여 2023년 자전거 대여 가능 대수 DB 활용
* **수요 예측 모듈 연동**: 각 대여소의 미래 수요량을 예측하는 MachineLearning 모델을 로딩하여, 사용자 입력 데이터를 바탕으로 향후 수요량을 계산
* **경로 최적화 로직 구현**: 완화된 TSP 문제를 정수 선형 계획법으로 수식화하고, 배송 직원 동선을 최적화
* **API 설계 및 반환**: 사용자 입력(날짜, 시간대, 권역)을 바탕으로 최적 경로 데이터를 JSON 형태로 API 응답

### 🌐 Frontend
담당자: [👾NURI PARK](https://github.com/Hello-Nuri)</br>
언어: [📁JavaScript](https://github.com/Public-BIke-Project/ddareuwang/tree/main/flask_server/static/js), [📁HTML](https://github.com/Public-BIke-Project/ddareuwang/tree/main/flask_server/templates), [📁CSS](https://github.com/Public-BIke-Project/ddareuwang/blob/main/flask_server/static/css/style.css), </br>
주요 기능:</br>
* 사용자로부터 권역, 날짜, 시간대 입력값을 수집하고 서버에 전달
* 서버로부터 응답받은 최적 경로 데이터를 시각적으로 표시
* Tmap API를 활용하여 경로 가시화 및 사용자 편의성 증대


