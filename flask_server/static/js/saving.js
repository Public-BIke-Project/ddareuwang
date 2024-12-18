var map;
var marker_s, marker_e;
var resultMarkerArr = [];
var resultInfoArr = [];
var bikeMarkers = []; // 따릉이
let animationMarker; // 애니메이션 마커를 저장할 변수
let animationInterval; // 애니메이션의 타이머

// 사용자 지정 아이콘 생성 함수
function createCustomIcon(label) {
    const canvas = document.createElement('canvas');
    canvas.width = 50;
    canvas.height = 57;
    const ctx = canvas.getContext('2d');

    ctx.beginPath();
    ctx.moveTo(25, 60); // 아래쪽 중심점
    ctx.lineTo(45, 15); // 오른쪽 위
    ctx.lineTo(5, 15); // 왼쪽 위
    ctx.closePath();
    ctx.fillStyle = '#0066ff'; // 삼각형 색상
    ctx.fill();

    // 텍스트 그리기
    ctx.fillStyle = 'black';
    ctx.font = 'bold 20px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, 25, 30);

    return canvas.toDataURL();
}

// Tmap 지도 초기화 함수
function initTmap() {
    resultMarkerArr = [];

    // 지도 초기화
    map = new Tmapv2.Map("map_div", {
        center: new Tmapv2.LatLng(37.501228,	127.050362),
        width: "50%",
        height: "700px",
        zoom: 14,
        zoomControl: true,
        scrollwheel: true
    });

    // 시작과 도착 마커 추가
    marker_s = new Tmapv2.Marker({
        position: new Tmapv2.LatLng(37.4957886, 127.0717955),
        icon: createCustomIcon("S"), // 사용자 지정 아이콘
        iconSize: new Tmapv2.Size(50, 57),
        zIndex: 1000,
        title: '출발',
        visible: false,
        map: map
    });
    resultMarkerArr.push(marker_s);

    marker_e = new Tmapv2.Marker({
        position: new Tmapv2.LatLng(37.4957886, 127.0717950),
        icon: createCustomIcon("E"), // 사용자 지정 아이콘
        iconSize: new Tmapv2.Size(50, 57),
        zIndex: 1000,
        title: '도착',
        visible: false,
        map: map
    });
    resultMarkerArr.push(marker_e);

    // 경유지 마커 추가
    addWaypointMarkers();

    // 버튼 이벤트 설정
    setupButtons();
}

// 경유지 마커 추가 함수
function addWaypointMarkers() {
    const waypoints = [
        { lat: 37.516811, lng: 127.040474, label: "1", title:'ST-963'}, //ST-963 37.516811	127.040474
        { lat: 37.512810, lng: 127.026367, label: "2", title:'ST-3208'}, //ST-3208 37.512810	127.026367
        { lat: 37.517590, lng: 127.035027, label: "3", title:'ST-962'}, //ST-962 37.517590	127.035027  
        { lat: 37.518639, lng: 127.035400, label: "4", title:'ST-961'}, //ST-961 37.518639	127.035400
        { lat: 37.515888, lng: 127.066200, label: "5", title:'ST-784'}, //ST-784 37.515888	127.066200
        { lat: 37.517773, lng: 127.043022, label: "6", title:'ST-786'}, //ST-786 37.517773	127.043022
        { lat: 37.509586, lng: 127.040909, label: "7", title:'ST-1366'}, //ST-1366 37.509586	127.040909
        { lat: 37.509785, lng: 127.042770, label: "8", title:'ST-2882'}, //ST-2882 37.509785	127.042770
        { lat: 37.506367, lng: 127.034523, label: "9", title:'ST-1246'}, //ST-1246 37.506367	127.034523
        { lat: 37.505703, lng: 127.029198, label: "10", title:'ST-3108'}, //ST-3108 37.505703	127.029198
         
    ];
    // { lat: 37.519787, lng: 127.056763, label: "2", title:'ST-953'}, //ST-953 37.519787	127.056763
    // { lat: 37.519257, lng: 127.051598, label: "3", title:'ST-2682'}, //ST-2682 	37.519257	127.051598
    // { lat: 37.511627, lng: 127.035652, label: "5", title:'ST-789'}, //ST-789 37.511627	127.035652
    // { lat: 37.509586, lng: 127.040909, label: "6", title:'ST-1366'}, //ST-1366 37.509586	127.040909
    // { lat: 37.519737, lng: 127.057678, label: "4", title:'ST-3164'}, //ST-3164 37.519737	127.057678
    // { lat: 37.507999, lng: 127.045250, label: "6", title:'ST-1883'}, //ST-1883 37.507999	127.045250

    waypoints.forEach((waypoint) => {
        const marker = new Tmapv2.Marker({
            position: new Tmapv2.LatLng(waypoint.lat, waypoint.lng),
            icon: createCustomIcon(waypoint.label, waypoint.color), // 사용자 지정 아이콘
            iconSize: new Tmapv2.Size(50, 57),
            zIndex: 1000,
            title: waypoint.title,
            label: waypoint.label,
            visible: false,
            map: map
        });
        resultMarkerArr.push(marker);
    });
}
// function updateMarkersFromFlask() {
//     fetch('/show')
//         .then(response => response.json())
//         .then(stations => {
//             // Flask에서 가져온 데이터를 로그로 확인
//             console.log("Flask에서 반환된 데이터:", stations);

//             // 기존 마커를 Flask 데이터와 매칭
//             resultMarkerArr.forEach(marker => {
//                 const markerId = marker.title; // 마커의 title 속성에서 station_id 가져오기
//                 console.log("마커 title 확인:", markerId); // title 값을 로그로 출력

//                 // Flask 데이터에서 station_id가 일치하는 항목 찾기
//                 const stationData = stations.find(station => station.station_id === markerId);

//                 if (stationData) {
//                     const { station_id, stock, supply_demand } = stationData;

//                     // 마커의 타이틀 업데이트
//                     marker.title = `${station_id}<br>현 재고: ${stock}<br>필요 재고: ${supply_demand}</br>`;

//                     // 마커 클릭 이벤트 추가
//                     marker.addListener("click", function () {
//                         alert(waypoint.title);
//                     });

//                     console.log(`마커 ${station_id}의 타이틀이 업데이트되었습니다.`);
//                 } else {
//                     console.warn(`Flask 데이터에서 ${markerId}에 해당하는 정보가 없습니다.`);
//                 }
//             });
//         })
//         .catch(error => {
//             console.error("Flask에서 데이터를 가져오는 중 오류 발생:", error);
//         });
// }

// 루트 생성 함수
function generateRoute() {
    // 마커를 표시
    resultMarkerArr.forEach(marker => marker.setVisible(true));

    const headers = {
        appKey: "6ockLdPQfZatxQctpKLtn5Lg7B5kDO555UzRAx0B",
        "Content-Type": "application/json"
    };

    const param = JSON.stringify({
        startName: "출발지",
        startX: "127.0717955",
        startY: "37.4957886",
        startTime: "202411200600",
        endName: "도착지",
        endX: "127.0717955",
        endY: "37.4957886",
        viaPoints: [
            { viaPointId: "test01", viaPointName: "ST-963", viaX: "127.040474", viaY: "37.516811" ,viaTime : 900},
            { viaPointId: "test07", viaPointName: "ST-3208", viaX: "127.026367", viaY: "37.512810",viaTime : 900}, 	
            { viaPointId: "test02", viaPointName: "ST-962", viaX: "127.056763", viaY: "37.519787",viaTime : 900},
            { viaPointId: "test07", viaPointName: "ST-961", viaX: "127.035400", viaY: "37.518639",viaTime : 900},
            { viaPointId: "test05", viaPointName: "ST-784", viaX: "127.066200", viaY: "37.515888" ,viaTime : 900},
            { viaPointId: "test08", viaPointName: "ST-786", viaX: "127.043022", viaY: "37.517773" ,viaTime : 900},
            { viaPointId: "test07", viaPointName: "ST-1366", viaX: "127.040909", viaY: "37.509586",viaTime : 900},
            { viaPointId: "test07", viaPointName: "ST-2882", viaX: "127.042770", viaY: "37.509785",viaTime : 900},
            { viaPointId: "test08", viaPointName: "ST-1246", viaX: "127.034523", viaY: "37.506367" ,viaTime : 900},
            { viaPointId: "test07", viaPointName: "ST-3108", viaX: "127.029198", viaY: "37.505703",viaTime : 900},
            
            //ST-784 37.515888	127.066200
            //ST-1366 37.509586	127.040909
            //ST-2882 37.509785	127.042770
             // { viaPointId: "test07", viaPointName: "ST-3164", viaX: "127.057678", viaY: "37.519737",viaTime : 900}, 	 
            // { viaPointId: "test07", viaPointName: "ST-1883", viaX: "127.045250", viaY: "37.507999",viaTime : 900},
            // { viaPointId: "test05", viaPointName: "ST-789", viaX: "127.035652", viaY: "37.511627" ,viaTime : 900}
           
        ],
        reqCoordType: "WGS84GEO",
        resCoordType: "EPSG3857",
        searchOption: "2" // 고정: 교통최적+최소시간
    });
   

    $.ajax({
        method: "POST",
        url: "https://apis.openapi.sk.com/tmap/routes/routeSequential30?version=1&format=json",
        headers: headers,
        async: false,
        data: param,
        success: function (response) {
            console.log("API 응답 데이터:", response); // API 응답 데이터 출력
            displayRoute(response);
        },
        error: function (request, status, error) {
            console.error(`Error: ${request.status}, ${request.responseText}, ${error}`);
        }
    });
}
// 시간 포맷 함수
function formatTime(rawTime) {
    if (!rawTime || rawTime.length !== 14) return "정보 없음"; // 시간 데이터가 없거나 길이가 다르면 처리
    const hour = rawTime.slice(8, 10); // 시간 추출
    const minute = rawTime.slice(10, 12); // 분 추출
    return `${hour}:${minute}`; // HH:mm 형식으로 반환
}
function displayRoute(response) {
    const resultFeatures = response.features; // 경로 및 포인트 정보
    let routeCoordinates = []; // 경로 좌표 배열 (애니메이션용)

    // 테이블의 <table> 요소를 선택
    const table = document.querySelector("#schedule_div table");

    // 기존 데이터를 모두 초기화 (헤더 제외)
    table.innerHTML = `
        <tr>
            <th>시간</th>
            <th>위치</th>
            <th>작업</th>
        </tr>
    `;

    // Flask에서 스테이션 데이터 가져오기
    fetch('/route')
        .then(response => response.json())
        .then(moves => {
            console.log("Flask에서 가져온 스테이션 데이터:", moves);

            // 이미 테이블에 추가된 대여소 정보를 추적하기 위한 Set 객체
            const processedStationsForTable = new Set();

            // 경로 데이터의 총 거리와 시간
            const resultData = response.properties;
            const tDistance = `${(resultData.totalDistance / 1000).toFixed(1)} km`;
            
           
            const travelTime = resultData.totalTime / 60; // 분 단위

            // 경유지 체류 시간 추가 (8곳 × 15분 = 120분)
            const stayTime = 8 * 15; // 경유지 체류 시간 (분 단위)

            // 총 시간 (이동 시간 + 체류 시간)
            const totalTimeWithStay = travelTime + stayTime;

            // 시간 표시
            const tTime = `${Math.floor(totalTimeWithStay)} 분`;

            // 총 요약 행 추가
            const summaryRow = document.createElement("tr");
            summaryRow.innerHTML = `
                <td>-</td>
                <td>경로 요약</td>
                <td>총 거리: ${tDistance}, 총 시간: ${tTime}</td>
            `;
            table.appendChild(summaryRow);

            // 출발지 정보 추가
            const startFeature = resultFeatures[0]; // 첫 번째 경유지 = 출발지
            if (startFeature && startFeature.properties) {
                const arriveTime = formatTime(startFeature.properties.arriveTime) || "-";
                const startRow = document.createElement("tr");
                startRow.innerHTML = `
                    <td>${arriveTime}</td>
                    <td>배송센터 출발</td>
                    <td>자전거 15대</td>
                `;
                table.appendChild(startRow);
            }

            // 배차 정보를 순회하며 추가
            resultFeatures.forEach((feature, idx) => {
                const properties = feature.properties;
                console.log(`Properties ${idx}:`, properties); // properties 데이터

                // 경유지 데이터 처리
                if (properties && properties.pointType && properties.pointType.startsWith("B")) {
                    const uniqueKey = `${properties.index}-${properties.viaPointName}`;

                    if (!processedStationsForTable.has(uniqueKey)) {
                        processedStationsForTable.add(uniqueKey); // 중복 확인용 Set에 추가

                        const arriveTime = formatTime(properties.arriveTime) || "정보 없음";
                        const completeTime = formatTime(properties.completeTime) || "정보 없음";
                        const viaPointName = `${properties.viaPointName.replace(/^\[\d+\]\s*/, "")}`;
                        const detailInfo = `다음 대여소 까지: ${(properties.distance / 1000).toFixed(1)}km`;


                        // Flask 데이터에서 스테이션 정보 매핑
                        const stationData = moves.find(moves => {
                            const fromstation = moves.from_station 
                            const tostation = moves.to_station
                            return fromstation === properties.viaPointName.replace(/^\[\d+\]\s*/, '') || tostation === properties.viaPointName.replace(/^\[\d+\]\s*/, '');
                        });
                        

                        const stockInfo = stationData
                            ? `현 재고: ${stationData.stock}\n필요 재고: ${stationData.predicted_rental}`
                            : "";

                        // 배치 시작 작업 추가
                        const arriveRow = document.createElement("tr");
                        arriveRow.innerHTML = `
                            <td>${arriveTime}\n~\n${completeTime}</td>
                            <td>${viaPointName}</td>
                            <td>${stockInfo}</br>${detailInfo}</td>
                        `;
                        table.appendChild(arriveRow);

                    }
                }

                // 지도에 경로 그리기
                if (feature.geometry.type === "LineString") {
                    const drawInfoArr = feature.geometry.coordinates.map(coord => {
                        const point = new Tmapv2.Point(coord[0], coord[1]);
                        const converted = new Tmapv2.Projection.convertEPSG3857ToWGS84GEO(point);
                        return new Tmapv2.LatLng(converted._lat, converted._lng);
                    });

                        const polyline = new Tmapv2.Polyline({
                            path: drawInfoArr,
                            strokeColor: "#0099FF",
                            strokeWeight: 6,
                            map: map
                        });

                        resultInfoArr.push(polyline);
                    
                

                        // 경로 좌표 추가 (애니메이션용)
                        routeCoordinates = routeCoordinates.concat(drawInfoArr);
                    }
                });

                // 도착지 정보 추가
                const endFeature = resultFeatures[resultFeatures.length - 1]; // 마지막 경유지 = 도착지
                if (endFeature && endFeature.properties) {
                    const completeTime = formatTime(endFeature.properties.completeTime) || "-";
                    const endRow = document.createElement("tr");
                    endRow.innerHTML = `
                        <td>${completeTime}</td>
                        <td>배송센터 도착</td>
                        <td>-</td>
                    `;
                    table.appendChild(endRow);
                }
                // 경로 애니메이션 시작
                startRouteAnimation(routeCoordinates);
            })
            .catch(error => {
            console.error("Flask에서 데이터를 가져오는 중 오류 발생:", error);
            });
}

// 경로 애니메이션 함수
function startRouteAnimation(routeCoordinates) {
    if (routeCoordinates.length === 0) {
        console.error("경로 좌표가 없습니다.");
        return;
    }

    // 애니메이션 마커 생성
    const animationMarker = new Tmapv2.Marker({
        position: routeCoordinates[0], // 경로의 시작점
        icon: "https://maps.google.com/mapfiles/kml/shapes/cabs.png", // 애니메이션 마커 아이콘
        map: map
    });

    let index = 0; // 현재 위치 인덱스
    const animationSpeed = 100; // 애니메이션 속도 (ms)

    const interval = setInterval(() => {
        if (index < routeCoordinates.length - 1) {
            index++;
            animationMarker.setPosition(routeCoordinates[index]); // 마커 위치 업데이트
        } else {
            clearInterval(interval); // 경로 끝에 도달하면 애니메이션 종료
            console.log("애니메이션 완료");
        }
    }, animationSpeed);
}

// 지도 초기화 함수
function resetMap() {
    if (resultInfoArr.length > 0) {
        resultInfoArr.forEach(info => info.setMap(null));
        resultInfoArr = [];
    }

    if (resultMarkerArr.length > 0) {
        resultMarkerArr.forEach(marker => marker.setMap(null));
        resultMarkerArr = [];
    }

    console.log("지도 초기화 완료");
}

// 따릉이 대여소 정보를 지도에 표시하는 함수
function displayBikeStations() {
    // 텍스트 파일 읽기
    fetch('/static/station_1_list.txt')
        .then(response => response.text())
        .then(text => {
            // 텍스트 파일을 줄 단위로 배열로 변환
            const allowedStations = text.split('\n').map(id => id.trim());

            // 따릉이 대여소 정보 가져오기
            // 두 개의 따릉이 대여소 정보 가져오기
            Promise.all([
                fetch('http://openapi.seoul.go.kr:8088/787a4c4f41736d6133365464694c56/json/bikeList/1001/2000/').then(response => response.json()),
                fetch('http://openapi.seoul.go.kr:8088/787a4c4f41736d6133365464694c56/json/bikeList/2001/3000/').then(response => response.json())
            ])
            .then(results => {
                // 두 개의 결과 데이터를 합침
                const bikeList = [...results[0].rentBikeStatus.row, ...results[1].rentBikeStatus.row];

                    // 텍스트 파일에 포함된 대여소만 필터링
                    const filteredStations = bikeList.filter(station =>
                        allowedStations.includes(station.stationId)
                    );

                    // 필터된 대여소 마커 추가
                    filteredStations.forEach(station => {
                        var marker = new Tmapv2.Marker({
                            position: new Tmapv2.LatLng(station.stationLatitude, station.stationLongitude),
                            map: map,
                            zIndex: 500,
                            icon: "./static/images/icon/bike_icon.png", 
                            title: `${station.stationId}\n남은 자전거: ${station.parkingBikeTotCnt} / ${station.rackTotCnt}`
                        });

                        // 마커 클릭 이벤트
                        marker.addListener("click", function () {
                            alert(`대여소 이름: ${station.stationId}\n남은 자전거: ${station.parkingBikeTotCnt} / ${station.rackTotCnt}`);
                        });

                        bikeMarkers.push(marker);
                    });
                })
                .catch(error => {
                    console.error("대여소 정보를 가져오는 데 실패했습니다.", error);
                });
        })
        .catch(error => {
            console.error("텍스트 파일을 읽는 중 오류가 발생했습니다:", error);
        });
}

// 버튼 이벤트 설정
function setupButtons() {
    document.getElementById("create_traffic_route_btn").addEventListener("click", function () {
        console.log("루트 생성 버튼 클릭됨");
        generateRoute();
    });

    document.getElementById("reset_map_btn").addEventListener("click", function () {
        console.log("지도 초기화 버튼 클릭됨");
        resetMap();
    });
}

// DOMContentLoaded 이벤트 리스너 등록
document.addEventListener("DOMContentLoaded", function () {
    console.log("페이지 로드 완료. Tmap 초기화 실행");
    initTmap();
    // 따릉이 대여소 표시
    displayBikeStations();
    // Flask 데이터를 기반으로 마커 타이틀 업데이트
    // updateMarkersFromFlask();
});





