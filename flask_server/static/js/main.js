var map;
var marker_s, marker_e;
var resultMarkerArr = [];
var resultInfoArr = [];
var bikeMarkers = []; // 따릉이
let animationMarker; // 애니메이션 마커를 저장할 변수
let animationInterval; // 애니메이션의 타이머
const waypointVisits = {};

function createCustomIcon(labels, isRevisited) {
    const canvas = document.createElement('canvas');
    canvas.width = 50;
    canvas.height = 100;  // 높이를 더 크게 설정하여 모래시계 형태를 표현

    const ctx = canvas.getContext('2d');

    // 중심점 위치를 기준으로 삼각형을 그림
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;

    if (isRevisited) {
        // 역삼각 위쪽 삼각형
        ctx.beginPath();
        ctx.moveTo(centerX, centerY); // 꼭지점 - 위쪽 중앙 (25,50)
        ctx.lineTo(centerX + 25, centerY + 50); // 오른쪽 아래
        ctx.lineTo(centerX - 25, centerY + 50 ); // 왼쪽 아래
        ctx.closePath();
        ctx.fillStyle = '#0066ff'; // 색상 설정
        ctx.fill();

        // 정삼각 아래쪽 삼각형
        ctx.beginPath();
        ctx.moveTo(centerX, centerY); // 아래쪽 중심점
        ctx.lineTo(centerX + 25, centerY - 50); // 오른쪽 위
        ctx.lineTo(centerX - 25, centerY - 50 ); // 왼쪽 위
        ctx.closePath();
        ctx.fillStyle = '#ff0000'; // 다른 색상으로 설정
        ctx.fill();
    } else {
        // 기존 역삼각형 그리기 (단일 방문 시)
        ctx.beginPath();
        ctx.moveTo(centerX, centerY); // 아래쪽 중심점 (25,50)
        ctx.lineTo(centerX + 25, centerY - 50); // 오른쪽 위
        ctx.lineTo(centerX - 25, centerY - 50); // 왼쪽 위
        ctx.closePath();
        ctx.fillStyle = '#3281a8'; // 색상 설정
        ctx.fill();
    }


    // if (isRevisited) {
    //     // 정삼각형 그리기 - 위쪽 삼각형
    //     ctx.beginPath();
    //     ctx.moveTo(25, 50); // 꼭지점 - 위쪽 중앙
    //     ctx.lineTo(50, 100); // 오른쪽 아래
    //     ctx.lineTo(0, 100); // 왼쪽 아래
    //     ctx.closePath();
    //     ctx.fillStyle = '#0066ff'; // 파랑
    //     ctx.fill();

    //     // 역삼각형 그리기 - 아래쪽 삼각형
    //     ctx.beginPath();
    //     ctx.moveTo(25, 50); // 아래쪽 중심점
    //     ctx.lineTo(50, 0); // 오른쪽 위
    //     ctx.lineTo(0, 0); // 왼쪽 위
    //     ctx.closePath();
    //     ctx.fillStyle = '#ff0000'; // 빨강
    //     ctx.fill();
    // } else {
    //     // 기존 역삼각형 그리기 (단일 방문 시)
    //     ctx.beginPath();
    //     ctx.moveTo(25, 60); // 아래쪽 중심점
    //     ctx.lineTo(45, 15); // 오른쪽 위
    //     ctx.lineTo(5, 15); // 왼쪽 위
    //     ctx.closePath();
    //     ctx.fillStyle = '#3281a8'; // 색상 설정
    //     ctx.fill();
    // }

    // 텍스트 설정
    ctx.fillStyle = 'black';
    ctx.font = 'bold 12px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    // 방문 순서 표시
    if (Array.isArray(labels)) {
        // 여러 레이블 표시
        labels.forEach((label, index) => {
            const yOffset = centerY + 30 + index * 30; // 레이블 간격 조정 (세로로 나열)
            ctx.fillText(label, centerX, yOffset);
        });
    } else {
        // 단일 레이블 표시
        ctx.fillText(labels, centerX, centerY-30);
    }

    return canvas.toDataURL();
}


// Tmap 지도 초기화 함수
function initTmap() {
    resultMarkerArr = [];

    // 지도 초기화
    map = new Tmapv3.Map("map_div", {
        center: new Tmapv3.LatLng(37.501228,127.050362),
        width: "50%",
        height: "700px",
        zoom: 14,
        zoomControl: true,
        scrollwheel: true
    });
    // 시작과 도착 마커 추가
    marker_s = new Tmapv3.Marker({
        position: new Tmapv3.LatLng(37.4957886, 127.0717955),
        icon: createCustomIcon("S"), // 사용자 지정 아이콘
        iconSize: new Tmapv3.Size(50, 57),
        zIndex: 1000,
        title: '출발',
        visible: false,
        map: map
    });
    resultMarkerArr.push(marker_s);

    marker_e = new Tmapv3.Marker({
        position: new Tmapv3.LatLng(37.4957886, 127.0717950),
        icon: createCustomIcon("E"), // 사용자 지정 아이콘
        iconSize: new Tmapv3.Size(50, 57),
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
        { lat: 37.516811, lng: 127.040474, label: "1", title:'ST-963', visitCount: 1}, //ST-963 37.516811	127.040474
        { lat: 37.512810, lng: 127.026367, label: "2", title:'ST-3208',visitCount: 1}, //ST-3208 37.512810	127.026367
        { lat: 37.516811, lng: 127.040474, label: "3", title:'ST-963',visitCount: 2}, //ST-962 37.517590	127.035027  
        { lat: 37.518639, lng: 127.035400, label: "4", title:'ST-961',visitCount: 1}, //ST-961 37.518639	127.035400
        { lat: 37.515888, lng: 127.066200, label: "5", title:'ST-784',visitCount: 1}, //ST-784 37.515888	127.066200
        { lat: 37.517773, lng: 127.043022, label: "6", title:'ST-786',visitCount: 2}, //ST-786 37.517773	127.043022
        { lat: 37.509586, lng: 127.040909, label: "7", title:'ST-1366',visitCount: 1}, //ST-1366 37.509586	127.040909
        { lat: 37.509785, lng: 127.042770, label: "8", title:'ST-2882',visitCount: 1}, //ST-2882 37.509785	127.042770
        { lat: 37.506367, lng: 127.034523, label: "9", title:'ST-1246',visitCount: 1}, //ST-1246 37.506367	127.034523
        { lat: 37.505703, lng: 127.029198, label: "10", title:'ST-3108',visitCount: 1}, //ST-3108 37.505703	127.029198
         
    ];
    // 각 대여소별 방문 순서 추적
    waypoints.forEach((waypoint) => {
        if (!waypointVisits[waypoint.title]) {
            waypointVisits[waypoint.title] = [];
        }
        waypointVisits[waypoint.title].push(waypoint.label);
    });
    // 마커 추가 (방문 순서 레이블을 표시)
    waypoints.forEach((waypoint) => {
        const labels = waypointVisits[waypoint.title];
        const isRevisited = labels.length > 1;

        // createCustomIcon 함수 호출 시 올바른 labels 배열과 isRevisited 값 전달
        const marker = new Tmapv3.Marker({
            position: new Tmapv3.LatLng(waypoint.lat, waypoint.lng),
            icon: createCustomIcon(labels, isRevisited), // 사용자 지정 아이콘 (다중 방문 레이블 포함)
            iconSize: new Tmapv3.Size(50, 100), // 모래시계 형태이므로 크기 조정
            anchor: 'center', // 앵커 설정 (중앙 하단)
            zIndex: 1000,
            title: waypoint.title,
            visible: false, // 마커를 보이도록 설정
            map: map
        });
        resultMarkerArr.push(marker);
    });
}



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
        startTime: "202411202000",
        endName: "도착지",
        endX: "127.0717955",
        endY: "37.4957886",
        viaPoints: [
            { viaPointId: "test01", viaPointName: "ST-963", viaX: "127.040474", viaY: "37.516811" ,viaTime : 900},
            { viaPointId: "test02", viaPointName: "ST-953", viaX: "127.056763", viaY: "37.519787",viaTime : 900},
            { viaPointId: "test03", viaPointName: "ST-2682", viaX: "127.051598", viaY: "37.519257" ,viaTime : 900},
            { viaPointId: "test04", viaPointName: "ST-2882", viaX: "127.042770", viaY: "37.509785" ,viaTime : 900},
            { viaPointId: "test05", viaPointName: "ST-789", viaX: "127.035652", viaY: "37.511627" ,viaTime : 900}, 
            { viaPointId: "test06", viaPointName: "ST-1366", viaX: "127.040909", viaY: "37.509586" ,viaTime : 900},
            { viaPointId: "test07", viaPointName: "ST-961", viaX: "127.035400", viaY: "37.518639",viaTime : 900},
            { viaPointId: "test08", viaPointName: "ST-1246", viaX: "127.034523", viaY: "37.506367" ,viaTime : 900}
           
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
    fetch('/show')
        .then(response => response.json())
        .then(stations => {
            console.log("Flask에서 가져온 스테이션 데이터:", stations);

            // 이미 테이블에 추가된 대여소 정보를 추적하기 위한 Set 객체
            const processedStationsForTable = new Set();

            // 경로 데이터의 총 거리와 시간
            const resultData = response.properties;
            const tDistance = `${(resultData.totalDistance / 1000).toFixed(1)} km`;
            const tTime = `${(resultData.totalTime / 60).toFixed(0)} 분`;

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
                        const stationData = stations.find(
                            station => station.station_id === properties.viaPointName.replace(/^\[\d+\]\s*/, "")
                        );
                        const stockInfo = stationData
                            ? `현 재고: ${stationData.stock}\n필요 재고: ${stationData.supply_demand}`
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
                        const point = new Tmapv3.Point(coord[0], coord[1]);
                        const converted = new Tmapv3.Projection.convertEPSG3857ToWGS84GEO(point);
                        return new Tmapv3.LatLng(converted._lat, converted._lng);
                    });

                    const polyline = new Tmapv3.Polyline({
                        path: drawInfoArr,
                        strokeColor: "#0099FF",
                        strokeWeight: 6,
                        map: map
                    });

                    resultInfoArr.push(polyline);
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
        })
        .catch(error => {
            console.error("Flask에서 데이터를 가져오는 중 오류 발생:", error);
        });
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
                        var marker = new Tmapv3.Marker({
                            position: new Tmapv3.LatLng(station.stationLatitude, station.stationLongitude),
                            map: map,
                            zIndex: 500,
                            icon: "./static/images/icon/bike_icon.png", // 기본 파란색 아이콘
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

document.addEventListener("DOMContentLoaded", function () {
    console.log("페이지 로드 완료. Tmap 초기화 실행");
    initTmap();
    // 따릉이 대여소 표시
    displayBikeStations();
});