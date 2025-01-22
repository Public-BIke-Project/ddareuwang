var map;
var marker_s, marker_e, waypoint;
var resultMarkerArr = [];
var bikeMarkers = []; // 따릉이
const waypointVisits = {}; // 중복방문
var waypoints = []; // api 마커와 루트 경유지
var resultInfoArr = []; // 루트그리기
var animationMarker; // 애니메이션 마커를 저장할 변수
var animationInterval; // 애니메이션의 타이머


function createCustomIcon(labels, isRevisited) {
    const canvas = document.createElement('canvas');
    canvas.width = 50;
    canvas.height = 100;  // 높이를 더 크게 설정하여 모래시계 형태를 표현

    const ctx = canvas.getContext('2d');

    // 중심점 위치를 기준으로 삼각형을 그림
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;

    if (isRevisited) {
        // 아래 삼각형
        ctx.beginPath();
        ctx.moveTo(centerX, centerY); // 꼭지점 - 위쪽 중앙 (15,60)
        ctx.lineTo(centerX + 25, centerY + 50); // 오른쪽 아래
        ctx.lineTo(centerX - 25, centerY + 50); // 왼쪽 아래
        ctx.closePath();
        ctx.fillStyle = '#6600ff'; // 다른 색상 설정
        ctx.fill();

        // 기존 삼각형
        ctx.beginPath();
        ctx.moveTo(centerX, centerY); // 아래쪽 중심점
        ctx.lineTo(centerX + 25, centerY - 50); // 오른쪽 위
        ctx.lineTo(centerX - 25, centerY - 50); // 왼쪽 위
        ctx.closePath();
        ctx.fillStyle = '#6600ff'; // 색상 설정
        ctx.fill();
    } else {
        // 기존 역삼각형 그리기 (단일 방문 시)
        ctx.beginPath();
        ctx.moveTo(centerX, centerY); // 아래쪽 중심점 (15,60)
        ctx.lineTo(centerX + 25, centerY - 50); // 오른쪽 위
        ctx.lineTo(centerX - 25, centerY - 50); // 왼쪽 위
        ctx.closePath();
        ctx.fillStyle = '#07c2db'; // 색상 설정
        ctx.fill();
    }
    // 텍스트 설정
    ctx.fillStyle = 'black';
    ctx.font = 'bold 15px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    
    if (isRevisited && labels.length > 1) {
        ctx.fillText(labels[0], centerX, centerY - 30); // 첫 번째 레이블 위쪽
        ctx.fillText(labels[1], centerX, centerY + 30); // 두 번째 레이블 아래쪽
        } 
    else {
        ctx.fillText(labels[0], centerX, centerY - 30 ); // 단일 레이블 중앙
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
        anchor : "center",
        icon: createCustomIcon("C"), // 사용자 지정 아이콘
        iconSize: new Tmapv3.Size(60, 110),
        zIndex: 1000,
        visible: false,
        map: map
    });
    resultMarkerArr.push(marker_s);

    marker_e = new Tmapv3.Marker({
        position: new Tmapv3.LatLng(37.4957886, 127.0717950),
        anchor : "center",
        icon: createCustomIcon("C"), // 사용자 지정 아이콘
        iconSize: new Tmapv3.Size(60, 110),
        zIndex: 1000,
        visible: false,
        map: map
    });
    resultMarkerArr.push(marker_e);

    // 경유지 마커 추가
    addWaypointMarkers();
    
}
// 경유지 마커 추가 함수
function addWaypointMarkers() {
    fetch('/final_output')
        .then(response => response.json())
        .then(simple_moves => {
            console.log("마커추가api 데이터:", simple_moves);

            // 필요한 데이터만 waypoints 배열에 담기
            const waypoints = simple_moves.map(move => ({
                lat: move.latitude.toString(),               // 위도
                lng: move.longitude.toString(),             // 경도
                index: move.visit_index.toString(),         // 방문 인덱스
                label: move.visit_station_name.toString()     // 대여소 ID
            }));
            // 각 대여소별 방문 순서 추적
            waypoints.forEach((waypoint) => {
                if (!waypointVisits[waypoint.label]) {
                    waypointVisits[waypoint.label] = [];
                }
                waypointVisits[waypoint.label].push(waypoint.index);
            });
            
            // 경유지에 마커 추가
            waypoints.forEach((waypoint) => {
                const labels = waypointVisits[waypoint.label];
                const isRevisited = labels.length > 1;

                const marker = new Tmapv3.Marker({
                    position: new Tmapv3.LatLng(waypoint.lat, waypoint.lng), // 중심 좌표 설정
                    anchor : "center",
                    icon: createCustomIcon(labels,isRevisited), // 사용자 지정 아이콘
                    iconSize: new Tmapv3.Size(60, 110), // 아이콘 크기 설정
                    zIndex: 1000,
                    visible : false,
                    map: map // 마커가 표시될 지도 설정
                });
            resultMarkerArr.push(marker); // 마커를 배열에 추가
        });
    })
}

// 경로 tmap 교통정보 호출 함수
function generateRoute() {
    // 마커를 표시
    resultMarkerArr.forEach(marker => marker.setVisible(true));

    fetch('/final_output')
        .then(response => response.json())
        .then(simple_moves => {
            console.log("경로정보 호출 데이터:", simple_moves);

            // API 데이터 기반으로 viaPoints 생성
            const viaPoints = simple_moves.map(move => ({
                viaPointId: move.visit_index.toString(),
                viaPointName: move.visit_station_name.toString(),
                viaX: move.longitude.toString(),
                viaY: move.latitude.toString(),
                viaTime: 900
            }));

            const headers = {
                appKey: TMAP_API_KEY
            };

            const param = JSON.stringify({
                startName: "출발지",
                startX: "127.0717955",
                startY: "37.4957886",
                startTime: "2024"+month+day+hour+"00",
                endName: "도착지",
                endX: "127.0717955",
                endY: "37.4957886",
                viaPoints: viaPoints,
                reqCoordType: "WGS84GEO",
                resCoordType: "WGS84GEO",
                searchOption: "2"
            });
            console.log("startTime:", "2024" + month + day + hour + "00");

            // Ajax 호출을 Promise로 변경
            return new Promise((resolve, reject) => {
                $.ajax({
                    method: "POST",
                    url: "https://apis.openapi.sk.com/tmap/routes/routeSequential30?version=1&format=json",
                    headers: headers,
                    async: true, // async를 true로 변경
                    contentType: "application/json",
                    data: param,
                    success: resolve,
                    error: reject
                });
            });
        })
        .then(response => {
            console.log("API 응답 데이터:", response);
            displayRoute(response);
        })
        .catch(error => {
            console.error(`Error: ${error}`);
        });
}

// // 경로 tmap 교통정보 호출 함수
// function generateRoute() {
//     // 마커를 표시
//     resultMarkerArr.forEach(marker => marker.setVisible(true));

//     fetch('/final_output') // 올바른 fetch 호출
//         .then(response => response.json()) // JSON 데이터로 변환
//         .then(simple_moves => {
//             console.log("경로정보 호출 데이터:", simple_moves);

//         // API 데이터 기반으로 viaPoints 생성
//         const viaPoints = simple_moves.map(move => ({
//             viaPointId: move.visit_index.toString(), // 방문 인덱스 (문자열로 변환)
//             viaPointName: move.visit_station_id.toString(),  // 대여소 이름
//             viaX: move.longitude.toString(),       // 경도 (문자열로 변환)
//             viaY: move.latitude.toString(),        // 위도 (문자열로 변환)
//             viaTime: 900                           // 기본 시간 설정
//         }));

//         const headers = {
//             appKey: TMAP_API_KEY
//         };
        

//         const param = JSON.stringify({
//             startName: "출발지",
//             startX: "127.0717955",
//             startY: "37.4957886",
//             startTime: "2024"+month+day+hour+"00", /// ⭐️⭐️⭐️ 사용자 입력시간 ⭐️⭐️⭐️ ///
//             endName: "도착지",
//             endX: "127.0717955",
//             endY: "37.4957886",
//             viaPoints: viaPoints,
//             reqCoordType: "WGS84GEO",
//             resCoordType: "WGS84GEO",
//             searchOption: "2" // 고정: 교통최적+최소시간
//         });
//         console.log("startTime:", "2024" + month + day + hour + "00");

//         $.ajax({
//             method: "POST",
//             url: "https://apis.openapi.sk.com/tmap/routes/routeSequential30?version=1&format=json",
//             headers: headers,
//             async: false,
//             contentType: "application/json",
//             data: param,
//             success: function (response) {
//                 console.log("API 응답 데이터:", response); // API 응답 데이터 출력
//                 displayRoute(response);
//             },
//             error: function (request, status, error) {
//                 console.error(`Error: ${request.status}, ${request.responseText}, ${error}`);
//             }
//         });
//     });
// }

// 시간 포맷 함수 -- 필수
function formatTime(rawTime) {
    if (!rawTime || rawTime.length !== 14) return "정보 없음"; // 시간 데이터가 없거나 길이가 다르면 처리
    const hour = rawTime.slice(8, 10); // 시간 추출
    const minute = rawTime.slice(10, 12); // 분 추출
    return `${hour}:${minute}`; // HH:mm 형식으로 반환
}


function displayRoute(response) {
    const resultFeatures = response.features; // 경로 및 포인트 정보
    const routeCoordinates = []; // 경로 좌표 배열 (애니메이션용)
    const processedStationsForTable = new Set(); // 중복 확인용 Set 객체

    // 테이블의 <table> 요소를 선택
    const table = document.querySelector("#schedule_div table");

    // 기존 데이터를 모두 초기화 (헤더 제외)
    table.innerHTML = `
        <tr>
            <th>순서<th>
            <th>시간</th>
            <th>위치</th>
            <th>작업</th>
        </tr>
    `;

    // 경로 요약 정보 추가
    const resultData = response.properties;
    const tDistance = `${(resultData.totalDistance / 1000).toFixed(1)} km`;
    const tTime = `${(resultData.totalTime / 60).toFixed(0)} 분`;
    const summaryRow = document.createElement("tr");
    summaryRow.innerHTML = `
        <td>-</td>
        <td>경로 요약</td>
        <td>총 거리: ${tDistance}, 총 시간: ${tTime}</td>
    `;
    table.appendChild(summaryRow);

    // 출발지 정보 추가
    const startFeature = resultFeatures[0];
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

    // Flask에서 가져온 경유지 데이터 매칭
    fetch('/final_output')
        .then(response => response.json())
        .then(simple_moves => {
            console.log("final_output_API에서 가져온 데이터:", simple_moves);

            // 경유지 데이터 처리
            resultFeatures.forEach(feature => {
                const properties = feature.properties;
                if (properties && properties.pointType && properties.pointType.startsWith("B")) {
                    const uniqueKey = `${properties.index}-${properties.viaPointName}`;
                    if (!processedStationsForTable.has(uniqueKey)) {
                        processedStationsForTable.add(uniqueKey); // 중복 제거
                        const visit_index = properties.visit_index;
                        const arriveTime = formatTime(properties.arriveTime) || "정보 없음";
                        const completeTime = formatTime(properties.completeTime) || "정보 없음";
                        const viaPointName = properties.viaPointName.replace(/^\[\d+\]\s*/, "");
                        const detailInfo = `다음 대여소 까지: ${(properties.distance / 1000).toFixed(1)} km`;

                        const stationData = simple_moves.find(
                            station => station.visit_station_name === properties.viaPointName.replace(/^\[\d+\]\s*/, "")
                        );
                        const stockInfo = stationData
                            ? `상태: ${stationData.status}\n현 재고: ${stationData.current_stock}\n필요 재고: ${stationData.move_bikes}`
                            : "";

                        const waypointRow = document.createElement("tr");
                        waypointRow.innerHTML = `
                            <td>${visit_index}<td>
                            <td>${arriveTime} ~ ${completeTime}</td>
                            <td>${viaPointName}</td>
                            <td>${stockInfo}<br>${detailInfo}</td>
                        `;
                        table.appendChild(waypointRow);
                    }
                }

                // 경로 애니메이션 좌표 추가
                if (feature.geometry.type === "LineString") {
                    const drawInfoArr = feature.geometry.coordinates.map(coord => {
                        return new Tmapv3.LatLng(coord[1], coord[0]);
                    });
                    routeCoordinates.push(...drawInfoArr);

                    // 지도에 경로 추가
                    new Tmapv3.Polyline({
                        path: drawInfoArr,
                        strokeColor: "#07c2db",
                        strokeWeight: 6,
                        direction: true,
                        map: map
                    });
                }
            });

            // 도착지 정보 추가
            const endFeature = resultFeatures[resultFeatures.length - 1];
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
    const animationMarker = new Tmapv3.Marker({
        position: routeCoordinates[0], // 경로의 시작점
        icon: "https://lh3.googleusercontent.com/d/1Vfe_log1g5Yv0-eG6XXsPMfvJvZ3PjP-", // 애니메이션 마커 아이콘
        anchor : "center",
        iconSize: new Tmapv3.Size(40, 40),
        zIndex: 2000,
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

// 따릉이 대여소 정보를 지도에 표시하는 함수
function displayBikeStations() {
    // 텍스트 파일 읽기
    // fetch('/static/all_gangnamgu_stations.txt')
    fetch('/data/all_gangnamgu_stations.txt')
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
                            zIndex: 100,
                            icon: "./static/images/icon/bike_icon.png", 
                        });
                        //Popup 객체 생성.
		                infoWindow = new Tmapv3.InfoWindow({
			            position: new Tmapv3.LatLng(station.stationLatitude, station.stationLongitude), //Popup 이 표출될 맵 좌표
			            content: `${station.stationId}\n남은 자전거: ${station.parkingBikeTotCnt} / ${station.rackTotCnt}`, //Popup 표시될 text
			            visible : false,
			            type: 1, //Popup의 type 설정.
			            map: map //Popup이 표시될 맵 객체
		                });

                        // 마커 클릭 이벤트
                        marker.on("click", function () {
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

//버튼 이벤트 설정
function setupButtons() {
    document.getElementById("create_traffic_route_btn").addEventListener("click", function(e) {
        e.preventDefault(); // 기본 이벤트 방지
        console.log("루트 생성 버튼 클릭됨");
        generateRoute();
    });
}

document.addEventListener("DOMContentLoaded", function () {
    console.log("페이지 로드 완료. Tmap 초기화 실행");
    initTmap();
    displayBikeStations();
    //버튼 이벤트 설정 //
    setupButtons();
});