var map;
var marker_s, marker_e, waypoint;
var resultMarkerArr = [];
var bikeMarkers = []; // 따릉이
const waypointVisits = {}; // 중복방문
var resultInfoArr = []; // 루트그리기
var animationMarker; // 애니메이션 마커를 저장할 변수
var animationInterval; // 애니메이션의 타이머

// 사용자 시간입력 받기
function displaySelectedOptions(event) {
    event.preventDefault(); // 폼 제출 방지

    // 선택된 값 가져오기
    const month = document.getElementById('month').value;
    const day = document.getElementById('day').value;
    const hour = document.getElementById('hour').value;

    // 결과를 표시할 요소에 값 업데이트
    const resultDisplay = document.getElementById('result');
    resultDisplay.innerHTML = `선택한 날짜와 시간: ${month}월 ${day}일 ${hour}시`;
}

// 버튼을 다시 표시하는 함수
function showControls() {
    const controls = document.querySelector('.controlbotton'); // .controlbotton 요소 선택
    controls.style.display = 'flex'; // display 속성을 flex로 변경
}


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

    // 버튼 이벤트 설정
    setupButtons();
    
}

// 경유지 마커 추가 함수
function addWaypointMarkers() {
    const waypoints = [
        { lat: 37.516811, lng: 127.040474, index: "1", label:'ST-963'}, //ST-963 37.516811	127.040474
        { lat: 37.512810, lng: 127.026367, index: "2", label:'ST-3208'}, //ST-3208 37.512810	127.026367
        { lat: 37.516811, lng: 127.040474, index: "3", label:'ST-963'}, //ST-962 37.517590	127.035027  
        { lat: 37.518639, lng: 127.035400, index: "4", label:'ST-961'}, //ST-961 37.518639	127.035400
        { lat: 37.515888, lng: 127.066200, index: "5", label:'ST-784'}, //ST-784 37.515888	127.066200
        { lat: 37.517773, lng: 127.043022, index: "6", label:'ST-786'}, //ST-786 37.517773	127.043022
        { lat: 37.509586, lng: 127.040909, index: "7", label:'ST-1366'}, //ST-1366 37.509586	127.040909
        { lat: 37.509785, lng: 127.042770, index: "8", label:'ST-2882'}, //ST-2882 37.509785	127.042770
        { lat: 37.506367, lng: 127.034523, index: "9", label:'ST-1246'}, //ST-1246 37.506367	127.034523
        { lat: 37.505703, lng: 127.029198, index: "10", label:'ST-3108'}//ST-3108 37.505703	127.029198
         
    ];
   // 각 대여소별 방문 순서 추적
    waypoints.forEach((waypoint) => {
        if (!waypointVisits[waypoint.label]) {
            waypointVisits[waypoint.label] = [];
        }
        waypointVisits[waypoint.label].push(waypoint.index);
    });
    waypoints.forEach((waypoint) => {
        const labels = waypointVisits[waypoint.label];
        const isRevisited = labels.length > 1;

        // console.log("Labels array:", labels); // 배열 내용 확인
        // console.log("First label:", labels[0]); // 첫 번째 레이블 확인
        // console.log("Second label:", labels[1]); // 두 번째 레이블 확인
        // console.log("Is revisited:", isRevisited);

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
}



// 루트 생성 함수
function generateRoute() {
    // 마커를 표시
    resultMarkerArr.forEach(marker => marker.setVisible(true));

    const headers = {
        appKey: "6ockLdPQfZatxQctpKLtn5Lg7B5kDO555UzRAx0B"
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
            { viaPointId: "1", viaPointName:'ST-963',viaX: "127.040474", viaY: "37.516811", viaTime : 900}, //ST-963 37.516811	127.040474
            { viaPointId: "2", viaPointName:'ST-3208',viaX: "127.026367", viaY: "37.512810", viaTime : 900}, //ST-3208 37.512810	127.026367
            { viaPointId: "3", viaPointName:'ST-963',viaX: "127.040474", viaY: "37.516811", viaTime : 900}, //ST-962 37.517590	127.035027  
            { viaPointId: "4", viaPointName:'ST-961',viaX: "127.035400", viaY: "37.518639", viaTime : 900}, //ST-961 37.518639	127.035400
            { viaPointId: "5", viaPointName:'ST-784',viaX: "127.066200", viaY: "37.515888", viaTime : 900}, //ST-784 37.515888	127.066200
            { viaPointId: "6", viaPointName:'ST-786',viaX: "127.043022", viaY: "37.517773", viaTime : 900}, //ST-786 37.517773	127.043022
            { viaPointId: "7", viaPointName:'ST-1366',viaX: "127.040909", viaY: "37.509586", viaTime : 900}, //ST-1366 37.509586	127.040909
            { viaPointId: "8", viaPointName:'ST-2882',viaX: "127.042770", viaY: "37.509785", viaTime : 900}, //ST-2882 37.509785	127.042770
            { viaPointId: "9", viaPointName:'ST-1246',viaX: "127.034523", viaY: "37.506367",viaTime : 900}, //ST-1246 37.506367	127.034523
            { viaPointId: "10", viaPointName:'ST-3108',viaX: "127.029198", viaY: "37.505703", viaTime : 900}//ST-3108 37.505703	127.029198
           
        ],
        reqCoordType: "WGS84GEO",
        resCoordType: "WGS84GEO",
        searchOption: "2" // 고정: 교통최적+최소시간
    });

    $.ajax({
        method: "POST",
        url: "https://apis.openapi.sk.com/tmap/routes/routeSequential30?version=1&format=json",
        headers: headers,
        async: false,
        contentType: "application/json",
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
function displayRoute(response) {
    let routeCoordinates = []; // 경로 좌표 배열 (애니메이션용)
    // 지도에 경로 그리기
    const resultFeatures = response.features;
    resultFeatures.forEach(feature => {

        if (feature.geometry.type === "LineString") {
            const drawInfoArr = feature.geometry.coordinates.map(coord => {
                return new Tmapv3.LatLng(coord[1], coord[0]);

            });

            const polyline = new Tmapv3.Polyline({
                path: drawInfoArr,
                strokeColor: "#07c2db",
                strokeWeight: 6,
                direction: true,
                map: map
            });
            resultInfoArr.push(polyline);
            // 경로 좌표 추가 (애니메이션용)
            routeCoordinates = routeCoordinates.concat(drawInfoArr);
        }
    });
    // 경로 애니메이션 시작
    startRouteAnimation(routeCoordinates);
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