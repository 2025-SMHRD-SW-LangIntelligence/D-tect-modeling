import requests
import json

# FastAPI 서버의 analyze_images 엔드포인트 주소
# 이 주소로 요청을 보내면 FastAPI가 분석을 시작하고 결과를 반환합니다.

# 이 코드는 사용 안했습니다
# 기존 방식에서 실시간 분석으로 바꿨기때문,,,
API_ENDPOINT = "http://127.0.0.1:8001/analyze_images"



def run_analysis():
    """
    FastAPI 서버에 분석 요청을 보냅니다.
    """
    print("--- 분석 시작 ---")
    print(f"FastAPI 서버의 '{API_ENDPOINT}'로 분석 요청을 보냅니다.")
    
    try:
        # POST 요청을 보냅니다. 이 엔드포인트는 요청 본문(body)이 필요하지 않습니다.
        response = requests.post(API_ENDPOINT)
        
        # HTTP 응답 상태 코드 확인
        if response.status_code == 200:
            print("요청 성공!")
            # 서버가 반환한 JSON 결과 출력
            analysis_results = response.json()
            
            print("\n--- 분석 결과 (JSON) ---")
            print(json.dumps(analysis_results, indent=2, ensure_ascii=False))
            print("------------------------")
            
        else:
            print(f"요청 실패: HTTP 상태 코드 {response.status_code}")
            print("응답 내용:")
            print(response.text)
            
    except requests.exceptions.RequestException as e:
        print(f"요청 중 오류 발생: {e}")

if __name__ == "__main__":
    run_analysis()