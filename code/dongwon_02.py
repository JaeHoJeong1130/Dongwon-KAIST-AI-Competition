# ==============================================================================
# 📝 사전 준비: 라이브러리 임포트 및 API 키 설정 (기존과 동일)
# ==============================================================================
# 최초 1회만 실행: !pip install google-generativeai pandas
import google.generativeai as genai
import pandas as pd
import json
import os
import time
import re

PATH = './09_dongwon/'
os.makedirs(PATH, exist_ok=True)

# [중요] 사용자의 API 키를 입력하세요.
try:
    # -------------------------------------------------------------------------
    genai.configure(api_key="AIzaSyApXG30wdefn3AwnlnyZVVK0zdfMmzVHPA") # <--- 여기에 실제 API 키를 입력하세요.
    # -------------------------------------------------------------------------
    print("✅ Gemini API 키가 설정되었습니다.")
except Exception as e:
    print(f"❗️ API 키 설정 중 오류가 발생했습니다: {e}")

# ==============================================================================
# 🤖 1단계: 범용 페르소나 생성을 위한 프롬프트 정의 (✨ 간소화 버전)
# ==============================================================================
PROMPT_FOR_PERSONAS = """
당신은 한국 시장의 소비자 데이터를 분석하는 마케팅 전문가 AI입니다.

[지시사항]
1.  **임무**: '2024년 대한민국 소비자'를 대표하는 가상 페르소나 30개를 생성합니다.
2.  **조건**:
    * 생성되는 페르소나의 인구통계학적 분포는 실제 대한민국 통계와 유사해야 합니다.
    * `[속성 가이드]`를 참고하여 다양한 속성을 조합하되, 각 페르소나의 속성과 구매 행동(확률/빈도)은 논리적으로 연결되어야 합니다.
3.  **출력**: 다른 설명 없이 `[출력 형식 예시]`를 완벽히 따르는 단일 JSON 배열만 반환하세요.

[속성 가이드]
-   `age`: "10대", "20대", "30대", "40대", "50대", "60대 이상"
-   `gender`: "남성", "여성"
-   `occupation`: "직장인", "학생", "주부", "프리랜서", "자영업자", "무직", "은퇴"
-   `income_level`: "상", "중상", "중", "중하", "하"
-   `household_size`: "1인 가구", "2인 가구", "3인 가구", "4인 가구 이상"
-   `lifestyle`: "건강지향", "자기관리", "편의성추구", "가성비중시", "트렌드추구", "요리애호가", "집밥선호", "미식가"
-   `media_consumption`: "TV", "YouTube", "Instagram", "Facebook", "네이버뉴스", "커뮤니티사이트" (다중 선택 가능)
-   `price_sensitivity`: "높음", "중간", "낮음"
-   `brand_loyalty`: "높음", "중간", "낮음"
-   `dietary_preferences`: "고단백", "저칼로리", "유당불내증케어", "소화편한음식선호", "매운맛선호", "해산물선호", "해당없음"
-   `shopping_channel`: "대형마트", "온라인", "편의점", "백화점", "전통시장"

[출력 형식 예시]
[
  {
    "persona_id": "P00001",
    "attributes": {
      "age": "30대",
      "gender": "여성",
      "occupation": "직장인",
      "income_level": "중상",
      "household_size": "1인 가구",
      "lifestyle": "건강지향",
      "media_consumption": ["YouTube", "Instagram"],
      "price_sensitivity": "중간",
      "brand_loyalty": "낮음",
      "dietary_preferences": "고단백",
      "shopping_channel": "온라인"
    },
    "purchase_probability": 0.75,
    "base_purchase_frequency_per_month": 3.0,
    "monthly_propensity_modifier": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
  }
]
"""
print("✅ 1단계: [간소화] 핵심 지시사항을 담은 페르소나 생성 프롬프트가 준비되었습니다.")

# ==============================================================================
# 📥 2단계: Gemini API 호출 및 페르소나 데이터 생성 (✨ 안정성 및 데이터 품질 대폭 개선)
# ==============================================================================

def extract_json_from_response(text):
    """정규표현식을 사용해 모델 응답에서 JSON 배열 부분만 정확히 추출합니다."""
    # 가장 먼저 나오는 '[' 와 가장 마지막에 나오는 ']' 사이의 내용을 찾습니다.
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        return match.group(0)
    return None

def validate_persona(persona_dict):
    """생성된 페르소나 객체에 필수 키가 모두 있는지 검증합니다."""
    required_keys = ['persona_id', 'attributes', 'purchase_probability', 'base_purchase_frequency_per_month']
    return all(key in persona_dict for key in required_keys)


persona_filename = os.path.join(PATH, 'personas_2_3000.json')
total_personas_to_generate = 3000
batch_size = 30
num_batches = total_personas_to_generate // batch_size
max_retries_per_batch = 5 # 배치 당 최대 재시도 횟수

if not os.path.exists(persona_filename):
    print(f"\n⏳ 2단계: 총 {total_personas_to_generate}개의 페르소나 생성을 시작합니다 ({batch_size}개씩 {num_batches}회).")
    all_personas = []
    model = genai.GenerativeModel('models/gemini-2.0-flash')

    for i in range(num_batches):
        current_batch_success = False
        for attempt in range(max_retries_per_batch):
            print(f"  - 배치 {i+1}/{num_batches} 생성 중... (시도 {attempt+1}/{max_retries_per_batch})")
            try:
                response = model.generate_content(PROMPT_FOR_PERSONAS)
                
                # 1. 지능적인 JSON 추출
                json_text = extract_json_from_response(response.text)
                if not json_text:
                    raise ValueError("응답에서 JSON 배열을 찾을 수 없습니다.")

                batch_personas = json.loads(json_text)

                # 2. 데이터 검증
                if not all(validate_persona(p) for p in batch_personas):
                    raise ValueError("일부 페르소나에 필수 키가 누락되었습니다.")

                all_personas.extend(batch_personas)
                print(f"  ✅ 배치 {i+1} 생성 완료! (현재 {len(all_personas)}개)")
                current_batch_success = True
                break # 현재 배치 성공 시, 재시도 루프 탈출

            except Exception as e:
                print(f"  ❗️ 배치 {i+1} 시도 {attempt+1} 실패: {e}")
                if attempt < max_retries_per_batch - 1:
                    print("  15초 후 재시도합니다...")
                    time.sleep(15)
                else:
                    print(f"  ❌ 배치 {i+1} 생성 최종 실패. 다음 배치로 넘어갑니다.")
        
        # 3. API 할당량 준수를 위한 대기
        if i < num_batches - 1:
            print("  🕒 API 할당량 준수를 위해 60초간 대기합니다...")
            time.sleep(60)

    # 최종 ID 재설정
    for i, p in enumerate(all_personas):
        p['persona_id'] = f"P{i+1:05d}"
            
    with open(persona_filename, 'w', encoding='utf-8') as f:
        json.dump(all_personas, f, ensure_ascii=False, indent=2)
    print(f"✅ 성공! 총 {len(all_personas)}개의 페르소나를 '{persona_filename}' 파일로 저장했습니다.")
else:
    print(f"\n✅ 2단계: '{persona_filename}' 파일이 이미 존재하므로 API 호출을 생략합니다.")

# ==============================================================================
# 📂 데이터 로드 및 전처리 (기존과 동일)
# ==============================================================================
try:
    personas_df = pd.read_json(persona_filename)
    attributes_df = pd.json_normalize(personas_df['attributes'])
    personas_df = pd.concat([personas_df.drop('attributes', axis=1), attributes_df], axis=1)
    # product_info.csv는 이제 SIMULATION_PARAMS로 대체되므로 로드하지 않아도 됩니다.
    submission_df = pd.read_csv(os.path.join(PATH, 'sample_submission.csv'))
    print("\n✅ 모든 데이터 로드 및 전처리가 완료되었습니다.")
except FileNotFoundError as e:
    print(f"❗️ 파일 로드 실패: {e}. 필요한 파일들이 모두 있는지 확인해주세요.")
    exit()

# ==============================================================================
# 🔬 3단계: 제품별 고객 세분화 및 가중치 부여 함수 정의 (✨ 대폭 고도화됨)
# ==============================================================================
def get_weights_by_product(product_name, df):
    weights = pd.Series(1.0, index=df.index)
    
    # ------------------- [ 1. 제품군별 기본 가중치 부여 (기존 로직) ] -------------------
    if '하이그릭요거트' in product_name:
        primary_mask = df['age'].isin(['20대', '30대', '40대']) & df['lifestyle'].isin(['건강지향', '자기관리'])
        weights[primary_mask] *= 2.0 # 기존 가중치에 곱하는 방식(*=)으로 변경하여 중첩 적용
        secondary_mask = (df['household_size'] == '1인 가구') | (df['income_level'].isin(['중상', '상']))
        weights[secondary_mask & ~primary_mask] *= 1.5
        
    elif '맛참' in product_name:
        primary_mask = df['age'].isin(['10대', '20대', '30대']) & df['media_consumption'].apply(lambda x: isinstance(x, list) and ('YouTube' in x or 'Instagram' in x))
        primary_mask &= df['shopping_channel'].isin(['편의점', '온라인'])
        weights[primary_mask] *= 2.5
        secondary_mask = (df['household_size'] == '1인 가구') & (df['lifestyle'] == '편의성추구')
        weights[secondary_mask & ~primary_mask] *= 1.8

    elif '리챔' in product_name:
        primary_mask = df['age'].isin(['30대', '40대']) & df['occupation'].isin(['주부', '전업주부']) & (df['household_size'] != '1인 가구')
        primary_mask &= df['shopping_channel'].isin(['대형마트'])
        weights[primary_mask] *= 2.0

    elif '참치액' in product_name:
        primary_mask = df['age'].isin(['30대', '40대', '50대']) & df['occupation'].isin(['주부', '전업주부'])
        primary_mask &= df['shopping_channel'].isin(['대형마트', '온라인'])
        weights[primary_mask] *= 2.2
        secondary_mask = (df['household_size'] == '1인 가구') & df['lifestyle'].isin(['요리애호가', '집밥선호'])
        weights[secondary_mask & ~primary_mask] *= 1.7

    elif '소화가 잘되는' in product_name:
        primary_mask = df['age'].isin(['20대', '30대', '40대', '50대']) & df['dietary_preferences'].isin(['유당불내증케어', '소화편한음식선호'])
        primary_mask &= df['shopping_channel'].isin(['편의점'])
        weights[primary_mask] *= 2.8
        secondary_mask = df['occupation'].isin(['직장인']) & df['lifestyle'].isin(['건강지향'])
        weights[secondary_mask & ~primary_mask] *= 1.5

    # ------------------- [ 2. 용량/맛/특성별 상세 가중치 추가 부여 ] -------------------
    # [가설 1] 가구 규모에 따른 용량 선호도
    # - 1인 가구: 소용량 선호 (+20%), 대용량 기피 (-10%)
    # - 다인 가구: 대용량 선호 (+20%), 소용량 기피 (-10%)
    is_large_capacity = any(x in product_name for x in ['400g', '900g', '340g'])
    is_small_capacity = any(x in product_name for x in ['90g', '135g', '200g', '500g']) # 500g는 중간이지만 여기선 작은편으로 분류
    
    single_household_mask = df['household_size'] == '1인 가구'
    multi_household_mask = df['household_size'] != '1인 가구'

    if is_large_capacity:
        weights[single_household_mask] *= 0.9
        weights[multi_household_mask] *= 1.2
    elif is_small_capacity:
        weights[single_household_mask] *= 1.2
        weights[multi_household_mask] *= 0.9

    # [가설 2] 맛(매운맛/진한맛) 선호도
    # - 매운맛: 젊은 층(10-20대) 선호도 높음 (+30%)
    # - 진한맛/프리미엄: 요리애호가, 고소득층 선호 (+30%)
    if '매콤' in product_name:
        spicy_lover_mask = df['age'].isin(['10대', '20대'])
        weights[spicy_lover_mask] *= 1.3
        
    if '진' in product_name or '프리미엄' in product_name:
        pro_cook_mask = df['lifestyle'].isin(['요리애호가']) | df['income_level'].isin(['중상', '상'])
        weights[pro_cook_mask] *= 1.3
        
    # [가설 3] RTD 커피 맛 선호도
    # - 바닐라라떼: 단맛 선호 경향이 있는 10-20대 선호 (+15%)
    if '바닐라라떼' in product_name:
        sweet_lover_mask = df['age'].isin(['10대', '20대'])
        weights[sweet_lover_mask] *= 1.15
        
    return weights

print("✅ 3단계: [고도화] 제품 특성(용량, 맛)을 반영한 가중치 부여 함수가 준비되었습니다.")


# ==============================================================================
# 📈 4단계 & 5단계: 시뮬레이션 및 최종 파일 생성 (✨ SKU 단위로 파라미터 세분화)
# ==============================================================================
# ✨ [핵심 수정] 제품군이 아닌, 개별 제품(SKU) 단위로 시장 규모(TAM)와 점유율(market_share)을 재정의
#    - 동일 제품군 내에서 용량/맛에 따라 시장 점유율을 분배 (합계는 기존과 유사하게 설정)
#    - 명절 Modifiers는 관련 제품군(리챔, 참치액)에 동일하게 적용
HOLIDAY_MODIFIERS = {
    'seollal_chuseok': [1.8, 5.5, 1.0, 0.9, 1.0, 1.1, 1.2, 2.5, 6.5, 1.0, 1.0, 1.1],
    'seollal_chuseok_sub': [1.3, 3.0, 1.0, 1.1, 1.2, 1.1, 1.2, 1.8, 3.5, 1.1, 1.0, 1.3]
}
DEFAULT_MODIFIERS = [1.0] * 12

SIMULATION_PARAMS = {
    # 하이그릭요거트 (TAM: 60만, 점유율: 0.1)
    '덴마크 하이그릭요거트 400g': {'tam': 600000, 'market_share': 0.05, 'modifiers': [1.2, 1.2, 1.2, 1.4, 1.4, 1.3, 1.2, 1.2, 0.9, 0.8, 0.7, 0.7]},
    
    # 맛참 (TAM: 100만, 점유율: 0.2) -> 4개 제품이 0.2 점유율을 나눠 가짐
    '동원맛참 고소참기름 135g': {'tam': 2000000, 'market_share': 0.07, 'modifiers': [1.2, 1.2, 1.1, 1.0, 1.0, 1.0, 1.0, 1.2, 1.2, 1.0, 1.1, 1.2]},
    '동원맛참 고소참기름 90g':  {'tam': 2000000, 'market_share': 0.05, 'modifiers': [1.2, 1.2, 1.1, 1.0, 1.0, 1.0, 1.0, 1.2, 1.2, 1.0, 1.1, 1.2]},
    '동원맛참 매콤참기름 135g': {'tam': 2000000, 'market_share': 0.05, 'modifiers': [1.2, 1.2, 1.1, 1.0, 1.0, 1.0, 1.0, 1.2, 1.2, 1.0, 1.1, 1.2]},
    '동원맛참 매콤참기름 90g':  {'tam': 2000000, 'market_share': 0.03, 'modifiers': [1.2, 1.2, 1.1, 1.0, 1.0, 1.0, 1.0, 1.2, 1.2, 1.0, 1.1, 1.2]},
    
    # 리챔 (TAM: 300만, 점유율: 0.8) -> 2개 제품이 0.8 점유율을 나눠 가짐 (명절 특수성 반영)
    '리챔 오믈렛햄 200g': {'tam': 3000000, 'market_share': 0.3, 'modifiers': HOLIDAY_MODIFIERS['seollal_chuseok']},
    '리챔 오믈렛햄 340g': {'tam': 3000000, 'market_share': 0.5, 'modifiers': HOLIDAY_MODIFIERS['seollal_chuseok']},

    # 참치액 (TAM: 25만, 점유율: 0.125) -> 6개 제품이 0.125 점유율을 나눠 가짐 (명절 특수성 반영)
    '동원참치액 순 500g':  {'tam': 250000, 'market_share': 0.025, 'modifiers': HOLIDAY_MODIFIERS['seollal_chuseok_sub']},
    '동원참치액 순 900g':  {'tam': 250000, 'market_share': 0.020, 'modifiers': HOLIDAY_MODIFIERS['seollal_chuseok_sub']},
    '동원참치액 진 500g':  {'tam': 250000, 'market_share': 0.030, 'modifiers': HOLIDAY_MODIFIERS['seollal_chuseok_sub']},
    '동원참치액 진 900g':  {'tam': 250000, 'market_share': 0.025, 'modifiers': HOLIDAY_MODIFIERS['seollal_chuseok_sub']},
    '프리미엄 동원참치액 500g': {'tam': 250000, 'market_share': 0.015, 'modifiers': HOLIDAY_MODIFIERS['seollal_chuseok_sub']},
    '프리미엄 동원참치액 900g': {'tam': 250000, 'market_share': 0.010, 'modifiers': HOLIDAY_MODIFIERS['seollal_chuseok_sub']},
    
    # 소화가 잘되는 우유 (TAM: 150만, 점유율: 0.4) -> 2개 제품이 0.4 점유율을 나눠 가짐
    '소화가 잘되는 우유로 만든 바닐라라떼 250mL': {'tam': 2000000, 'market_share': 0.17, 'modifiers': [1.0, 1.1, 1.2, 1.1, 1.0, 1.0, 1.0, 1.0, 1.1, 1.1, 1.1, 1.1]},
    '소화가 잘되는 우유로 만든 카페라떼 250mL':   {'tam': 2000000, 'market_share': 0.23, 'modifiers': [1.0, 1.1, 1.2, 1.1, 1.0, 1.0, 1.0, 1.0, 1.1, 1.1, 1.1, 1.1]},
    
    'default': {'tam': 1000000, 'market_share': 0.10, 'modifiers': DEFAULT_MODIFIERS}
}
print("✅ 4단계: [세분화] SKU별 시장 점유율 및 '명절 특수성'이 포함된 시뮬레이션 파라미터가 설정되었습니다.")

START_MONTH_INDEX = 6 # 7월은 리스트에서 인덱스 6에 해당

for product, params in SIMULATION_PARAMS.items():
    original_modifiers = params['modifiers']
    # 7월부터 끝까지의 데이터를 앞으로 가져오고, 1월부터 6월까지의 데이터를 뒤로 붙입니다.
    reordered_modifiers = original_modifiers[START_MONTH_INDEX:] + original_modifiers[:START_MONTH_INDEX]
    SIMULATION_PARAMS[product]['modifiers'] = reordered_modifiers
    # 예: [1,2,3,4,5,6,7,8,9,10,11,12] -> [7,8,9,10,11,12,1,2,3,4,5,6]

print("✅ 모든 제품의 월별 가중치 재정렬이 완료되었습니다.")
# ==============================================================================

# 파일명에 타임스탬프 추가하여 버전 관리
timestamp = time.strftime("%Y%m%d_%H%M%S")
submission_filename = os.path.join(PATH, f'my_submission_v2.1_{timestamp}.csv')

for index, row in submission_df.iterrows():
    product_name = row['product_name']
    
    params = SIMULATION_PARAMS.get(product_name, SIMULATION_PARAMS['default'])
    
    weights = get_weights_by_product(product_name, personas_df)

    monthly_sales = []
    num_personas = len(personas_df)
    for month_index in range(12):
        total_purchase_points = (
            personas_df['purchase_probability'] *
            personas_df['base_purchase_frequency_per_month'] *
            weights *
            params['modifiers'][month_index] # 이제 params['modifiers'][0]은 7월 가중치를 의미합니다.
        ).sum()

        final_sales = (total_purchase_points / num_personas) * params['tam'] * params['market_share']
        monthly_sales.append(int(final_sales))

    submission_df.iloc[index, 1:] = monthly_sales

submission_df.to_csv(submission_filename, index=False)
print(f"\n✅ 5단계: 최종 제출 파일 '{submission_filename}' 생성이 완료되었습니다.")
print("\n[ 최종 예측 결과 (상위 5개 제품) ]")
print(submission_df.head())