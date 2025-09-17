# ==============================================================================
# 📝 사전 준비: 라이브러리 임포트 및 API 키 설정
# ==============================================================================
# 최초 1회만 실행: !pip install google-generativeai pandas
import google.generativeai as genai
import pandas as pd
import json
import os
import time

PATH = './09_dongwon/'
os.makedirs(PATH, exist_ok=True) # 🔹 경로 수정: 폴더가 없으면 자동 생성

# [중요] 사용자의 API 키를 입력하세요.
try:
    # -------------------------------------------------------------------------
    genai.configure(api_key="AIzaSyApXG30wdefn3AwnlnyZVVK0zdfMmzVHPA")
    # -------------------------------------------------------------------------
    print("✅ Gemini API 키가 설정되었습니다.")
except Exception as e:
    print(f"❗️ API 키 설정 중 오류가 발생했습니다: {e}")
    
# print("✨ 사용 가능한 모델 목록:")
# for m in genai.list_models():
#   if 'generateContent' in m.supported_generation_methods:
#     print(m.name)

"""
✨ 사용 가능한 모델 목록:
models/gemini-1.5-pro-latest
models/gemini-1.5-pro-002
models/gemini-1.5-pro
models/gemini-1.5-flash-latest
models/gemini-1.5-flash
models/gemini-1.5-flash-002
models/gemini-1.5-flash-8b
models/gemini-1.5-flash-8b-001
models/gemini-1.5-flash-8b-latest
models/gemini-2.5-pro-preview-03-25
models/gemini-2.5-flash-preview-05-20
models/gemini-2.5-flash
models/gemini-2.5-flash-lite-preview-06-17
models/gemini-2.5-pro-preview-05-06
models/gemini-2.5-pro-preview-06-05
models/gemini-2.5-pro
models/gemini-2.0-flash-exp
models/gemini-2.0-flash
models/gemini-2.0-flash-001
models/gemini-2.0-flash-exp-image-generation
models/gemini-2.0-flash-lite-001
models/gemini-2.0-flash-lite
models/gemini-2.0-flash-preview-image-generation
models/gemini-2.0-flash-lite-preview-02-05
models/gemini-2.0-flash-lite-preview
models/gemini-2.0-pro-exp
models/gemini-2.0-pro-exp-02-05 [2]
models/gemini-exp-1206
models/gemini-2.0-flash-thinking-exp-01-21
models/gemini-2.0-flash-thinking-exp
models/gemini-2.0-flash-thinking-exp-1219
models/gemini-2.5-flash-preview-tts
models/gemini-2.5-pro-preview-tts
models/learnlm-2.0-flash-experimental
models/gemma-3-1b-it
models/gemma-3-4b-it
models/gemma-3-12b-it
models/gemma-3-27b-it
models/gemma-3n-e4b-it
models/gemma-3n-e2b-it
models/gemini-2.5-flash-lite 50 20 [1]
"""
   
# ==============================================================================
# 🤖 1단계: 범용 페르소나 생성을 위한 프롬프트 정의
# ==============================================================================
PROMPT_FOR_PERSONAS = """
당신은 대한민국 시장 전문 마케팅 분석가 AI입니다.
대한민국 소비자 전체를 대표할 수 있는 가상 소비자 페르소나 10000개를 생성하는 것이 당신의 임무입니다.
연령, 성별, 직업, 소득, 가구 형태, 라이프스타일 등이 현실의 분포와 유사하면서도 다양하게 생성되어야 합니다.
다른 설명은 모두 제외하고, 오직 아래의 구조를 따르는 JSON 배열 형식으로만 응답해주세요.

[
  {
    "persona_id": "P00001",
    "attributes": {
      "age": "30대", "gender": "여성", "occupation": "직장인", "income_level": "중상",
      "household_size": "1인 가구", "lifestyle": "건강지향", "media_consumption": ["YouTube", "Instagram"],
      "price_sensitivity": "중간", "brand_loyalty": "낮음", "dietary_preferences": "고단백"
    },
    "purchase_probability": 0.65, "base_purchase_frequency_per_month": 2.5,
    "monthly_propensity_modifier": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
  }
]
"""
print("✅ 1단계: 페르소나 생성 프롬프트가 준비되었습니다.")

# ==============================================================================
# 📥 2단계: Gemini API 호출 및 페르소나 데이터 생성 (속도 조절 적용)
# ==============================================================================
persona_filename = os.path.join(PATH, 'personas_10000.json') # 🔹 경로 수정
total_personas_to_generate = 10000
batch_size = 200
num_batches = total_personas_to_generate // batch_size

if not os.path.exists(persona_filename):
    print(f"\n⏳ 2단계: 총 {total_personas_to_generate}개의 페르소나 생성을 시작합니다 ({batch_size}개씩 {num_batches}회).")
    all_personas = []
    # (최신 버전)
    model = genai.GenerativeModel('models/gemini-2.0-pro-exp-02-05')
    
    for i in range(num_batches):
        print(f"  - 배치 {i+1}/{num_batches} 생성 중...")
        try:
            response = model.generate_content(PROMPT_FOR_PERSONAS)
            cleaned_response = response.text.strip().replace('```json', '').replace('```', '').strip()
            batch_personas = json.loads(cleaned_response)
            all_personas.extend(batch_personas)
            print(f"  ✅ 배치 {i+1} 생성 완료!")

        except Exception as e:
            print(f"  ❗️ 배치 {i+1} 생성 중 오류 발생: {e}. 다음 배치를 시도합니다.")
            # 오류 발생 시에는 조금 더 긴 시간 대기 후 재시도할 수 있습니다.
            time.sleep(20)

        # -------------------------------------------------------------------------
        # ✨ [수정된 부분] API 과부하를 막기 위해 각 배치 사이에 60초 대기합니다.
        # 마지막 배치가 완료된 후에는 기다릴 필요가 없습니다.
        if i < num_batches - 1:
            print("  🕒 API 무료 할당량(Quota)을 준수하기 위해 60초간 대기합니다...")
            time.sleep(60)
        # -------------------------------------------------------------------------

    # 생성된 모든 페르소나에 대해 고유 ID를 다시 부여합니다.
    for i, p in enumerate(all_personas):
        p['persona_id'] = f"P{i+1:05d}"
            
    with open(persona_filename, 'w', encoding='utf-8') as f:
        json.dump(all_personas, f, ensure_ascii=False, indent=2)
    print(f"✅ 성공! 총 {len(all_personas)}개의 페르소나를 '{persona_filename}' 파일로 저장했습니다.")
else:
    print(f"\n✅ 2단계: '{persona_filename}' 파일이 이미 존재하므로 API 호출을 생략합니다.")

# ==============================================================================
# 📂 데이터 로드 및 전처리
# ==============================================================================
try:
    personas_df = pd.read_json(persona_filename)
    attributes_df = pd.json_normalize(personas_df['attributes'])
    personas_df = pd.concat([personas_df.drop('attributes', axis=1), attributes_df], axis=1)
    
    product_info_df = pd.read_csv(os.path.join(PATH, 'product_info.csv')) # 🔹 경로 수정
    submission_df = pd.read_csv(os.path.join(PATH, 'sample_submission.csv')) # 🔹 경로 수정
    
    print("\n✅ 모든 데이터 로드 및 전처리가 완료되었습니다.")
except FileNotFoundError as e:
    print(f"❗️ 파일 로드 실패: {e}. 필요한 파일들이 모두 있는지 확인해주세요.")
    exit()

# ==============================================================================
# 🔬 3단계: 제품별 고객 세분화 및 가중치 부여 함수 정의
# ==============================================================================
def get_weights_by_product(product_name, df):
    weights = pd.Series(1.0, index=df.index)
    if '하이그릭요거트' in product_name:
        primary_mask = df['age'].isin(['20대', '30대', '40대']) & df['lifestyle'].isin(['건강지향', '자기관리'])
        weights[primary_mask] = 2.0
        secondary_mask = (df['household_size'] == '1인 가구') | (df['income_level'].isin(['중상', '상']))
        weights[secondary_mask & ~primary_mask] = 1.5
    elif '맛참' in product_name:
        primary_mask = df['age'].isin(['10대', '20대', '30대']) & df['media_consumption'].apply(lambda x: isinstance(x, list) and ('YouTube' in x or 'Instagram' in x))
        weights[primary_mask] = 2.5
        secondary_mask = (df['household_size'] == '1인 가구') & (df['lifestyle'] == '편의성추구')
        weights[secondary_mask & ~primary_mask] = 1.8
    elif '리챔' in product_name:
        primary_mask = df['age'].isin(['30대', '40대']) & df['occupation'].isin(['주부', '전업주부']) & (df['household_size'] != '1인 가구')
        weights[primary_mask] = 2.0
    elif '참치액' in product_name:
        primary_mask = df['age'].isin(['30대', '40대', '50대']) & df['occupation'].isin(['주부', '전업주부'])
        weights[primary_mask] = 2.2
        secondary_mask = (df['household_size'] == '1인 가구') & df['lifestyle'].isin(['요리애호가', '집밥선호'])
        weights[secondary_mask & ~primary_mask] = 1.7
    elif '소화가 잘되는' in product_name:
        primary_mask = df['age'].isin(['20대', '30대', '40대', '50대']) & df['dietary_preferences'].isin(['유당불내증케어', '소화편한음식선호'])
        weights[primary_mask] = 2.8
        secondary_mask = df['occupation'].isin(['직장인']) & df['lifestyle'].isin(['건강지향'])
        weights[secondary_mask & ~primary_mask] = 1.5
    return weights

print("✅ 3단계: 제품별 가중치 부여 함수가 준비되었습니다.")

# ==============================================================================
# 📈 4단계 & 5단계: 시뮬레이션 및 최종 파일 생성
# ==============================================================================
SIMULATION_PARAMS = {
    '하이그릭요거트': {'tam': 2000000, 'modifiers': [1.8, 1.5, 1.2, 1.0, 0.9, 0.9, 1.0, 1.0, 1.1, 1.1, 1.0, 1.0]},
    '맛참': {'tam': 5000000, 'modifiers': [3.0, 2.5, 1.5, 1.2, 1.0, 1.0, 1.0, 0.9, 0.9, 1.0, 1.1, 1.2]},
    '리챔': {'tam': 3000000, 'modifiers': [1.2, 1.1, 1.0, 1.0, 1.1, 1.2, 1.3, 1.2, 1.1, 1.0, 1.0, 1.1]},
    '참치액': {'tam': 2500000, 'modifiers': [1.1, 1.0, 1.0, 1.1, 1.2, 1.1, 1.2, 1.5, 1.4, 1.1, 1.0, 1.3]},
    '소화가 잘되는': {'tam': 1500000, 'modifiers': [1.0, 1.1, 1.2, 1.1, 1.0, 0.9, 0.9, 1.0, 1.1, 1.2, 1.2, 1.1]},
    'default': {'tam': 1000000, 'modifiers': [1.0] * 12}
}
print("✅ 4단계: 시뮬레이션 파라미터가 설정되었습니다.")

for index, row in submission_df.iterrows():
    product_name = row['product_name']
    params = SIMULATION_PARAMS.get(next((key for key in SIMULATION_PARAMS if key in product_name), 'default'))
    weights = get_weights_by_product(product_name, personas_df)
    
    monthly_sales = []
    num_personas = len(personas_df)
    for month_index in range(12):
        total_purchase_points = (
            personas_df['purchase_probability'] *
            personas_df['base_purchase_frequency_per_month'] *
            weights *
            params['modifiers'][month_index]
        ).sum()
        final_sales = (total_purchase_points / num_personas) * params['tam']
        monthly_sales.append(int(final_sales))
        
    submission_df.iloc[index, 1:] = monthly_sales

submission_df.to_csv(os.path.join(PATH, 'my_submission_08251525.csv'), index=False) # 🔹 경로 수정
print("\n✅ 5단계: 최종 제출 파일 'my_submission.csv' 생성이 완료되었습니다.")
print("\n[ 최종 예측 결과 (상위 5개 제품) ]")
print(submission_df.head())