# ==============================================================================
# 📝 사전 준비: 라이브러리 임포트 및 API 키 설정
# ==============================================================================
import google.generativeai as genai
import pandas as pd
import json
import os
import time
import re
import logging
import numpy as np # ✨ 계산을 위해 NumPy 임포트

# ✨ 경로 설정
PATH = './09_dongwon/'
os.makedirs(PATH, exist_ok=True)

# 🪵 ======================= 로거 설정 ======================= 🪵
timestamp = time.strftime("%Y%m%d_%H%M%S")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# 스트림 핸들러 (콘솔 출력)
if not logger.handlers:
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # 파일 핸들러 (파일 출력)
    log_filename = os.path.join(PATH, f'v5_log_{timestamp}.log')
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

logger.info("✅ 로깅 설정이 완료되었습니다.")
# 🪵 =============================================================== 🪵

# [중요] 사용자의 API 키를 입력하세요.
try:
    # -------------------------------------------------------------------------
    genai.configure(api_key="AIzaSyApXG30wdefn3AwnlnyZVVK0zdfMmzVHPA") # <--- ⚠️ 여기에 실제 API 키를 입력하세요.
    # -------------------------------------------------------------------------
    model = genai.GenerativeModel('models/gemini-2.0-flash') # 모델명은 최신 버전으로 유지하는 것이 좋습니다.
    logger.info("✅ Gemini API 키가 설정되었습니다.")
except Exception as e:
    logger.error(f"❗️ API 키 설정 중 오류가 발생했습니다: {e}")

# 헬퍼 함수
def extract_json_from_response(text):
    """응답 텍스트에서 JSON 블록을 추출합니다."""
    # 마크다운 JSON 블록(` ```json ... ``` `)을 찾기 위한 정규 표현식
    match = re.search(r'```json\s*(\[.*\])\s*```', text, re.DOTALL)
    if match:
        return match.group(1)
    # JSON이 마크다운 블록 없이 바로 오는 경우를 대비
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        return match.group(0)
    return None

# ==============================================================================
# ✨ 1. 제품별 페르소나 생성을 위한 동적 프롬프트 생성 함수
# ==============================================================================
def create_product_specific_prompt(product_name, num_personas=30):
    """
    제품 이름과 특성을 기반으로 맞춤형 페르소나 생성 프롬프트를 생성합니다.
    """
    target_customer_profile = "일반적인 대한민국 소비자" # 기본값

    # --- 제품군별 타겟 고객 프로필 정의 ---
    if '하이그릭요거트' in product_name:
        target_customer_profile = "20-40대 여성으로, 건강과 자기관리에 관심이 많고, 소득 수준은 중상 이상인 1인 가구. 주로 온라인 채널을 통해 건강식품을 구매하는 경향이 있음."
    elif '맛참' in product_name:
        target_customer_profile = "10-30대 남녀로, 편의성과 새로운 맛을 추구하며 유튜브, 인스타그램 등 소셜 미디어에 익숙함. 편의점이나 온라인에서 간편하게 식사를 해결하려는 1인 가구 학생 및 직장인."
        if '매콤' in product_name:
            target_customer_profile += " 특히 매운맛을 선호하는 경향이 뚜렷함."
    elif '리챔' in product_name:
        target_customer_profile = "30-50대 주부로, 3인 이상 가구의 식사를 책임지고 있음. 대형마트에서 장을 보며, 명절 등 특별한 날에 가족을 위한 요리를 준비하는 것을 중요하게 생각함."
    elif '참치액' in product_name:
        target_customer_profile = "요리에 관심이 많은 30-60대 주부 또는 1인 가구. 집밥을 선호하며, 음식의 깊은 맛을 내기 위한 조미료에 투자를 아끼지 않음. 대형마트와 온라인 채널을 모두 이용함."
        if '진' in product_name or '프리미엄' in product_name:
            target_customer_profile += " 특히 요리 실력이 뛰어나고, 소득 수준이 높아 프리미엄 제품을 선호하는 미식가적 성향을 보임."
    elif '소화가 잘되는' in product_name:
        target_customer_profile = "유당불내증이 있거나 소화 건강에 신경 쓰는 20-50대. 건강을 위해 일반 유제품 대신 락토프리 제품을 선택하며, 출근길이나 점심시간에 편의점에서 자주 구매함."
        if '바닐라라떼' in product_name:
            target_customer_profile += " 단맛을 선호하는 젊은 층의 비중이 상대적으로 높음."

    logger.info(f" 🎯 타겟 프로필 설정: {target_customer_profile}")

    prompt = f"""
    당신은 특정 제품의 핵심 구매 고객 페르소나를 생성하는 마케팅 분석 AI입니다.

    [지시사항]
    1.  **임무**: 아래 [제품 정보]에 명시된 제품을 구매할 가능성이 매우 높은 **핵심 고객 페르소나 {num_personas}개**를 생성합니다.
    2.  **핵심 조건**: 생성되는 페르소나는 아래 [타겟 고객 프로필]의 특성을 집중적으로 반영해야 합니다. 페르소나의 모든 속성은 이 프로필과 논리적으로 강력하게 연결되어야 합니다.
    3.  **출력**: 다른 설명 없이, [출력 형식 예시]를 완벽히 따르는 단일 JSON 배열만 반환하세요. JSON은 반드시 ```json ... ``` 코드 블록 안에 포함시켜 주세요.

    [제품 정보]
    - 제품명: "{product_name}"

    [타겟 고객 프로필]
    - {target_customer_profile}

    [속성 가이드]
    - `age`: "10대", "20대", "30대", "40대", "50대", "60대 이상"
    - `gender`: "남성", "여성"
    - `occupation`: "직장인", "학생", "주부", "프리랜서", "자영업자", "무직", "은퇴"
    - `income_level`: "상", "중상", "중", "중하", "하"
    - `household_size`: "1인 가구", "2인 가구", "3인 가구", "4인 가구 이상"
    - `lifestyle`: "건강지향", "자기관리", "편의성추구", "가성비중시", "트렌드추구", "요리애호가", "집밥선호", "미식가"
    - `media_consumption`: ["YouTube", "Instagram", "TV", "커뮤니티사이트"] 등
    - `price_sensitivity`: "높음", "중간", "낮음"
    - `brand_loyalty`: "높음", "중간", "낮음"
    - `dietary_preferences`: "고단백", "저칼로리", "유당불내증케어", "소화편한음식선호", "매운맛선호" 등
    - `shopping_channel`: "대형마트", "온라인", "편의점", "백화점"

    [출력 형식 예시]
    ```json
    [
      {{
        "persona_id": "P00001", "attributes": {{"age": "30대", "gender": "여성", "occupation": "직장인", "income_level": "중상", "household_size": "1인 가구", "lifestyle": "건강지향", "media_consumption": ["YouTube", "Instagram"], "price_sensitivity": "중간", "brand_loyalty": "낮음", "dietary_preferences": "고단백", "shopping_channel": "온라인"}}, "purchase_probability": 0.85, "base_purchase_frequency_per_month": 4.0, "monthly_propensity_modifier": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
      }}
    ]
    ```
    """
    return prompt

logger.info("✅ 1. 제품별 동적 프롬프트 생성 함수가 준비되었습니다.")

# ==============================================================================
# ✨ 2. [신규] 에이전트 및 시장 시뮬레이션 클래스 정의
# ==============================================================================
class PersonaAgent:
    """개별 페르소나(에이전트)의 상태와 행동을 관리하는 클래스"""
    def __init__(self, persona_data):
        self.id = persona_data.get('persona_id', 'N/A')
        self.attributes = persona_data.get('attributes', {})
        self.base_purchase_rate = persona_data.get('base_purchase_frequency_per_month', 0) / 30.0
        self.state = 'Unaware'  # 초기 상태: Unaware, Aware, Active, Churned
        self.p_churn = 0.05      # 월별 이탈 확률 (5%)

    def update_state(self, p_innovation, p_imitation, adoption_rate):
        """배스 확산 모델에 따라 에이전트의 상태를 업데이트합니다."""
        if self.state == 'Unaware':
            prob_aware = p_innovation + p_imitation * adoption_rate
            if np.random.rand() < prob_aware:
                self.state = 'Active'
        elif self.state == 'Active':
            if np.random.rand() < self.p_churn:
                self.state = 'Churned'

    def attempt_purchase(self, month_modifier):
        """이번 달에 구매를 시도하는지 여부를 결정합니다."""
        if self.state == 'Active':
            monthly_purchase_prob = (1 - (1 - self.base_purchase_rate)**30) * month_modifier
            if np.random.rand() < monthly_purchase_prob:
                return 1
        return 0

class MarketSimulation:
    """ABM과 배스 확산 모델을 사용하여 시장을 시뮬레이션하는 클래스"""
    def __init__(self, personas, tam, market_share, modifiers, initial_adoption_rate=0.10):
        self.agents = [PersonaAgent(p) for p in personas if p] # 비어있는 페르소나 데이터 방지
        self.potential_market_size = int(tam * market_share)
        self.modifiers = modifiers
        self.p_innovation = 0.01  # 혁신 계수 (Bass Model)
        self.q_imitation = 0.38   # 모방 계수 (Bass Model)
        
        # ⭐ 시작 시점에 일부 에이전트를 'Active' 상태로 전환
        num_initial_adopters = int(len(self.agents) * initial_adoption_rate)
        for i in range(num_initial_adopters):
            self.agents[i].state = 'Active'
        
        self.adopters = int(self.potential_market_size * initial_adoption_rate) # 초기 채택자 수 업데이트

    def run_simulation(self, months=12):
        """지정된 개월 수 동안 시뮬레이션을 실행합니다."""
        monthly_sales_results = []
        num_agents = len(self.agents)

        if num_agents == 0:
            logger.warning("⚠️ 에이전트가 없어 시뮬레이션을 진행할 수 없습니다. 0을 반환합니다.")
            return [0] * months

        for month_index in range(months):
            adoption_rate = self.adopters / self.potential_market_size if self.potential_market_size > 0 else 0
            
            current_month_sales = 0
            active_agents_count = 0
            
            for agent in self.agents:
                agent.update_state(self.p_innovation, self.q_imitation, adoption_rate)
                month_modifier = self.modifiers[month_index]
                current_month_sales += agent.attempt_purchase(month_modifier)
                if agent.state == 'Active':
                    active_agents_count += 1
            
            sample_purchase_rate = current_month_sales / num_agents
            extrapolated_sales = sample_purchase_rate * self.potential_market_size
            monthly_sales_results.append(int(extrapolated_sales))
            
            # 다음 달을 위해 채택자 수 업데이트 (샘플 비율을 전체 시장에 적용)
            self.adopters = (active_agents_count / num_agents) * self.potential_market_size

        return monthly_sales_results

logger.info("✅ 2. 에이전트 기반 시뮬레이션 클래스가 준비되었습니다.")

# ==============================================================================
# ✨ 3. 시뮬레이션 파라미터 정의
# ==============================================================================

# ⭐ 수정/추가된 부분 1: 제품별 출시일 정보를 딕셔너리로 정의합니다. (연도, 월)
LAUNCH_DATES = {
    '덴마크 하이그릭요거트 400g': (2025, 2),
    '리챔 오믈렛햄 200g': (2025, 5),
    '리챔 오믈렛햄 340g': (2025, 5),
    '소화가 잘되는 우유로 만든 바닐라라떼 250mL': (2025, 2),
    '소화가 잘되는 우유로 만든 카페라떼 250mL': (2025, 2)
}
logger.info("✅ 3-1. 제품별 출시일 정보가 설정되었습니다.")

HOLIDAY_MODIFIERS = {
    'seollal_chuseok': [1.8, 5.5, 1.0, 0.9, 1.0, 1.1, 1.2, 2.5, 6.5, 1.0, 1.0, 1.1],
    'seollal_chuseok_sub': [1.3, 3.0, 1.0, 1.1, 1.2, 1.1, 1.2, 1.8, 3.5, 1.1, 1.0, 1.3]
}
DEFAULT_MODIFIERS = [1.0] * 12
SIMULATION_PARAMS = {
    '덴마크 하이그릭요거트 400g': {'tam': 6000000, 'market_share': 0.05, 'modifiers': [1.2, 1.2, 1.2, 1.4, 1.4, 1.3, 1.2, 1.2, 0.9, 0.8, 0.7, 0.7]},
    '동원맛참 고소참기름 135g': {'tam': 20000000, 'market_share': 0.07, 'modifiers': [1.2, 1.2, 1.1, 1.0, 1.0, 1.0, 1.0, 1.2, 1.2, 1.0, 1.1, 1.2]},
    '동원맛참 고소참기름 90g':  {'tam': 20000000, 'market_share': 0.05, 'modifiers': [1.2, 1.2, 1.1, 1.0, 1.0, 1.0, 1.0, 1.2, 1.2, 1.0, 1.1, 1.2]},
    '동원맛참 매콤참기름 135g': {'tam': 20000000, 'market_share': 0.05, 'modifiers': [1.2, 1.2, 1.1, 1.0, 1.0, 1.0, 1.0, 1.2, 1.2, 1.0, 1.1, 1.2]},
    '동원맛참 매콤참기름 90g':  {'tam': 20000000, 'market_share': 0.03, 'modifiers': [1.2, 1.2, 1.1, 1.0, 1.0, 1.0, 1.0, 1.2, 1.2, 1.0, 1.1, 1.2]},
    '리챔 오믈렛햄 200g': {'tam': 3000000, 'market_share': 0.3, 'modifiers': HOLIDAY_MODIFIERS['seollal_chuseok']},
    '리챔 오믈렛햄 340g': {'tam': 3000000, 'market_share': 0.5, 'modifiers': HOLIDAY_MODIFIERS['seollal_chuseok']},
    '동원참치액 순 500g':  {'tam': 2500000, 'market_share': 0.075, 'modifiers': HOLIDAY_MODIFIERS['seollal_chuseok_sub']},
    '동원참치액 순 900g':  {'tam': 2500000, 'market_share': 0.060, 'modifiers': HOLIDAY_MODIFIERS['seollal_chuseok_sub']},
    '동원참치액 진 500g':  {'tam': 2500000, 'market_share': 0.090, 'modifiers': HOLIDAY_MODIFIERS['seollal_chuseok_sub']},
    '동원참치액 진 900g':  {'tam': 2500000, 'market_share': 0.075, 'modifiers': HOLIDAY_MODIFIERS['seollal_chuseok_sub']},
    '프리미엄 동원참치액 500g': {'tam': 2500000, 'market_share': 0.05, 'modifiers': HOLIDAY_MODIFIERS['seollal_chuseok_sub']},
    '프리미엄 동원참치액 900g': {'tam': 2500000, 'market_share': 0.02, 'modifiers': HOLIDAY_MODIFIERS['seollal_chuseok_sub']},
    '소화가 잘되는 우유로 만든 바닐라라떼 250mL': {'tam': 12000000, 'market_share': 0.09, 'modifiers': [1.0, 1.1, 1.2, 1.1, 1.0, 1.0, 1.0, 1.0, 1.1, 1.1, 1.1, 1.1]},
    '소화가 잘되는 우유로 만든 카페라떼 250mL':   {'tam': 12000000, 'market_share': 0.13, 'modifiers': [1.0, 1.1, 1.2, 1.1, 1.0, 1.0, 1.0, 1.0, 1.1, 1.1, 1.1, 1.1]},
    'default': {'tam': 10000000, 'market_share': 0.10, 'modifiers': DEFAULT_MODIFIERS}
}
START_MONTH_INDEX = 6 # 7월부터 시작
for product, params in SIMULATION_PARAMS.items():
    original_modifiers = params['modifiers']
    reordered_modifiers = original_modifiers[START_MONTH_INDEX:] + original_modifiers[:START_MONTH_INDEX]
    SIMULATION_PARAMS[product]['modifiers'] = reordered_modifiers

logger.info("✅ 3-2. SKU별 시뮬레이션 파라미터 설정 및 월별 가중치 재정렬이 완료되었습니다.")

# ==============================================================================
# ✨ 4. 메인 시뮬레이션 루프
# ==============================================================================
try:
    submission_df = pd.read_csv(os.path.join(PATH, 'sample_submission.csv'))
    logger.info("✅ 4. 제출용 데이터프레임 로드를 완료했습니다. 이제 시뮬레이션을 시작합니다.")
except FileNotFoundError as e:
    logger.error(f"❗️ 파일 로드 실패: {e}. 'sample_submission.csv' 파일이 있는지 확인해주세요.")
    exit()

# 시뮬레이션 설정
PERSONAS_PER_BATCH = 30
NUM_BATCHES_PER_PRODUCT = 4 # 총 120개 페르소나 생성 (30 * 4)
MAX_RETRIES_PER_BATCH = 3
submission_filename = os.path.join(PATH, f'my_submission_v5_{timestamp}.csv')

# --- 메인 시뮬레이션 루프 ---
for index, row in submission_df.iterrows():
    product_name = row['product_name']
    logger.info(f"\n==================== [ {product_name} ] 판매량 예측 시작 ====================")

    params = SIMULATION_PARAMS.get(product_name, SIMULATION_PARAMS['default'])
    
    # --- 페르소나 생성 ---
    product_personas = []
    for i in range(NUM_BATCHES_PER_PRODUCT):
        for attempt in range(MAX_RETRIES_PER_BATCH):
            try:
                prompt = create_product_specific_prompt(product_name, PERSONAS_PER_BATCH)
                logger.info(f" ⏳ 배치 {i+1}/{NUM_BATCHES_PER_PRODUCT} API 호출 중... (시도 {attempt+1})")
                response = model.generate_content(prompt)
                
                json_text = extract_json_from_response(response.text)
                if not json_text:
                    raise ValueError("응답에서 JSON을 찾을 수 없습니다.")
                
                batch_personas = json.loads(json_text)
                product_personas.extend(batch_personas)
                logger.info(f" ✅ 배치 {i+1} 생성 완료! ({len(batch_personas)}명 추가)")
                time.sleep(20) # API 과호출 방지
                break
            except Exception as e:
                logger.warning(f" ❗️ 배치 {i+1} 시도 {attempt+1} 실패: {e}")
                if attempt < MAX_RETRIES_PER_BATCH - 1:
                    logger.info(" 20초 후 재시도합니다...")
                    time.sleep(20)
                else:
                    logger.error(f" ❌ 배치 {i+1} 생성 최종 실패.")
    
    if not product_personas:
        logger.error(f" 🚫 페르소나 생성에 실패하여 [ {product_name} ]의 판매량을 0으로 처리합니다.")
        submission_df.iloc[index, 1:] = [0] * 12
        continue

    # --- 시뮬레이션 실행 ---
    logger.info(f"--- [ {product_name} ] ABM & Bass Model 시뮬레이션 시작 ---")
    market_sim = MarketSimulation(
        personas=product_personas,
        tam=params['tam'],
        market_share=params['market_share'],
        modifiers=params['modifiers']
    )
    
    monthly_sales = market_sim.run_simulation(months=12)
    
    # ⭐ 수정/추가된 부분 2: 출시일 이전 기간의 판매량을 0으로 처리하는 로직 추가
    if product_name in LAUNCH_DATES:
        launch_year, launch_month = LAUNCH_DATES[product_name]
        logger.info(f"    - ⚠️ 출시 예정 제품 ({launch_year}년 {launch_month}월). 출시일 이전 판매량을 0으로 조정합니다.")
        
        # 예측 기간은 2024년 7월부터 12개월
        for month_index in range(12):
            # 현재 예측 월 계산 (2024년 7월 = 0)
            current_month = 7 + month_index
            current_year = 2024
            
            if current_month > 12:
                current_month -= 12
                current_year = 2025
            
            # 출시일 이전인지 확인
            is_before_launch = (current_year < launch_year) or \
                               (current_year == launch_year and current_month < launch_month)

            if is_before_launch:
                monthly_sales[month_index] = 0 # 출시 전이므로 판매량을 0으로 설정

    submission_df.iloc[index, 1:] = monthly_sales
    logger.info(f"📈 [ {product_name} ] 12개월 판매량 예측 완료!")
    logger.info(f"   - 예측 판매량: {monthly_sales}")

    if index < len(submission_df) - 1:
        logger.info("🕒 다음 제품 분석 전, API 할당량 준수를 위해 60초간 대기합니다...")
        time.sleep(60)

# --- 최종 파일 저장 ---
submission_df.to_csv(submission_filename, index=False, encoding='utf-8-sig')
logger.info(f"\n\n🎉🎉🎉 모든 제품의 시뮬레이션이 완료되었습니다!")
logger.info(f"✅ 최종 제출 파일 '{submission_filename}' 생성이 완료되었습니다.")
logger.info(f"✅ 상세 로그는 '{log_filename}' 파일에 저장되었습니다.")

"""
제품 카테고리	관련 제품	핵심 타겟 고객	tam 추정 근거	추천 tam 값
그릭 요거트	덴마크 하이그릭요거트	20-40대 여성, 건강/자기관리 관심층	대한민국 20-40대 인구 약 1,900만 명 중, 건강 및 프리미엄 식품에 꾸준히 지출하는 핵심 소비자층을 약 30%로 추정.	6,000,000
가미 참치캔	동원맛참 4종	10-30대 남녀, 편의성/새로운 맛 추구	참치캔은 전 연령대가 소비하는 국민 식품. 10-30대 인구(약 1,800만 명)를 핵심 타겟으로 하되, 40대 이상 확장 가능성을 고려하여 대한민국 전체 가구 수를 기반으로 한 잠재 시장 규모 설정.	20,000,000
캔햄	리챔 오믈렛햄 2종	30-50대 주부, 3인 이상 가구	캔햄은 명절 등 특정 시즌 수요가 크지만, '오믈렛햄'은 아이들 반찬 등 일상 요리 목적이 뚜렷. 자녀가 있는 30-50대 가구(약 300~400만 가구)를 핵심 타겟으로 설정.	3,000,000
액상 조미료	동원참치액 6종	30-60대 주부 및 요리 애호가	집에서 요리를 즐겨 하는 인구는 한정적. '국/탕/찌개' 등 한식 조리에 관심이 많은 핵심 '집밥' 인구를 약 250만 명으로 추정. 이는 전문 조미료 시장의 핵심 고객 규모와 유사.	2,500,000
RTD 컵커피	소화가 잘되는 우유 라떼 2종	20-50대 직장인/학생, 유당불내증 우려	RTD 커피 시장은 성인 대부분이 잠재 고객. 대한민국 20-50대 인구 약 2,700만 명 중, 일 1회 이상 커피를 마시는 인구를 약 70%로 추정하여 tam 설정. '락토프리'는 이 거대 시장 내 특정 니즈를 공략.	20,000,000
"""