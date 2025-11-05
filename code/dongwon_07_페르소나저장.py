import google.generativeai as genai
import pandas as pd
import json
import os
import time
import re
import logging
import numpy as np

PATH = './09_dongwon/'
os.makedirs(PATH, exist_ok=True)
PERSONA_CACHE_PATH = os.path.join(PATH, 'personas')
os.makedirs(PERSONA_CACHE_PATH, exist_ok=True)

timestamp = time.strftime("%Y%m%d_%H%M%S")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

if not logger.handlers:
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_filename = os.path.join(PATH, f'final_log_{timestamp}.log')
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

logger.info("로깅 설정 완료")

USE_API_TO_GENERATE_PERSONAS = False
if USE_API_TO_GENERATE_PERSONAS:
    try:
        genai.configure(api_key="a")
        model = genai.GenerativeModel('models/gemini-2.0-flash')
        logger.info("Gemini API 키 설정 완료")
    except Exception as e:
        logger.error(f"API 키 설정 오류: {e}")
        USE_API_TO_GENERATE_PERSONAS = False
        logger.warning("캐시 사용 모드로 전환")

def extract_json_from_response(text):
    match = re.search(r'```json\s*(\[.*\])\s*```', text, re.DOTALL)
    if match:
        return match.group(1)
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        return match.group(0)
    return None

def sanitize_filename(name):
    """파일명으로 사용할 수 없는 문자를 '_'로 변경"""
    return re.sub(r'[\\/*?:"<>|]', '_', name)
def create_product_specific_prompt(product_name, num_personas=30):
    # (기존 코드와 동일)
    target_customer_profile = "일반적인 대한민국 소비자"
    if '하이그릭요거트' in product_name:
        target_customer_profile = "20-40대 여성으로, 건강과 자기관리에 관심이 많고, 소득 수준은 중상 이상인 1인 가구. 주로 온라인 채널을 통해 건강식품을 구매하는 경향이 있음."
    elif '맛참' in product_name:
        target_customer_profile = "20-40대 남녀로, 편의성과 새로운 맛을 추구하며 유튜브, 인스타그램 등 소셜 미디어에 익숙하고 편의점이나 온라인에서 간편하게 식사를 해결하려는 1인 가구 학생 및 직장인. 30-50대 주부로, 3인 이상 가구의 식사를 책임지고 있음. 대형마트에서 장을 보며, 명절 등 특별한 날에 가족을 위한 요리를 준비하는 것을 중요하게 생각함."
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
    logger.info(f"타겟 프로필 설정: {target_customer_profile}")
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

logger.info("페르소나 생성 함수 준비 완료")
class PersonaAgent:
    def __init__(self, persona_data):
        self.id = persona_data.get('persona_id', 'N/A')
        self.attributes = persona_data.get('attributes', {})
        self.base_purchase_rate = persona_data.get('base_purchase_frequency_per_month', 0) / 30.0
        self.state = 'Unaware'
        self.p_churn = 0.05

    def update_state(self, p_innovation, p_imitation, adoption_rate):
        if self.state == 'Unaware':
            prob_aware = p_innovation + p_imitation * adoption_rate
            if np.random.rand() < prob_aware:
                self.state = 'Active'
        elif self.state == 'Active':
            if np.random.rand() < self.p_churn:
                self.state = 'Churned'

    def attempt_purchase(self, month_modifier):
        if self.state == 'Active':
            monthly_purchase_prob = (1 - (1 - self.base_purchase_rate)**30) * month_modifier
            if np.random.rand() < monthly_purchase_prob:
                return 1
        return 0

class MarketSimulation:
    def __init__(self, personas, tam, market_share, modifiers, initial_adoption_rate=0.1):
        self.agents = [PersonaAgent(p) for p in personas if p]
        self.potential_market_size = int(tam * market_share)
        self.modifiers = modifiers
        self.p_innovation = 0.01
        self.q_imitation = 0.38
        
        num_initial_adopters = int(len(self.agents) * initial_adoption_rate)
        np.random.shuffle(self.agents)
        for i in range(num_initial_adopters):
            if i < len(self.agents):
                self.agents[i].state = 'Active'
        
        self.adopters = int(self.potential_market_size * initial_adoption_rate)
        logger.info(f"시뮬레이션 시작. 초기 채택률: {initial_adoption_rate*100:.1f}%, 초기 활성 고객 수: {self.adopters}")

    def run_simulation(self, months=12):
        monthly_sales_results = []
        num_agents = len(self.agents)

        if num_agents == 0:
            logger.warning("에이전트 없음. 0 반환")
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
            
            sample_purchase_rate = current_month_sales / num_agents if num_agents > 0 else 0
            extrapolated_sales = sample_purchase_rate * self.potential_market_size
            monthly_sales_results.append(int(extrapolated_sales))
            
            self.adopters = (active_agents_count / num_agents) * self.potential_market_size if num_agents > 0 else 0

        return monthly_sales_results

logger.info("시뮬레이션 클래스 준비 완료")
ESTABLISHED_PRODUCTS = [
    '동원맛참 고소참기름 135g', '동원맛참 고소참기름 90g', '동원맛참 매콤참기름 135g', '동원맛참 매콤참기름 90g',
    '동원참치액 순 500g', '동원참치액 순 900g', '동원참치액 진 500g', '동원참치액 진 900g',
    '프리미엄 동원참치액 500g', '프리미엄 동원참치액 900g'
]

NEW_PRODUCT_LAUNCH_DATES = {
    '덴마크 하이그릭요거트 400g': (2025, 2),
    '리챔 오믈레햄 200g': (2025, 5),
    '리챔 오믈레햄 340g': (2025, 5),
    '소화가 잘되는 우유로 만든 바닐라라떼 250mL': (2025, 2),
    '소화가 잘되는 우유로 만든 카페라떼 250mL': (2025, 2)
}
logger.info("기존/신제품 정보 설정 완료")

HOLIDAY_MODIFIERS = {
    'seollal_chuseok': [1.8, 5.5, 1.0, 0.9, 1.0, 1.1, 1.2, 2.5, 6.5, 1.0, 1.0, 1.1],
    'seollal_chuseok_sub': [1.3, 3.0, 1.0, 1.1, 1.2, 1.1, 1.2, 1.8, 3.5, 1.1, 1.0, 1.3]
}
DEFAULT_MODIFIERS = [1.0] * 12
SIMULATION_PARAMS = {
    '덴마크 하이그릭요거트 400g': {'tam': 6000000, 'market_share': 0.05, 'modifiers': [1.2, 1.2, 1.2, 1.4, 1.4, 1.3, 1.2, 1.2, 0.9, 0.8, 0.7, 0.7]},
    '동원맛참 고소참기름 135g': {'tam': 20000000, 'market_share': 0.07, 'modifiers': [0.9, 0.8, 1.2, 1.2, 1.2, 1.1, 1.2, 1.3, 0.9, 1.1, 1.0, 1.1]},
    '동원맛참 고소참기름 90g':  {'tam': 20000000, 'market_share': 0.05, 'modifiers': [0.9, 0.8, 1.2, 1.2, 1.2, 1.1, 1.2, 1.3, 0.9, 1.1, 1.0, 1.1]},
    '동원맛참 매콤참기름 135g': {'tam': 20000000, 'market_share': 0.05, 'modifiers': [0.9, 0.8, 1.2, 1.2, 1.2, 1.1, 1.2, 1.3, 0.9, 1.1, 1.0, 1.1]},
    '동원맛참 매콤참기름 90g':  {'tam': 20000000, 'market_share': 0.03, 'modifiers': [0.9, 0.8, 1.2, 1.2, 1.2, 1.1, 1.2, 1.3, 0.9, 1.1, 1.0, 1.1]},
    '리챔 오믈레햄 200g': {'tam': 3000000, 'market_share': 0.3, 'modifiers': HOLIDAY_MODIFIERS['seollal_chuseok']},
    '리챔 오믈레햄 340g': {'tam': 3000000, 'market_share': 0.5, 'modifiers': HOLIDAY_MODIFIERS['seollal_chuseok']},
    '동원참치액 순 500g':  {'tam': 2500000, 'market_share': 0.075, 'modifiers': HOLIDAY_MODIFIERS['seollal_chuseok_sub']},
    '동원참치액 순 900g':  {'tam': 2500000, 'market_share': 0.060, 'modifiers': HOLIDAY_MODIFIERS['seollal_chuseok_sub']},
    '동원참치액 진 500g':  {'tam': 2500000, 'market_share': 0.090, 'modifiers': HOLIDAY_MODIFIERS['seollal_chuseok_sub']},
    '동원참치액 진 900g':  {'tam': 2500000, 'market_share': 0.075, 'modifiers': HOLIDAY_MODIFIERS['seollal_chuseok_sub']},
    '프리미엄 동원참치액 500g': {'tam': 2500000, 'market_share': 0.05, 'modifiers': HOLIDAY_MODIFIERS['seollal_chuseok_sub']},
    '프리미엄 동원참치액 900g': {'tam': 2500000, 'market_share': 0.02, 'modifiers': HOLIDAY_MODIFIERS['seollal_chuseok_sub']},
    '소화가 잘되는 우유로 만든 바닐라라떼 250mL': {'tam': 15000000, 'market_share': 0.08, 'modifiers': [1.0, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.0, 1.1, 1.1, 1.1, 1.0]},
    '소화가 잘되는 우유로 만든 카페라떼 250mL':  {'tam': 15000000, 'market_share': 0.12, 'modifiers': [1.0, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.0, 1.1, 1.1, 1.1, 1.0]},
    'default': {'tam': 10000000, 'market_share': 0.10, 'modifiers': DEFAULT_MODIFIERS}
}
START_MONTH_INDEX = 6
for product, params in SIMULATION_PARAMS.items():
    original_modifiers = params['modifiers']
    reordered_modifiers = original_modifiers[START_MONTH_INDEX:] + original_modifiers[:START_MONTH_INDEX]
    SIMULATION_PARAMS[product]['modifiers'] = reordered_modifiers

logger.info("SKU 파라미터 설정 완료")

try:
    submission_df = pd.read_csv(os.path.join(PATH, 'sample_submission.csv'))
    logger.info("제출용 데이터프레임 로드 완료")
except FileNotFoundError:
    logger.error("sample_submission.csv 파일을 찾을 수 없습니다")
    exit()

PERSONAS_PER_BATCH = 30
NUM_BATCHES_PER_PRODUCT = 20
MAX_RETRIES_PER_BATCH = 3
submission_filename = os.path.join(PATH, f'submission_final_{timestamp}.csv')

for index, row in submission_df.iterrows():
    product_name = row['product_name']
    logger.info(f"\n==================== [ {product_name} ] 판매량 예측 시작 ====================")

    params = SIMULATION_PARAMS.get(product_name, SIMULATION_PARAMS['default'])
    
    product_personas = []
    safe_product_name = sanitize_filename(product_name)
    persona_cache_file = os.path.join(PERSONA_CACHE_PATH, f'{safe_product_name}_personas.json')

    if not USE_API_TO_GENERATE_PERSONAS and os.path.exists(persona_cache_file):
        try:
            with open(persona_cache_file, 'r', encoding='utf-8') as f:
                product_personas = json.load(f)
            logger.info(f"캐시에서 페르소나 {len(product_personas)}명 로드 완료")
        except Exception as e:
            logger.warning(f"캐시 파일 로드 오류: {e}. API를 통해 재생성 시도")
            product_personas = []
    if not product_personas:
        if USE_API_TO_GENERATE_PERSONAS:
            logger.info("Gemini API로 페르소나 생성 시작")
            for i in range(NUM_BATCHES_PER_PRODUCT):
                for attempt in range(MAX_RETRIES_PER_BATCH):
                    try:
                        prompt = create_product_specific_prompt(product_name, PERSONAS_PER_BATCH)
                        logger.info(f"배치 {i+1}/{NUM_BATCHES_PER_PRODUCT} API 호출 중... (시도 {attempt+1})")
                        response = model.generate_content(prompt)
                        
                        json_text = extract_json_from_response(response.text)
                        if not json_text:
                            raise ValueError("응답에서 JSON을 찾을 수 없습니다")
                        
                        batch_personas = json.loads(json_text)
                        product_personas.extend(batch_personas)
                        logger.info(f"배치 {i+1} 생성 완료 ({len(batch_personas)}명 추가)")
                        time.sleep(20)
                        break
                    except Exception as e:
                        logger.warning(f"배치 {i+1} 시도 {attempt+1} 실패: {e}")
                        if attempt < MAX_RETRIES_PER_BATCH - 1:
                            logger.info("20초 후 재시도")
                            time.sleep(20)
                        else:
                            logger.error(f"배치 {i+1} 생성 최종 실패")
            
            if product_personas:
                try:
                    with open(persona_cache_file, 'w', encoding='utf-8') as f:
                        json.dump(product_personas, f, ensure_ascii=False, indent=4)
                    logger.info(f"생성된 페르소나 {len(product_personas)}명 저장 완료")
                except Exception as e:
                    logger.error(f"페르소나 캐시 파일 저장 오류: {e}")

        else:
             logger.warning(f"캐시 파일 없음: {persona_cache_file}. 이 제품은 건너뜀")

    if not product_personas:
        logger.error(f"페르소나 데이터 없음. {product_name} 판매량을 0으로 처리")
        submission_df.iloc[index, 1:] = [0] * 12
        continue
    if product_name in ESTABLISHED_PRODUCTS:
        initial_rate = np.random.uniform(0.4, 0.6)
    else:
        initial_rate = 1.0

    logger.info(f"{product_name} ABM & Bass Model 시뮬레이션 시작")
    market_sim = MarketSimulation(
        personas=product_personas,
        tam=params['tam'],
        market_share=params['market_share'],
        modifiers=params['modifiers'],
        initial_adoption_rate=initial_rate
    )
    
    monthly_sales = market_sim.run_simulation(months=12)
    
    if product_name in NEW_PRODUCT_LAUNCH_DATES:
        launch_year, launch_month = NEW_PRODUCT_LAUNCH_DATES[product_name]
        logger.info(f"신제품 ({launch_year}년 {launch_month}월 출시). 출시일 이전 판매량을 0으로 조정")
        
        for month_index in range(12):
            current_month = 7 + month_index
            current_year = 2024
            if current_month > 12:
                current_month -= 12
                current_year = 2025
            
            is_before_launch = (current_year < launch_year) or \
                               (current_year == launch_year and current_month < launch_month)

            if is_before_launch:
                monthly_sales[month_index] = 0

    submission_df.iloc[index, 1:] = monthly_sales
    logger.info(f"{product_name} 12개월 판매량 예측 완료")
    logger.info(f"최종 예측 판매량: {monthly_sales}")

    if index < len(submission_df) - 1:
        wait_time = 60 if USE_API_TO_GENERATE_PERSONAS else 1
        logger.info(f"다음 제품 분석 전 {wait_time}초 대기")
        time.sleep(wait_time)

submission_df.to_csv(submission_filename, index=False, encoding='utf-8-sig')
logger.info(f"\n모든 제품의 시뮬레이션 완료")
logger.info(f"최종 제출 파일 '{submission_filename}' 생성 완료")
logger.info(f"상세 로그는 '{log_filename}' 파일에 저장됨")