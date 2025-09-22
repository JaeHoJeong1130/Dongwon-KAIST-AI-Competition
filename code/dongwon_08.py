# ==============================================================================
# 사전 준비: 라이브러리 임포트 및 API 키 설정
# ==============================================================================
import google.generativeai as genai
import pandas as pd
import json
import os
import time
import re
import logging
import numpy as np

# 랜덤 고정
np.random.seed(22917)

# 경로 설정
PATH = './competition/09_dongwon/'
os.makedirs(PATH, exist_ok=True)
# 페르소나 JSON 파일을 저장할 캐시 폴더 생성
PERSONA_CACHE_PATH = os.path.join(PATH, 'personas')
os.makedirs(PERSONA_CACHE_PATH, exist_ok=True)

# ======================= 로거 설정 =======================
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

logger.info("로깅 설정이 완료되었습니다.")
# ===============================================================

##################################################################################
# 0. 실행 모드 설정 (가장 중요한 부분!)
# ------------------------------------------------------------------------------
# True : API를 호출하여 페르소나를 '새로 생성'하고 JSON 파일로 저장합니다. (시간/비용 발생)
# False: 기존에 저장된 JSON 파일을 '불러와서' 사용합니다. (빠른 테스트용, API 호출 X)
# ------------------------------------------------------------------------------
# USE_API_TO_GENERATE_PERSONAS = True
USE_API_TO_GENERATE_PERSONAS = False
##################################################################################

if USE_API_TO_GENERATE_PERSONAS:
    try:
        genai.configure(api_key="aaa") 
        model = genai.GenerativeModel('models/gemini-2.5-flash')
        logger.info("Gemini API 키가 설정되었습니다. [API 모드]")
    except Exception as e:
        logger.error(f"[ERROR] API 키 설정 중 오류가 발생했습니다: {e}")
        # API 키가 없으면 API 모드를 강제로 비활성화
        USE_API_TO_GENERATE_PERSONAS = False
        logger.warning("[ERROR] API 사용이 불가능하여 [캐시 사용 모드]로 강제 전환합니다.")

# 헬퍼 함수
def extract_json_from_response(text):
    match = re.search(r'```json\s*(\[.*\])\s*```', text, re.DOTALL)
    if match:
        return match.group(1)
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        return match.group(0)
    return None

# 파일명으로 사용하기 안전한 문자열로 변환하는 함수
def sanitize_filename(name):
    """제품명에서 파일명으로 사용할 수 없는 문자를 '_'로 변경합니다."""
    return re.sub(r'[\\/*?:"<>|]', '_', name)

# ==============================================================================
# 페르소나 생성 프롬프트 함수
# ==============================================================================
def create_product_specific_prompt(product_name, num_personas=30):
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

logger.info("페르소나 생성 함수가 준비되었습니다.")


# ==============================================================================
# 에이전트 및 시장 시뮬레이션 클래스
# ==============================================================================
class PersonaAgent:
    def __init__(self, persona_data):
        self.id = persona_data.get('persona_id', 'N/A') # 고유 id 저장
        self.attributes = persona_data.get('attributes', {}) # 특성 저장
        self.base_purchase_rate = persona_data.get('base_purchase_frequency_per_month', 0) / 30.0 # 한달에 몇개 구매할지를 하루당 구매확률로 변환
        self.state = 'Unaware' # 에이전트는 비인지 상태에서 시작
        self.p_churn = 0.05 # 활성일때 한달에 5% 확률로 탈주

    def update_state(self, p_innovation, p_imitation, adoption_rate):
        if self.state == 'Unaware':
            # 배스 확산 방정식
            # prob_aware = 광고와 같은 외부 요인으로 스스로 제품을 인지할 확률 + 모방계수 * adoption_rate
            prob_aware = p_innovation + p_imitation * adoption_rate
            if np.random.rand() < prob_aware: # prob_aware가 랜덤보다 크면 활성화
                self.state = 'Active'
        elif self.state == 'Active':
            if np.random.rand() < self.p_churn: # 랜덤보다 5%가 크면 탈주
                self.state = 'Churned'

    def attempt_purchase(self, month_modifier):
        if self.state == 'Active':
            # (1 - self.base_purchase_rate) : 하루동안 구매하지 않을 확률 -> 30번 제곱하면 한달동안 구매안할 확률
            # 1 - 위의 식 : 한달동안 한번이상 구매할 확률
            # 한달동안 한번이상 구매할 확률 * 월별 계수
            monthly_purchase_prob = (1 - (1 - self.base_purchase_rate)**30) * month_modifier
            if np.random.rand() < monthly_purchase_prob: # 랜덤보다 크면 구매 (1)
                return 1
        return 0 # 구매X (0)

class MarketSimulation:
    def __init__(self, personas, tam, market_share, modifiers, initial_adoption_rate=0.1):
        self.agents = [PersonaAgent(p) for p in personas if p] # PersonaAgent객체들 생성해서 리스트로 저장
        self.potential_market_size = int(tam * market_share)
        self.modifiers = modifiers
        self.p_innovation = 0.01
        self.q_imitation = 0.38 # 모방계수
        
        num_initial_adopters = int(len(self.agents) * initial_adoption_rate)
        np.random.shuffle(self.agents)
        for i in range(num_initial_adopters):
            if i < len(self.agents):
                self.agents[i].state = 'Active'
        
        self.adopters = int(self.potential_market_size * initial_adoption_rate)
        logger.info(f"   - 시뮬레이션 시작. 초기 채택률: {initial_adoption_rate*100:.1f}%, 초기 활성 고객 수(추정): {self.adopters}")

    def run_simulation(self, months=12):
        monthly_sales_results = []
        num_agents = len(self.agents)

        if num_agents == 0:
            logger.warning("[ERROR] 에이전트가 없어 시뮬레이션을 진행할 수 없습니다. 0을 반환합니다.")
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

logger.info("시뮬레이션 클래스가 준비되었습니다.")

# 회사차원 명절 판매량 추가 함수
def apply_holiday_gift_set_boost(product_name, monthly_sales, logger):
    """
    특정 제품에 대해 명절(2월, 9월) 회사차원 선물 세트 판매량을 추가합니다.
    - product_name (str): 제품명
    - monthly_sales (list): 시뮬레이션으로 예측된 12개월 판매량
    - logger (Logger): 로깅 객체
    Returns:
    """
    TARGET_PRODUCTS = ['동원참치액 순 500g', '동원참치액 진 500g', '동원참치액 순 900g', '동원참치액 진 900g']

    # 월별 판매량 설정 (2월: 설, 9월: 추석)
    #   - 이 값들은 예상되는 판매량이므로, 실제 데이터나 인사이트를 바탕으로 조정 가능합니다.
    #   - 약간의 무작위성을 부여하여 현실성을 높입니다.
    GIFT_SALES_CONFIG = {
        '동원참치액 순 500g': {
            'feb_pre_sales': int(np.random.uniform(30000, 33000)), # 1월(설)
            'sep_pre_sales': int(np.random.uniform(30000, 33000)), # 8월(추석)
            'feb_sales': int(np.random.uniform(100000, 110000)), # 2월(설)
            'sep_sales': int(np.random.uniform(100000, 110000)) # 9월(추석)
        },
        '동원참치액 진 500g': {
            'feb_pre_sales': int(np.random.uniform(30000, 33000)), # 1월(설)
            'sep_pre_sales': int(np.random.uniform(30000, 33000)), # 8월(추석)
            'feb_sales': int(np.random.uniform(100000, 110000)), # 2월(설)
            'sep_sales': int(np.random.uniform(100000, 110000)) # 9월(추석)
        },
        '동원참치액 순 900g': {
            'feb_pre_sales': int(np.random.uniform(20000, 22000)), # 1월(설)
            'sep_pre_sales': int(np.random.uniform(20000, 22000)), # 8월(추석)
            'feb_sales': int(np.random.uniform(60000, 66000)), # 2월(설)
            'sep_sales': int(np.random.uniform(60000, 66000)) # 9월(추석)
        },
        '동원참치액 진 900g': {
            'feb_pre_sales': int(np.random.uniform(20000, 22000)), # 1월(설)
            'sep_pre_sales': int(np.random.uniform(20000, 22000)), # 8월(추석)
            'feb_sales': int(np.random.uniform(60000, 66000)), # 2월(설)
            'sep_sales': int(np.random.uniform(60000, 66000)) # 9월(추석)
        }
    }

    # 현재 제품이 대상 제품인지 확인
    target_key = None
    for key in TARGET_PRODUCTS:
        if key in product_name:
            target_key = key
            break

    # 대상 제품이 아니면 원본 판매량 그대로 반환
    if not target_key:
        return monthly_sales

    logger.info(f"--- '{product_name}'에 대한 명절 선물세트 판매량 추가 시작 ---")

    # 예측 기간(24년 7월 ~ 25년 6월)에 해당하는 월 인덱스
    # 9월 (24년): index 2
    # 2월 (25년): index 7
    sep_pre_index = 1
    feb_pre_index = 6
    sep_index = 2
    feb_index = 7
    
    # 8월(추석) 판매량 추가
    sep_pre_boost = GIFT_SALES_CONFIG[target_key]['sep_pre_sales']
    monthly_sales[sep_pre_index] += sep_pre_boost
    logger.info(f"  - 9월(추석): 판매량 {sep_pre_boost}개 추가. (기존: {monthly_sales[sep_pre_index] - sep_pre_boost} -> 변경: {monthly_sales[sep_pre_index]})")

    # 9월(추석) 판매량 추가
    sep_boost = GIFT_SALES_CONFIG[target_key]['sep_sales']
    monthly_sales[sep_index] += sep_boost
    logger.info(f"  - 9월(추석): 판매량 {sep_boost}개 추가. (기존: {monthly_sales[sep_index] - sep_boost} -> 변경: {monthly_sales[sep_index]})")

    # 1월(설) 판매량 추가
    feb_pre_boost = GIFT_SALES_CONFIG[target_key]['feb_pre_sales']
    monthly_sales[feb_pre_index] += feb_pre_boost
    logger.info(f"  - 2월(설): 판매량 {feb_pre_boost}개 추가. (기존: {monthly_sales[feb_pre_index] - feb_pre_boost} -> 변경: {monthly_sales[feb_pre_index]})")

    # 2월(설) 판매량 추가
    feb_boost = GIFT_SALES_CONFIG[target_key]['feb_sales']
    monthly_sales[feb_index] += feb_boost
    logger.info(f"  - 2월(설): 판매량 {feb_boost}개 추가. (기존: {monthly_sales[feb_index] - feb_boost} -> 변경: {monthly_sales[feb_index]})")

    return monthly_sales

# 최종 판매량 숫자 정리 함수
def format_sales_numbers(product_name, monthly_sales, logger):
    """
    제품군별 규칙에 따라 최종 판매량 숫자를 반올림하여 정리합니다.
    - product_name (str): 제품명
    - monthly_sales (list): 모든 계산이 끝난 12개월 판매량
    - logger (Logger): 로깅 객체
    Returns:
    - list: 반올림 규칙이 적용된 12개월 판매량
    """
    rounding_base = 0
    rule_desc = "규칙 없음"

    # 1000의 자리에서 반올림
    if any(keyword in product_name for keyword in ['리챔 오믈레햄', '동원참치액 순', '동원참치액 진', '소화가 잘되는 우유로 만든']):
        rounding_base = 1000
        rule_desc = "100의 자리에서 반올림"
    # 500 단위로 반올림
    elif any(keyword in product_name for keyword in ['동원맛참', '덴마크 하이그릭요거트']):
        rounding_base = 500
        rule_desc = "500 단위로 반올림"
    # 200 단위로 반올림
    elif '프리미엄 동원참치액' in product_name:
        rounding_base = 200
        rule_desc = "200 단위로 반올림"

    # 적용할 규칙이 있는 경우에만 실행
    if rounding_base > 0:
        # 리스트의 각 판매량에 대해 반올림 규칙 적용
        # (sale / N)을 반올림하고 다시 N을 곱하는 방식 사용
        formatted_sales = [int(round(sale / rounding_base) * rounding_base) for sale in monthly_sales]
        logger.info(f"  - [숫자 정리] '{rule_desc}' 규칙을 적용합니다.")
        return formatted_sales
    else:
        # 적용할 규칙이 없으면 원본 그대로 반환
        return monthly_sales

# ==============================================================================
# 시뮬레이션 파라미터 정의
# ==============================================================================
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
logger.info("기존/신제품 정보가 설정되었습니다.")

DEFAULT_MODIFIERS = [1.0] * 12
SIMULATION_PARAMS = {
    '덴마크 하이그릭요거트 400g': {'tam': 8000000, 'market_share': 0.04, 'modifiers': [1.2, 1.1, 1.6, 1.8, 1.8, 1.9, 1.2, 1.2, 0.9, 0.8, 0.9, 0.9]},
    '동원맛참 고소참기름 135g': {'tam': 20000000, 'market_share': 0.043, 'modifiers': [0.7, 0.7, 0.9, 1.0, 1.0, 1.2, 1.4, 1.1, 0.6, 0.7, 0.6, 0.7]},
    '동원맛참 고소참기름 90g':  {'tam': 20000000, 'market_share': 0.055, 'modifiers': [0.7, 0.7, 0.9, 1.0, 1.0, 1.2, 1.4, 1.1, 0.6, 0.7, 0.6, 0.7]},
    '동원맛참 매콤참기름 135g': {'tam': 20000000, 'market_share': 0.033, 'modifiers': [0.7, 0.7, 0.9, 1.0, 1.0, 1.2, 1.4, 1.1, 0.6, 0.7, 0.6, 0.7]},
    '동원맛참 매콤참기름 90g':  {'tam': 20000000, 'market_share': 0.04, 'modifiers': [0.7, 0.7, 0.9, 1.0, 1.0, 1.2, 1.4, 1.1, 0.6, 0.7, 0.6, 0.7]},
    '리챔 오믈레햄 200g': {'tam': 5500000, 'market_share': 0.065, 'modifiers': [0.9, 0.8, 1.2, 1.2, 1.2, 1.1, 1.2, 1.3, 0.9, 1.1, 1.0, 1.1]},
    '리챔 오믈레햄 340g': {'tam': 5500000, 'market_share': 0.04, 'modifiers': [0.9, 0.8, 1.2, 1.2, 1.2, 1.1, 1.2, 1.3, 0.9, 1.1, 1.0, 1.1]},
    '동원참치액 순 500g':  {'tam': 2500000, 'market_share': 0.042, 'modifiers': [2.4, 2.2, 0.9, 1.0, 1.1, 1.0, 1.0, 2.4, 2.2, 0.8, 0.9, 0.8]},
    '동원참치액 순 900g':  {'tam': 2500000, 'market_share': 0.019, 'modifiers': [2.4, 2.2, 0.9, 1.0, 1.1, 1.0, 1.0, 2.4, 2.2, 0.8, 0.9, 0.8]},
    '동원참치액 진 500g':  {'tam': 2500000, 'market_share': 0.06, 'modifiers': [2.4, 2.2, 0.9, 1.0, 1.1, 1.0, 1.0, 2.4, 2.2, 0.8, 0.9, 0.8]},
    '동원참치액 진 900g':  {'tam': 2500000, 'market_share': 0.025, 'modifiers': [2.4, 2.2, 0.9, 1.0, 1.1, 1.0, 1.0, 2.4, 2.2, 0.8, 0.9, 0.8]},
    '프리미엄 동원참치액 500g': {'tam': 2500000, 'market_share': 0.014, 'modifiers': [1.3, 1.0, 1.0, 1.0, 1.1, 0.9, 0.8, 0.9, 1.4, 0.9, 1.2, 1.2]},
    '프리미엄 동원참치액 900g': {'tam': 2500000, 'market_share': 0.003, 'modifiers': [1.1, 1.0, 1.0, 1.0, 1.1, 0.9, 0.9, 0.9, 1.1, 0.9, 1.2, 1.1]},
    '소화가 잘되는 우유로 만든 바닐라라떼 250mL': {'tam': 16000000, 'market_share': 0.11, 'modifiers': [0.9, 0.9, 1.0, 1.1, 1.1, 1.1, 1.2, 1.2, 1.1, 1.1, 1.1, 1.0]},
    '소화가 잘되는 우유로 만든 카페라떼 250mL':  {'tam': 16000000, 'market_share': 0.11, 'modifiers': [0.9, 0.9, 1.0, 1.1, 1.1, 1.1, 1.2, 1.2, 1.1, 1.1, 1.1, 1.0]},
    'default': {'tam': 10000000, 'market_share': 0.10, 'modifiers': DEFAULT_MODIFIERS}
}
START_MONTH_INDEX = 6
for product, params in SIMULATION_PARAMS.items():
    original_modifiers = params['modifiers']
    reordered_modifiers = original_modifiers[START_MONTH_INDEX:] + original_modifiers[:START_MONTH_INDEX]
    SIMULATION_PARAMS[product]['modifiers'] = reordered_modifiers

logger.info("파라미터 설정 및 월별 가중치 재정렬이 완료되었습니다.")


# ==============================================================================
# 메인 시뮬레이션 루프
# ==============================================================================
try:
    submission_df = pd.read_csv(os.path.join(PATH, 'sample_submission.csv'))
    logger.info("제출용 데이터프레임 로드를 완료했습니다.")
except FileNotFoundError:
    logger.error(f"[ERROR] 'sample_submission.csv' 파일을 찾을 수 없습니다.")
    exit()

PERSONAS_PER_BATCH = 30
NUM_BATCHES_PER_PRODUCT = 10
MAX_RETRIES_PER_BATCH = 3
submission_filename = os.path.join(PATH, f'submission_final_{timestamp}.csv')

for index, row in submission_df.iterrows():
    product_name = row['product_name']
    logger.info(f"\n==================== [ {product_name} ] 판매량 예측 시작 ====================")

    params = SIMULATION_PARAMS.get(product_name, SIMULATION_PARAMS['default'])
    
    # 페르소나 생성 로직을 캐시 사용 여부에 따라 분기
    product_personas = []
    
    # 파일명으로 사용하기 위해 제품명을 안전하게 변환
    safe_product_name = sanitize_filename(product_name)
    persona_cache_file = os.path.join(PERSONA_CACHE_PATH, f'{safe_product_name}_personas.json')

    # 1. 캐시 사용 모드일 경우, 파일 로드를 먼저 시도
    if not USE_API_TO_GENERATE_PERSONAS and os.path.exists(persona_cache_file):
        try:
            with open(persona_cache_file, 'r', encoding='utf-8') as f:
                product_personas = json.load(f)
            logger.info(f"[캐시 사용] '{persona_cache_file}' 에서 페르소나 {len(product_personas)}명을 성공적으로 불러왔습니다.")
        except Exception as e:
            logger.warning(f"[ERROR] 캐시 파일 '{persona_cache_file}' 로드 중 오류 발생: {e}. API를 통해 재생성을 시도합니다.")
            product_personas = [] # 오류 발생 시, 리스트를 비워 아래 API 로직을 타도록 유도

    # 2. 페르소나가 비어있을 경우 (캐시를 사용하지 않거나, 캐시 파일이 없거나, 로드 실패 시)
    if not product_personas:
        if USE_API_TO_GENERATE_PERSONAS:
            logger.info("[API 모드] Gemini API를 통해 페르소나 생성을 시작")
            for i in range(NUM_BATCHES_PER_PRODUCT):
                for attempt in range(MAX_RETRIES_PER_BATCH):
                    try:
                        prompt = create_product_specific_prompt(product_name, PERSONAS_PER_BATCH)
                        logger.info(f"배치 {i+1}/{NUM_BATCHES_PER_PRODUCT} API 호출 중... (시도 {attempt+1})")
                        response = model.generate_content(prompt)
                        
                        json_text = extract_json_from_response(response.text)
                        if not json_text:
                            raise ValueError("[ERROR] 응답에서 JSON을 찾을 수 없습니다.")
                        
                        batch_personas = json.loads(json_text)
                        product_personas.extend(batch_personas)
                        logger.info(f"배치 {i+1} 생성 완료! ({len(batch_personas)}명 추가)")
                        time.sleep(20)
                        break
                    except Exception as e:
                        logger.warning(f"배치 {i+1} 시도 {attempt+1} 실패: {e}")
                        if attempt < MAX_RETRIES_PER_BATCH - 1:
                            logger.info(" 20초 후 재시도합니다...")
                            time.sleep(20)
                        else:
                            logger.error(f"[ERROR] 배치 {i+1} 생성 최종 실패.")
            
            # API로 성공적으로 생성 후, 파일로 저장
            if product_personas:
                try:
                    with open(persona_cache_file, 'w', encoding='utf-8') as f:
                        json.dump(product_personas, f, ensure_ascii=False, indent=4)
                    logger.info(f"[캐시 저장] 생성된 페르소나 {len(product_personas)}명을 '{persona_cache_file}'에 저장했습니다.")
                except Exception as e:
                    logger.error(f"[ERROR] 페르소나 캐시 파일 저장 중 오류 발생: {e}")

        else: # API 사용 안 함 & 캐시 파일도 없는 경우
             logger.warning(f"[ERROR] [캐시 없음] '{persona_cache_file}' 파일이 없습니다. 이 제품은 건너뜁니다.")
             logger.warning(f"(페르소나를 생성하려면 스크립트 상단의 USE_API_TO_GENERATE_PERSONAS를 True로 변경하세요.)")


    # 페르소나 생성에 최종 실패한 경우, 해당 제품은 0으로 처리하고 다음으로 넘어감
    if not product_personas:
        logger.error(f"페르소나 데이터가 없습니다. [ {product_name} ] 판매량을 0으로 처리합니다.")
        submission_df.iloc[index, 1:] = [0] * 12
        continue

    # --- 시뮬레이션 실행 ---
    if product_name in ESTABLISHED_PRODUCTS:
        initial_rate = np.random.uniform(0.45, 0.5)
    else:
        initial_rate = np.random.uniform(0.03, 0.05)

    logger.info(f"--- [ {product_name} ] ABM & Bass Model 시뮬레이션 시작 ---")
    market_sim = MarketSimulation(
        personas=product_personas,
        tam=params['tam'],
        market_share=params['market_share'],
        modifiers=params['modifiers'],
        initial_adoption_rate=initial_rate
    )
    
    monthly_sales = market_sim.run_simulation(months=12)
    
    monthly_sales = apply_holiday_gift_set_boost(product_name, monthly_sales, logger)
    
    if product_name in NEW_PRODUCT_LAUNCH_DATES:
        launch_year, launch_month = NEW_PRODUCT_LAUNCH_DATES[product_name]
        logger.info(f"   - 신제품 ({launch_year}년 {launch_month}월 출시). 출시일 이전 판매량을 0으로 조정합니다.")
        
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
                
    # 반올림
    monthly_sales = format_sales_numbers(product_name, monthly_sales, logger)

    submission_df.iloc[index, 1:] = monthly_sales
    logger.info(f"[ {product_name} ] 12개월 판매량 예측 완료!")
    logger.info(f"   - 최종 예측 판매량: {monthly_sales}")

    # 마지막 제품 실행 후에는 대기하지 않도록 조건 추가
    if index < len(submission_df) - 1:
        # API 모드일 때는 60초, 캐시 모드일때는 1초 대기
        wait_time = 60 if USE_API_TO_GENERATE_PERSONAS else 1
        logger.info(f"다음 제품 분석 전 {wait_time}초간 대기합니다...")
        time.sleep(wait_time)

# --- 최종 파일 저장 ---
submission_df.to_csv(submission_filename, index=False, encoding='utf-8-sig')
logger.info(f"\n\n@@@모든 제품의 시뮬레이션이 완료되었습니다@@@")
logger.info(f"최종 제출 파일 '{submission_filename}' 생성이 완료되었습니다.")
logger.info(f"상세 로그는 '{log_filename}' 파일에 저장되었습니다.")