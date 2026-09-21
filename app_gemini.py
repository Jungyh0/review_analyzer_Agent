import streamlit as st
import sqlite3
import json
import ast
import os
import operator
from typing import TypedDict, Optional, Dict, Any, Literal

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import re

from typing import TypedDict, Annotated, List, Optional, Literal, Dict, Any
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import END, START, StateGraph, MessagesState

# ==================================================
# 0. 기본 설정
# ==================================================
DB_PATH = "user_review_db.db"
ASPECT_LIST = [
    '가격', '구성', '제형 밀도', '향', '사용감', '지속력', '디자인', '만족도', '보습',
    '휴대성', '배송', '발림성', '용량', '사은품', '포장', '커버력', '피부표현', '세팅력',
    '발색', '소독효과', '흡수력', '유통기한', '유통 기한', '사이즈', '편의성', '품질',
    '색감', '제형', '피부타입', '유분감', '유효기간', '자외선 차단', '톤업', '진정',
    '크기', '피부결', '세정력', '번짐', '밀착력', '색상', '위생', '손상케어', '주름개선',
    '자극', '클렌징', '효과', '할인', '재구매', '성분', '피부'
]

# 제미나이 API 키 설정
# 방법 A) 코드에 직접 입력 (테스트용, 배포 시 권장하지 않음)
# os.environ["GOOGLE_API_KEY"] = "AIza..."
# 방법 B) api_key.txt 에서 로드 (권장)
def load_api_keys(filepath="api_key.txt"):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()

load_api_keys("api_key.txt")  # 파일 안에 GOOGLE_API_KEY=AIza... 형태로 저장

st.set_page_config(page_title="상품 리뷰 분석 Agent", layout="wide")

# 한글 폰트 설정
def set_korean_font():
    candidates = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            font_name = fm.FontProperties(fname=path).get_name()
            plt.rcParams["font.family"] = font_name
            break
    plt.rcParams["axes.unicode_minus"] = False

set_korean_font()

# ==================================================
# 1. Agent (1차 분석: analyzer / critic / supervisor)
# ==================================================
class ReviewState(TypedDict):
    messages: Annotated[list, add_messages]
    input_review: str
    max_num: int
    current_num: int
    score: int
    next_step: Optional[Literal["analyzer", "critic", "end"]]
    result: dict
    aspect_list: List[str]
    criti_score: str

GEMINI_MODEL = "gemini-3.6-flash"

visor_llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.2)
analyzer_llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.2)
criti_llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.2)


def get_text(msg) -> str:
    """최신 langchain-google-genai는 응답 content가 문자열이 아니라
    [{'type': 'text', 'text': '...'}] 같은 리스트로 올 수 있어서,
    항상 순수 문자열로 뽑아주는 헬퍼."""
    content = msg.content if hasattr(msg, "content") else msg
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("text", ""))
        return "".join(parts)
    return str(content)


def _extract_json(raw: str, fallback: dict) -> dict:
    """Gemini는 OpenAI보다 코드펜스(```json)를 더 자주 붙이는 경향이 있어
    파싱 실패 시 코드펜스 제거 후 재시도하는 fallback을 둔다."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return fallback


def analyzer_node(state: ReviewState):
    review = state.get("input_review") or ""
    aspect_list = state.get("aspect_list") or []
    score = state.get("score") or 0
    feedback = state.get("criti_score") or "없음 (최초 분석)"

    sys = """
너는 화장품 리뷰 분석 전문가야.
다음 지침에 따라 입력받은 사용자의 리뷰를 분석해줘.
만약 [수정 요청 피드백]이 제공된다면, 이전의 실수를 교정하여 결과를 다시 작성해야 해.

[지침]
1. 제시된 [후보 속성] 중에서 리뷰 내용과 일치하는 것을 추출해. **단, '성능', '효과'처럼 포괄적인 단어가 쓰였더라도 문맥상 화장품의 기능(예: 보습력)을 뜻한다면 유연하게 해석해서 해당 후보 속성으로 매칭해줘.**
2. **중요: 'aspect' 리스트에 담긴 속성의 순서는 반드시 [후보 속성]에 나열된 순서를 유지해야 해.**
3. 각 속성에 대한 만족도를 판별해서 만족/긍정은 1, 불만족/부정은 0으로 기록해.
4. 'label' 리스트의 값은 'aspect' 리스트의 순서와 1:1로 정확히 매칭되어야 해.
5. [중요: 별점 산정 규칙]
- 입력된 [별점 숫자]가 0보다 크면 해당 값을 그대로 사용해.
- 입력된 [별점 숫자]가 0이라면, 리뷰 원문의 뉘앙스를 분석하여 네가 직접 1~5점 사이의 별점을 책정해.(매우 만족: 5, 만족: 4, 보통: 3, 불만족: 2, 매우 불만족: 1)
6. 리뷰 내용이 후보 속성과 명확하게 연결되지 않는 경우에는 aspect와 label을 빈 리스트([])로 반환해.
7. 결과는 반드시 아래의 JSON 형식으로만 출력하고, 코드펜스(```)나 추가 설명은 절대 포함하지 마.

[출력 형식]
{
  "review": "리뷰 원문",
  "aspect": ["추출된 속성 리스트"],
  "label": [각 속성에 대응하는 1 또는 0 리스트],
  "score": 별점숫자
}
"""
    human = f"""
[리뷰 원문]
{review}

[후보 속성]
{aspect_list}

[별점 숫자]
{score if score else 0}

[수정 요청 피드백]
{feedback}
"""
    resp = analyzer_llm.invoke([SystemMessage(content=sys),
                                HumanMessage(content=human)])

    resp_text = get_text(resp).strip()
    fallback = {
        "review": review,
        "aspect": [],
        "label": [],
        "score": score,
    }
    data_dict = _extract_json(resp_text, fallback)

    data_dict.setdefault("review", review)
    data_dict.setdefault("aspect", [])
    data_dict.setdefault("label", [])
    data_dict.setdefault("score", score)

    return {
        "messages": [AIMessage(content=f"[Analyzer Result]\n{resp_text}")],
        "result": data_dict,
        "criti_score": ""
    }


def critic_node(state: ReviewState):
    review = state.get("input_review") or ""
    aspect_list = state.get("aspect_list") or []
    result = state.get("result") or ""
    current_num = state.get("current_num") or 0

    sys = """
너는 리뷰 분석 결과의 정확성을 검증하는 품질 관리(QA) 전문가야.
Analyzer가 추출한 결과가 리뷰 원문의 내용과 논리적으로 일치하는지 검토해줘.

#검토 지침
1. 일관성: 추출된 'aspect'가 리뷰 원문에 직접 언급되었거나, **문맥상 의미가 일치하는지 확인해 (예: '성능' -> '보습' 등 포괄적/유의어 매칭 허용).**
2. 정확성: 각 'aspect'에 매칭된 'label'(1: 만족, 0: 불만족)이 리뷰의 맥락과 맞는지 확인해.
3. 형식: 결과가 약속된 JSON 형식을 유지하고 있는지 확인해.
4. 순서: **[중요] 추출된 'aspect'들은 [후보 속성]에 나열된 상대적 순서를 유지해야 해.**

#출력
[FEEDBACK]
- ...
[VERDICT]
VERDICT: OK
또는
VERDICT: REVISE
"""
    human = f"""
[리뷰 원문]
{review}

[후보 속성]
{aspect_list}

[Analyzer의 분석 결과]
{result}
"""
    resp = criti_llm.invoke([SystemMessage(content=sys),
                             HumanMessage(content=human)])
    resp_text = get_text(resp)
    current_num = current_num + 1
    return {
        "messages": [AIMessage(content=f"[CRITIC RESULT]\n{resp_text}")],
        "criti_score": resp_text,
        "current_num": current_num
    }


def supervisor_node(state: ReviewState):
    print("-다음 경로 탐색중...-")
    if not state.get("result"):
        next_step = "analyzer"
    elif not state.get("current_num"):
        next_step = "critic"
    else:
        criti = state.get("criti_score") or ""
        if "VERDICT: OK" in criti:
            next_step = "end"
        elif "VERDICT: REVISE" in criti and state.get("current_num") <= state.get("max_num"):
            print("-!재분석이 요구됩니다.!-")
            next_step = "analyzer"
        else:
            next_step = "end"
    print(f"-{next_step}으로 이동합니다.-")
    return {"next_step": next_step}


def route_next(state: ReviewState):
    return state["next_step"]


builder = StateGraph(ReviewState)
builder.add_node("analyzer", analyzer_node)
builder.add_node("critic", critic_node)
builder.add_node("supervisor", supervisor_node)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor", route_next,
    {"analyzer": "analyzer",
     "critic": "critic",
     "end": END}
)
builder.add_edge("analyzer", "supervisor")
builder.add_edge("critic", "supervisor")

agent_app = builder.compile()


# ==================================================
# 1-2. 별점 1~2점 리뷰 피드백 Agent
#       (router → 전문가 노드들 → aggregator)
# ==================================================
class AnalyzerState(TypedDict):
    messages: Annotated[list, add_messages]
    review: str
    aspect: List[str]
    label: List[int]
    next_node: List[str]
    result_list: Annotated[List[str], operator.add]
    final_result: str


router_llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)
price_llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.7)
performance_llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.3)
Texture_llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.2)
Design_llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.7)
Service_llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=1)


def router_node(state: AnalyzerState):
    aspect = state.get("aspect") or []
    label = state.get("label") or []

    sys_msg = '''
너는 고객 리뷰의 불만 사항을 파악하고 해결할 전문 부서를 배정하는 라우터(Router) 에이전트야.
입력된 [aspect] 리스트와 [label] 리스트의 규칙을 분석하여, 불만족으로 평가된 속성을 해결할 전문가 에이전트 목록을 출력해.

[전문가 그룹 및 담당 속성]
1. price_node: 가격, 할인, 가성비, 구성, 용량
2. performance_node: 보습, 커버력, 발색, 자외선 차단, 톤업, 진정, 세정력, 손상케어, 주름개선, 클렌징, 효과, 피부결, 피부, 품질, 성분, 소독효과, 지속력, 피부표현, 세팅력
3. texture_node: 발림성, 흡수력, 사용감, 제형 밀도, 제형, 유분감, 밀착력, 번짐, 자극, 향
4. design_node: 디자인, 색감(색상), 사이즈(크기), 휴대성, 편의성, 위생, 포장
5. service_node: 배송, 만족도, 재구매, 피부타입, 사은품, 유통기한, 유통 기한, 유효기간

[분석 및 라우팅 지침]
1. 매칭 규칙: [aspect]와 [label]의 요소는 같은 인덱스로 1:1 매칭되어 있어.
2. 불만족 식별: [label] 값이 0인 인덱스를 모두 찾아.
3. 속성 추출: 찾아낸 인덱스와 동일한 위치에 있는 단어들을 [aspect]에서 추출해.
4. 에이전트 매칭: 해당 에이전트(LLM) 이름을 매칭해.
5. 중복 제거.
6. 우선순위: performance_node, texture_node가 있으면 맨 앞으로.
7. 예외 처리: 0(불만족)이 없으면 ["service_node"]를 출력해.

[출력 형식]
반드시 쌍따옴표가 포함된 올바른 JSON 리스트 형식으로만 응답해.
예) ["performance_node", "texture_node", "design_node"]
    '''
    user_msg = f"""
[aspect]
{aspect}

[label]
{label}
"""
    response = router_llm.invoke([SystemMessage(content=sys_msg),
                                  HumanMessage(content=user_msg)])
    content = get_text(response).strip()
    if content.startswith("```"):
        content = content.replace("```json", "").replace("```", "").strip()

    try:
        destination = json.loads(content)
    except json.JSONDecodeError:
        destination = ["service_node"]

    return {"next_node": destination}


def _build_expert_node(llm, role_name, area, target_attrs, persona):
    """전문가 노드 공통 빌더 (코드 중복 줄이기용)"""
    def _node(state: AnalyzerState):
        review = state.get("review") or ""
        aspect = state.get("aspect") or []
        label = state.get("label") or []

        sys_msg = f'''
너는 {persona}({role_name})야.
입력된 [aspect], [label] 데이터와 [review] 원문을 분석하여,
네가 담당하는 속성 중 불만족(0)으로 평가된 항목에 대한
구체적인 원인 분석과 맞춤형 해결 방안을 제시해.

[담당 속성 영역]
{target_attrs}

[출력 형식]
[개선 대상 속성]: 불만족 속성들을 쉼표로 구분

- 속성: 속성 이름
- 원인 분석: 리뷰 원문 기반 구체적인 이유
- 개선 방안: {area} 관점의 실질적 해결 방안
'''
        user_msg = f'''
[review]
{review}
[aspect]
{aspect}
[label]
{label}
'''
        response = llm.invoke([
            SystemMessage(content=sys_msg),
            HumanMessage(content=user_msg)
        ])
        return {
            "messages": [response],
            "result_list": [get_text(response)]
        }
    return _node


performance_node = _build_expert_node(
    performance_llm, "performance_llm", "연구원",
    "보습, 커버력, 발색, 자외선 차단, 톤업, 진정, 세정력, 손상케어, 주름개선, 클렌징, 효과, 피부결, 피부, 품질, 성분, 소독효과, 지속력, 피부표현, 세팅력",
    "제품의 '효능, 성분, 품질' 개선을 담당하는 제품 개발 전문가"
)
price_node = _build_expert_node(
    price_llm, "price_llm", "마케팅",
    "가격, 할인, 가성비, 구성, 용량",
    "제품의 '가격 경쟁력 및 경제적 가치' 개선을 담당하는 가격 전략 전문가"
)
texture_node = _build_expert_node(
    Texture_llm, "Texture_llm", "제형 연구원",
    "발림성, 흡수력, 사용감, 제형 밀도, 제형, 유분감, 밀착력, 번짐, 자극, 향",
    "제품의 '제형, 사용감, 피부 자극 및 향기' 개선을 담당하는 제형 개발 전문가"
)
design_node = _build_expert_node(
    Design_llm, "Design_llm", "디자이너",
    "디자인, 색감, 색상, 사이즈, 크기, 휴대성, 편의성, 위생, 포장",
    "제품의 '디자인 및 패키징' 개선을 담당하는 제품 디자인 전문가"
)
service_node = _build_expert_node(
    Service_llm, "Service_llm", "CS",
    "배송, 만족도, 재구매, 피부타입, 사은품, 유통기한, 유통 기한, 유효기간",
    "제품의 '배송 서비스 및 고객 경험 관리'를 담당하는 CS 전략 전문가"
)


def route_to_experts(state: AnalyzerState):
    sends = []
    next_nodes = state.get("next_node") or []
    payload = {
        "review": state.get("review"),
        "aspect": state.get("aspect"),
        "label": state.get("label")
    }
    for node_name in next_nodes:
        if node_name in ["performance_node", "price_node", "texture_node", "design_node", "service_node"]:
            sends.append(Send(node_name, payload))
        else:
            sends.append(Send("service_node", payload))
    if not sends:
        sends.append(Send("service_node", payload))
    return sends


def aggregator_node(state: AnalyzerState):
    results = state.get("result_list") or []
    if not results:
        final_answer = "분석 결과가 없습니다."
    else:
        combined_results = "\n\n".join(results)
        final_answer = f"""[종합 제품 개선 제안서]

고객님의 리뷰를 바탕으로 각 분야 전문가들이 분석한 개선 방안입니다.

{combined_results}

---
위 개선 방안이 제품 품질 향상에 도움이 되기를 바랍니다.
"""
    return {"final_result": final_answer}


fb_builder = StateGraph(AnalyzerState)
fb_builder.add_node("router_node", router_node)
fb_builder.add_node("performance_node", performance_node)
fb_builder.add_node("price_node", price_node)
fb_builder.add_node("texture_node", texture_node)
fb_builder.add_node("design_node", design_node)
fb_builder.add_node("service_node", service_node)
fb_builder.add_node("aggregator_node", aggregator_node)

fb_builder.add_edge(START, "router_node")
fb_builder.add_conditional_edges("router_node", route_to_experts,
    ["performance_node", "price_node", "texture_node", "design_node", "service_node"],
)
fb_builder.add_edge("performance_node", "aggregator_node")
fb_builder.add_edge("price_node", "aggregator_node")
fb_builder.add_edge("texture_node", "aggregator_node")
fb_builder.add_edge("design_node", "aggregator_node")
fb_builder.add_edge("service_node", "aggregator_node")
fb_builder.add_edge("aggregator_node", END)

feedback_app = fb_builder.compile()


# ==================================================
# 2. DB 함수 준비
# ==================================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS new_user_reviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        review TEXT,
        aspect TEXT,
        label TEXT,
        score INTEGER
    )
    """)
    conn.commit()
    conn.close()


def insert_data_db(state: ReviewState):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    result = state.get("result")
    cursor.execute(
        "INSERT INTO new_user_reviews (review, aspect, label, score) VALUES (?, ?, ?, ?)",
        (
            result["review"].strip(),
            json.dumps(result["aspect"], ensure_ascii=False),
            json.dumps(result["label"]),
            result["score"]
        )
    )
    conn.commit()
    conn.close()


def load_all_reviews() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM new_user_reviews ORDER BY id ASC", conn)
    conn.close()
    if df.empty:
        return df
    df["aspect_parsed"] = df["aspect"].apply(lambda x: json.loads(x) if x else [])
    df["label_parsed"] = df["label"].apply(lambda x: json.loads(x) if x else [])
    return df


def build_aspect_summary(df: pd.DataFrame, aspect_list) -> pd.DataFrame:
    rows = []
    for aspect in aspect_list:
        pos = neg = 0
        for _, r in df.iterrows():
            aspects = r["aspect_parsed"]
            labels = r["label_parsed"]
            if not aspects or not labels or len(aspects) != len(labels):
                continue
            if aspect in aspects:
                idx = aspects.index(aspect)
                if labels[idx] == 1:
                    pos += 1
                elif labels[idx] == 0:
                    neg += 1
        total = pos + neg
        ratio = (pos / total * 100) if total > 0 else 0.0
        rows.append({
            "aspect": aspect,
            "긍정": pos,
            "부정": neg,
            "언급횟수": total,
            "긍정비율(%)": round(ratio, 1),
        })
    return pd.DataFrame(rows)


# ==================================================
# 3. Streamlit UI
# ==================================================
init_db()

st.title("📊 상품 리뷰 분석 Agent 대시보드")

# --------------------------------------------------
# (상단) 리뷰 입력 / 분석 실행
# --------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("✏️ 리뷰 입력")
    review = st.text_area("리뷰 내용", placeholder="여기에 리뷰를 입력하세요. (20자 이상)")

    current_len = len(review.strip())
    if current_len < 20:
        st.caption(f"📝 현재 {current_len}자 / 최소 20자 필요")
    else:
        st.caption(f"✅ 현재 {current_len}자")

    score = st.radio("별점", [1, 2, 3, 4, 5], horizontal=True)
    run_btn = st.button("분석 실행", type="primary")

with col2:
    st.subheader("🤖 Agent 분석 결과")
    if run_btn:
        review_stripped = review.strip()
        review_len = len(review_stripped)

        if not review_stripped:
            st.warning("리뷰 내용을 입력해주세요.")
        elif review_len < 20:
            st.warning(
                f"⚠️ 리뷰는 20자 이상 작성해주세요. (현재 {review_len}자)"
            )
            st.caption(
                "정확한 분석을 위해서는 충분한 내용이 필요합니다.\n\n"
                "예: '보습력이 정말 좋고 향도 은은해서 마음에 들어요. 가격도 합리적이네요.'"
            )
        else:
            with st.spinner("Agent가 리뷰를 분석하고 있습니다..."):
                initial_state = {
                    "input_review": review,
                    "score": score,
                    "aspect_list": ASPECT_LIST,
                    "max_num": 3,
                    "current_num": 0,
                    "messages": []
                }
                final_state = agent_app.invoke(initial_state)

                result = final_state.get("result", {})
                aspects = result.get("aspect", []) or []
                labels = result.get("label", []) or []
                final_score = result.get("score", score)

                # 속성 추출에 성공한 경우에만 DB에 저장.
                # 실패한(빈 aspect/label) 리뷰까지 저장하면 사이드바 필터에
                # 걸려 화면에서 영구히 숨겨지면서 ID 번호만 낭비하게 되므로
                # 애초에 저장하지 않는다.
                saved = False
                if aspects and labels and len(aspects) == len(labels):
                    insert_data_db(final_state)
                    saved = True

            if saved:
                st.success("분석 및 저장이 완료되었습니다!")
            else:
                st.warning("⚠️ 속성 추출에 실패하여 이 리뷰는 저장되지 않았습니다. (다른 리뷰로 다시 시도해보세요)")

            st.write(f"**리뷰:** {result.get('review', 'N/A')}")
            st.write(f"**별점:** {final_score}")

            if aspects and labels and len(aspects) == len(labels):
                st.markdown("**🏷️ 속성별 평가**")
                detail_df = pd.DataFrame({
                    "속성": aspects,
                    "감성": ["😊 긍정" if l == 1 else "😞 부정" for l in labels],
                })
                st.table(detail_df)
            else:
                st.warning("⚠️ 리뷰에서 분석 대상 속성이 추출되지 않았습니다.")
                st.caption(
                    f"분석 가능한 속성: {', '.join(ASPECT_LIST)}\n\n"
                    "좀 더 구체적인 표현(예: '보습력이 좋아요', '향이 별로예요')을 포함해주세요."
                )

            # ============================================
            # 별점 1~2점이면 피드백 Agent 실행
            # ============================================
            if final_score in [1, 2] and aspects and labels:
                st.divider()
                st.markdown("### 💡 저평가 리뷰 개선 제안")
                st.caption(
                    f"⭐ 별점이 {final_score}점이라 전문가 Agent들이 "
                    "개선 방안을 도출 중입니다..."
                )

                with st.spinner("🔍 분야별 전문가 Agent가 개선안을 작성 중입니다..."):
                    fb_initial_state = {
                        "review": result.get("review", review),
                        "aspect": aspects,
                        "label": labels,
                        "next_node": [],
                        "result_list": [],
                        "final_result": "",
                        "messages": []
                    }
                    fb_final_state = feedback_app.invoke(fb_initial_state)

                routed = fb_final_state.get("next_node", [])
                if routed:
                    st.markdown(
                        f"**👥 투입된 전문가:** {', '.join(routed)}"
                    )

                expert_results = fb_final_state.get("result_list", [])
                if expert_results:
                    for i, opinion in enumerate(expert_results, 1):
                        with st.expander(f"📝 전문가 의견 #{i}", expanded=(i == 1)):
                            st.markdown(opinion)

                final_report = fb_final_state.get("final_result", "")
                if final_report:
                    with st.expander("📋 종합 개선 제안서", expanded=True):
                        st.markdown(final_report)
            elif final_score in [1, 2]:
                st.divider()
                st.info(
                    "⭐ 별점이 낮지만, 분석된 속성이 없어 "
                    "개선 제안을 생성하지 못했습니다."
                )
    else:
        st.info("리뷰를 입력하고 '분석 실행'을 눌러주세요.")

st.divider()

# --------------------------------------------------
# DB 로드 (이후 프레임1/프레임2에서 공유)
# --------------------------------------------------
df = load_all_reviews()

if df.empty:
    st.info("아직 분석된 리뷰가 없습니다. 위에서 리뷰를 분석해주세요.")
    st.stop()

# --------------------------------------------------
# 사이드바 필터 (프레임1/프레임2 공통)
# --------------------------------------------------
st.sidebar.header("🔍 필터")
score_min, score_max = st.sidebar.slider("별점 범위", 1, 5, (1, 5))
selected_aspects = st.sidebar.multiselect("포함할 속성", ASPECT_LIST, default=ASPECT_LIST)
keyword = st.sidebar.text_input("리뷰 키워드 검색", "")

mask = df["score"].between(score_min, score_max)
if selected_aspects:
    mask &= df["aspect_parsed"].apply(
        lambda lst: any(a in lst for a in selected_aspects) if lst else False
    )
if keyword:
    mask &= df["review"].fillna("").str.contains(keyword, case=False)

df_f = df[mask].reset_index(drop=True)

# --------------------------------------------------
# 상단 KPI
# --------------------------------------------------
summary_df = build_aspect_summary(df_f, ASPECT_LIST)
total_reviews = len(df_f)
avg_score = round(df_f["score"].mean(), 2) if total_reviews > 0 else 0

k1, k2, k3, k4 = st.columns(4)
k1.metric("총 리뷰 수", f"{total_reviews:,}")
k2.metric("평균 별점", f"{avg_score} / 5")
if not summary_df.empty and summary_df["언급횟수"].sum() > 0:
    best = summary_df.sort_values("긍정비율(%)", ascending=False).iloc[0]
    worst = summary_df.sort_values("긍정비율(%)", ascending=True).iloc[0]
    k3.metric("👍 최고 평가 속성", best["aspect"], f"{best['긍정비율(%)']}%")
    k4.metric("👎 최저 평가 속성", worst["aspect"], f"{worst['긍정비율(%)']}%")

st.divider()

# ==================================================
# 프레임1 : 리뷰 분석 결과 시각화
# ==================================================
st.header("📈 리뷰 분석 결과 시각화")

tab1, tab2, tab3 = st.tabs(["속성별 긍/부정", "별점 분포", "속성별 긍정비율"])

with tab1:
    if summary_df["언급횟수"].sum() == 0:
        st.info("표시할 데이터가 없습니다.")
    else:
        # 속성이 많으면 실제로 언급된 속성만 추려서 그래프가 안 겹치게 함
        plot_df = summary_df[summary_df["언급횟수"] > 0].reset_index(drop=True)
        if plot_df.empty:
            st.info("언급된 속성이 없습니다.")
        else:
            n = len(plot_df)
            fig_width = max(8, n * 0.5)
            fig, ax = plt.subplots(figsize=(fig_width, 5))
            x = np.arange(n)
            ax.bar(x - 0.2, plot_df["긍정"], width=0.4, label="긍정", color="#4C9AFF")
            ax.bar(x + 0.2, plot_df["부정"], width=0.4, label="부정", color="#FF6B6B")
            ax.set_xticks(x)
            ax.set_xticklabels(plot_df["aspect"], rotation=45, ha="right", fontsize=9)
            ax.set_ylabel("건수")
            ax.set_title("속성별 긍정/부정 건수 (언급된 속성만 표시)")
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig)
        st.dataframe(summary_df, use_container_width=True)

with tab2:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(data=df_f, x="score", order=[1, 2, 3, 4, 5], ax=ax, color="#4C9AFF")
    ax.set_xlabel("별점")
    ax.set_ylabel("건수")
    ax.set_title("별점 분포")
    st.pyplot(fig)

with tab3:
    if summary_df["언급횟수"].sum() == 0:
        st.info("표시할 데이터가 없습니다.")
    else:
        plot_df = summary_df[summary_df["언급횟수"] > 0].reset_index(drop=True)
        if plot_df.empty:
            st.info("언급된 속성이 없습니다.")
        else:
            fig_height = max(4, len(plot_df) * 0.4)
            fig, ax = plt.subplots(figsize=(8, fig_height))
            ax.barh(plot_df["aspect"], plot_df["긍정비율(%)"], color="#36B37E")
            ax.set_xlim(0, 100)
            ax.set_xlabel("긍정 비율 (%)")
            ax.set_title("속성별 긍정 비율 (언급된 속성만 표시)")
            for i, v in enumerate(plot_df["긍정비율(%)"]):
                ax.text(v + 1, i, f"{v}%", va="center")
            fig.tight_layout()
            st.pyplot(fig)

st.divider()

# ==================================================
# 프레임2 : 리뷰 분석 결과 건별 조회 (+ 피드백 버튼)
# ==================================================
st.header("🔎 리뷰 분석 결과 건별 조회")

# 전체 목록 (보기 좋게 컬럼 정리)
show_df = df_f[["id", "review", "aspect_parsed", "label_parsed", "score"]].copy()
show_df = show_df.rename(columns={"aspect_parsed": "aspect", "label_parsed": "label"})
st.dataframe(show_df, use_container_width=True, height=300)

# 단건 상세
st.subheader("📌 상세 조회")
selected_id = st.selectbox("리뷰 ID 선택", df_f["id"].tolist())
row = df_f[df_f["id"] == selected_id].iloc[0]

c_a, c_b = st.columns([2, 1])
with c_a:
    st.markdown(f"**📝 리뷰 원문**\n\n> {row['review']}")
    st.markdown(f"**⭐ 별점:** {row['score']}")

with c_b:
    st.markdown("**🏷️ 속성별 평가**")
    aspects = row["aspect_parsed"] or []
    labels = row["label_parsed"] or []
    if aspects and labels and len(aspects) == len(labels):
        detail_df = pd.DataFrame({
            "속성": aspects,
            "감성": ["😊 긍정" if l == 1 else "😞 부정" for l in labels],
        })
        st.table(detail_df)
    else:
        st.info("속성 분석 결과가 없습니다.")

# 저평가 리뷰에 대한 사후 피드백 생성 버튼
if row["score"] in [1, 2] and aspects and labels:
    st.divider()
    st.markdown("### 💡 이 리뷰에 대한 개선 제안 생성")
    st.caption(f"⭐ {row['score']}점 리뷰입니다. 전문가 Agent들의 개선안을 받아볼 수 있습니다.")

    if st.button("🚀 개선 제안 생성하기", key=f"gen_feedback_{selected_id}"):
        with st.spinner("🔍 분야별 전문가 Agent가 개선안을 작성 중입니다..."):
            fb_initial_state = {
                "review": row["review"],
                "aspect": aspects,
                "label": labels,
                "next_node": [],
                "result_list": [],
                "final_result": "",
                "messages": []
            }
            fb_final_state = feedback_app.invoke(fb_initial_state)

        routed = fb_final_state.get("next_node", [])
        if routed:
            st.markdown(f"**👥 투입된 전문가:** {', '.join(routed)}")

        expert_results = fb_final_state.get("result_list", [])
        for i, opinion in enumerate(expert_results, 1):
            with st.expander(f"📝 전문가 의견 #{i}", expanded=(i == 1)):
                st.markdown(opinion)

        final_report = fb_final_state.get("final_result", "")
        if final_report:
            with st.expander("📋 종합 개선 제안서", expanded=True):
                st.markdown(final_report)