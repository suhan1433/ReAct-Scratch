import requests
import wikipediaapi
import json
from enum import Enum, auto
from typing import Callable, Dict, List, Union, Optional
import logging
import os
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

# LLM 예시: openai (원하시는 LLM 패키지로 교체 가능)
# 간단한 logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Agent")

# # SerpAPI 키 직접 입력
# SERP_API_KEY = "여기에_본인_API_KEY_입력"

# 한글 위키
wiki = wikipediaapi.Wikipedia(user_agent='ReAct Agent', language='ko')

# 도구 이름 Enum
class Name(Enum):
    WIKIPEDIA = auto()
    GOOGLE = auto()
    NONE = auto()
    def __str__(self):
        return self.name.lower()

# 도구 래퍼
class Tool:
    def __init__(self, name: Name, func: Callable[[str], str]):
        self.name = name
        self.func = func
    def use(self, query: str) -> str:
        try:
            return self.func(query)
        except Exception as e:
            logger.error(f"Error executing tool {self.name}: {e}")
            return str(e)

# 메시지
class Message:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

# Agent
class Agent:
    def __init__(self, max_iterations: int = 5):
        self.tools: Dict[Name, Tool] = {}
        self.messages: List[Message] = []
        self.query = ""
        self.max_iterations = max_iterations
        self.current_iteration = 0

    def register(self, name: Name, func: Callable[[str], str]):
        self.tools[name] = Tool(name, func)

    def trace(self, role: str, content: str):
        self.messages.append(Message(role, content))
        logger.info(f"{role}: {content}")

    def get_history(self) -> str:
        return "\n".join([f"{m.role}: {m.content}" for m in self.messages])

    def think(self):
        self.current_iteration += 1
        if self.current_iteration > self.max_iterations:
            self.trace("assistant", "최대 반복 횟수 도달. 중단합니다.")
            return
        prompt = self.make_prompt()
        response = self.ask_llm(prompt)
        self.trace("assistant", f"Thought: {response}")
        self.decide(response)

    def make_prompt(self) -> str:
        # history, tools, query 포함 프롬프트
        return f"""당신은 ReAct(Reasoning and Acting) 에이전트입니다. 아래의 질의에 답하는 것이 임무입니다.

질문: {self.query}

이전 추론 및 관찰 기록: {self.get_history()}

사용 가능한 도구: {', '.join([str(t.name) for t in self.tools.values()])}

[지침]
1. 질문, 이전 추론, 관찰 기록을 분석하세요.
2. **질문이 명확하지 않거나, 추가 정보가 필요하다고 판단되면** 바로 구체적인 질문을 요청하세요.:
   {{"thought": "다음에 무엇을 해야할지와 질문이 명확하지 않다고 판단한 이유", "answer": "이유와 함께 질문을 더 구체적으로 해달라는 요청"}}
3. **질문이 명확하다면**, 아래의 순서로 행동하세요:
   - 대화 기록과 도구만으로 최종 답변이 가능하면 바로 답변하세요.
   - 답변이 불충분하거나, 추가 정보가 필요하다고 판단되면, 도구를 적극적으로 사용하세요.
   - 도구 사용은 한 번에 하나씩만이 아니라, 필요하다면 여러 번 반복적으로 사용해도 됩니다.
   - 도구 사용의 목적, 선택 이유, 입력값을 명확히 하세요.
   - 도구 사용 후에도 답이 불명확하면, 추가 도구 사용을 반복하세요.
4. 도구 사용 시에는 아래와 같은 JSON 포맷을 반드시 지키세요.
     {{
         "thought": "다음에 무엇을 해야 할지에 대한 상세한 추론",
         "action": {{
             "name": "도구 이름 (wikipedia, google, 또는 none)",
             "reason": "이 도구를 선택한 이유에 대한 설명",
             "input": "원래 질문과 다를 경우 사용할 구체적인 입력값"
         }}
     }}
5. 최종 답변이 가능할 때는 아래와 같은 JSON 포맷을 사용하세요.
     {{
         "thought": "최종적으로 도출한 추론 과정",
         "answer": "질문에 대한 종합적인 답변"
     }}
6. 도구 사용 결과가 없거나 실패하면, 이를 인지하고 다른 도구나 접근법을 고려하세요.
7. 여러 번 도구를 사용해도 답을 찾지 못하면, 솔직하게 답변이 어렵다고 인정하세요.

[예시]

1) 명확하지 않은 질문
질문: 의사는 어디?
응답:
{{
  "thought": "질문이 모호하므로, 사용자가 어떤 의사를 찾는지 추가 정보를 요청해야겠다."",
  "answer": "어떤 의사를 찾으시나요? 증상을 알려주세요."
}}

2) 도구 사용
질문: 파이썬이 뭐야?
응답:
{{
  "thought": "질문에 대한 정의를 제공하려면 신뢰할 수 있는 백과사전 정보가 필요하다. wikipedia 도구를 사용해 파이썬에 대해 검색해야겠다.",
  "action": {{
    "name": "wikipedia",
    "reason": "파이썬의 정의를 얻기 위해",
    "input": "파이썬"
  }}
}}

3) 도구 사용 반복
질문: 오늘 날씨 어때?
응답:
{{
  "thought": "오늘의 날씨 정보를 얻으려면 실시간 검색이 필요하다. google 도구를 사용해 최신 날씨 정보를 확인해야겠다.",
  "action": {{
    "name": "google",
    "reason": "오늘 날씨 정보를 얻기 위해",
    "input": "오늘 날씨"
  }}
}}

4) 최종 답변
질문: 자바란 뭐야? 자바(Java)는 객체지향 프로그래밍 언어로, 1995년 썬 마이크로시스템즈(Sun Microsystems)에서 처음 발표되었고, 현재는 오라클(Oracle)이 관리하고 있습니다.
응답:
{{
  "thought": "필요한 정보를 모두 수집했으니, 이제 종합하여 최종 답변을 제공해야겠다.",
  "answer": "자바는 객체지향 프로그래밍 언어로 ..."
}}

5) 도구 사용 실패/불충분
질문: 2024년 노벨상 수상자는 누구야?
응답:
{{
  "thought": "wikipedia에서 원하는 정보를 찾지 못했다. 추가 정보를 얻기 위해 google 도구를 사용해 다시 검색해야겠다.",
  "action": {{
    "name": "google",
    "reason": "wikipedia에서 정보를 찾지 못했기 때문",
    "input": "2024년 노벨상 수상자"
  }}    
}}

6) 여러 번 도구를 사용해도 답을 찾지 못한 경우
질문: 외계인의 실제 사진을 보여줘
응답:
{{          
  "thought": "여러 도구를 사용해도 신뢰할 수 있는 정보를 찾지 못했다. 더 이상 시도해도 의미가 없으므로 답변이 어렵다고 안내해야겠다.",
  "answer": "죄송합니다. 해당 질문에 대해 신뢰할 수 있는 정보를 찾을 수 없습니다."
}}

---
**각 예시에는 반드시 '질문'과 '응답'을 함께 명시하세요.**
"""

    def ask_llm(self, prompt: str) -> str:
        # OpenAI 예시 (gpt-3.5-turbo)
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "system", "content": "Assistant can use tools"},
                      {"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    



    def decide(self, response: str):
        try:
            cleaned = response.strip().strip('`').strip()
            if cleaned.startswith('json'):
                cleaned = cleaned[4:].strip()
            parsed = json.loads(cleaned)
            if "action" in parsed:
                action = parsed["action"]
                tool_name = Name[action["name"].upper()]
                if tool_name == Name.NONE:
                    self.trace("assistant", "도구 사용 없이 답변을 시도합니다.")
                    self.think()
                else:
                    self.trace("assistant", f"Action: {tool_name} 도구 사용")
                    self.act(tool_name, action.get("input", self.query))
            elif "answer" in parsed:
                self.trace("assistant", f"최종 답변: {parsed['answer']}")
            else:
                raise ValueError("응답 포맷 오류")
        except Exception as e:
            logger.error(f"응답 처리 오류: {e}")
            self.trace("assistant", "처리 오류. 다시 시도합니다.")
            self.think()

    def act(self, tool_name: Name, query: str):
        tool = self.tools.get(tool_name)
        if tool:
            result = tool.use(query)
            observation = f"{tool_name} 결과: {result}"
            self.trace("system", observation)
            self.think()
        else:
            self.trace("system", f"도구 {tool_name} 없음")
            self.think()

    def execute(self, query: str) -> str:
        self.query = query
        self.trace("user", query)
        self.think()
        return self.messages[-1].content

# 도구 함수
def google_search(query: str) -> str:
    url = "https://serpapi.com/search"
    params = {
        "engine": "google",
        "q": query,
        "api_key": SERP_API_KEY,
        "hl": "ko"
    }
    try:
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("organic_results", [])
        if not results:
            return "검색 결과 없음"
        top = results[0]
        return f"{top.get('title')}: {top.get('snippet')}"
    except Exception as e:
        return f"Google 검색 오류: {e}"

def wiki_search(query: str) -> str:
    page = wiki.page(query)
    if page.exists():
        return page.summary[:500]  # 500자 제한
    else:
        return "위키백과에 결과 없음"

# main
if __name__ == "__main__":
    # OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    os.environ["OPENAI_API_KEY"] = ""
   
   
   
    client = OpenAI()
    agent = Agent()
    agent.register(Name.WIKIPEDIA, wiki_search)
    agent.register(Name.GOOGLE, google_search)
    question = input("질문을 입력하세요: ")
    answer = agent.execute(question)
    print("\n[최종 답변]")
    print(answer)
