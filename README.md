# ReAct
ReAct Agent Description &amp; Code

- [ReAct란?](#react란)
- [CoT vs Act-Only vs ReAct](#cot-vs-act-only-vs-react)
- [CoT vs ReAct 비교 실험](#cot-vs-react-비교-실험)
- [실험](#실험)

## **개요**

LLM을 활용할 때 다음 두 가지 핵심 기능을 구현하고자 함:

- **모호한 사용자 질문에 대한 적절한 피드백 제공**
- **질문에 맞는 Tool을 효과적으로 호출하는 기능**

이를 실현하기 위한 방법론을 검토함.

– 실험부터 읽어야 이해하기 좋습니다

## **ReAct 선정 이유**

### **선정 이유**

1. 가장 직관적이고 대표적인 에이전트 디자인 패턴
2. Tool 호출과 사용자 피드백 처리 성능이 괜찮은 편

### **한계점**

- 현재 피드백 제공과 Tool 호출 모두 LLM의 성능에 전적으로 의존
- 대화기록들을 저장해야 하기에 메모리 문제가 있음
- 안정적이고 향상된 성능을 기대하려면 Trajectories 데이터셋으로 파인튜닝이 필요해 보임

```
# Trajectories 데이터
입력: "오늘 서울 날씨 알려줘"
대화기록:
Thought: 현재 서울의 날씨 정보를 찾아야 한다.
Action: search
Action Input: {"query": "서울 오늘 날씨"}
Observation: [검색 결과]
Thought: 검색 결과를 바탕으로 답변을 정리하겠다.
Final Answer: 서울의 오늘 날씨는...
```

Prompt로 In-Context learning을 진행할 시 성능 좋지 않음,

3000개로 Finetune 진행 시 성능 오름.
<img width="847" height="327" alt="Prompt_Fintune" src="https://github.com/user-attachments/assets/2bb099b9-b250-44dd-a3f7-a84da65aa35e" />

## **ReAct란?**

ReAct는 LLM이 Reasoning과 Acting을 결합하여 문제를 해결하도록 설계된 에이전트

이 방식은 크게 Thought → Action → Observation의 3단계 과정을 반복하면서 복잡한 문제를 단계적으로 해결

- **Thought (생각)**
    
    LLM이 현재까지의 정보를 바탕으로 다음에 해야 할 추론 또는 행동을 언어로 표현
    
    - 예: "날씨 정보를 얻기 위해 검색이 필요하다."
- **Action (행동)**
    
    LLM이 필요하다고 판단한 도구(tool)나 함수(function)를 명시적으로 호출
    
    - 예: `"action": { "name": "google", "input": "오늘 서울 날씨" }`
- **Observation (관찰)**
    
    도구 실행 결과로 얻은 정보를 LLM에 다시 제공하여 다음 추론의 기반으로 사용
    
    - 예: `"observation": "서울의 오늘 날씨는 맑고 29도입니다."`

e.g. 오늘 날씨 어때?

```
질문: 오늘 날씨 어때?
1. thought: "날씨 정보를 얻기 위해 검색 도구를 사용하자."
2. action: { "name": "google", "input": "오늘 날씨" }
3. observation: "서울의 오늘 날씨는 맑고 29도입니다."
4. thought: "관찰 결과를 바탕으로 답변을 생성하자."
5. response: "오늘 서울은 맑고 기온은 29도입니다."
```

### **Function Calling과 Tool의 차이**

Function Calling : LLM이 특정 함수 또는 도구를 명시적으로 호출하도록 지시하는 방식

Tool : LLM이 선택한 도구로, 실제로 기능을 수행하는 구현된 함수 또는 API

e.g. Function Calling

```
"action": {
  "name": "wikipedia",
  "reason": "치킨에 대한 정의와 정보를 얻기 위해",
  "input": "치킨"
}
```

e.g. Tool(Wiki Search)

```
def wiki_search(query: str) -> str:
    page = wiki.page(query)
    if page.exists():
        return page.summary[:500]  # 500자 제한
    else:
        return "위키백과에 결과 없음"
```

| **Reasoning** (Thought) | 어떤 정보를 얻기 위해 어떤 행동이 필요한지, 다음에 무엇을 어떻게 해야할지 추론 |
| --- | --- |
| **Function Calling** (Action) | 실제 필요한 행동을 Json 형태 등으로 명시 |
| **Observation** | 실행 결과를 받아 다음 추론에 활용 |
| **LLM Loop** | 위의 과정을 반복하여 문제 해결 |

## **CoT vs Act-Only vs ReAct**
<img width="1071" height="455" alt="Act_CoT_ReAct" src="https://github.com/user-attachments/assets/b77f08c5-0ca1-43b2-987f-7a56151445df" />


### **CoT의 한계점(1b)**

내부 모델의 정보로만 추론을 진행하기에 정보가 업데이트 되지 않아 할루시네이션 문제 있음

(1b) 결과를 보면,

Apple Remote는 Front Row라는 Mac용 미디어 소프트웨어를 제어를 하지만

CoT는 Apple Tv를 위해 생겨났다고 나오며, 이로인해 이후 추론도 잘못된 방향으로 됨.


### **Act-Only의 한계점(1c)**

Act만 수행하는 경우, 이전에 수행한 행동이나 관찰만으로는 현재 어떤 행동을 해야 할지 판단하기 어려워짐

(1c)를 보면,

`Apple Remote → Front Row → Front Row (software)`로 진행했지만,

이 정보들을 논리적으로 연결해서, "아, Apple Remote는 Front Row라는 소프트웨어를 조작하는구나"라는 결론 내리지 못 함

π(aₜ | cₜ) : Act를 결정하기 위한 방법

- a**ₜ**: 에이전트가 t 시점에 취할 수 있는 행동(action)
- cₜ: 시점 t까지의 문맥(context).
    
    예: `cₜ = (o₁, a₁, ..., oₜ)`
    
- π(aₜ | cₜ):
    
    주어진 문맥 cₜ에서 행동 aₜ를 선택할 확률
    

Act만 할 경우에는 현재까지 진행한 문맥(cₜ)과 현재 시점에서 진행해야 하는 행동(a**ₜ**) 매핑이 힘듦


### **ReAct(1d)**

1. Thought를 통해 다음에 할 행동이 무엇인지 판단
2. Act를 통해 검색 하도록 명시한 후
3. Obs에 검색된 결과가 나옴

위에 대화 기록들을 저장하고 순회하며 최종적으로 적절합 답변을 도출함


## **CoT vs ReAct 비교 실험**

<img width="1018" height="204" alt="CoT_ReAct_" src="https://github.com/user-attachments/assets/f5eadd94-0d0b-466a-a368-c0b13b7e0a6c" />

Model: PaLM 540B(Finetuning x)

Dataset : Hotpot QA

```
{
  "id": "5a8b57f25542995d1e6f1371",
  "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
  "answer": "yes",
  "type": "comparison",
  "level": "hard",
  "supporting_facts": [
    ["Scott Derrickson", 0],
    ["Ed Wood", 0]
  ],
  "context": [
    {
      "title": "Scott Derrickson",
      "sentences": [
        "Scott Derrickson is an American director...",
        "He directed films like..."
      ]
    },
    {
      "title": "Ed Wood",
      "sentences": [
        "Ed Wood was an American filmmaker...",
        "He is known for..."
      ]
    }
  ]
}
```

e.g. Hotpot QA Test 200개 [ReAct 성공:50, 실패:50),  CoT(성공:50, 실패:50)]

```
Question: "Were Scott Derrickson and Ed Wood of the same nationality?"
Agent reasoning process:
Thought: 두 사람의 국적을 각각 찾아야 한다
Action: WikipediaSearch["Scott Derrickson"]
Observation: Scott Derrickson is an American director...
Thought: Scott은 미국인이다. 이제 Ed Wood의 국적을 찾아보자
Action: WikipediaSearch["Ed Wood"]
Observation: Ed Wood was an American filmmaker...
Thought: Ed Wood도 미국인이다. 둘 다 미국인이므로 같은 국적이다
Action: Answer["yes"]
```

- Action 결과가 Sucess와 False인 경우를 분석

결과:

HotpotQA(CoT보다 약간 낮은 성능(27.4 vs. 29.4))

Fever(CoT보다 더 우수한 성능(60.9 vs. 56.3))

### **분석**

1. Success(Type: FP) :
    
    CoT의 FP에서도 할루시네이션이 많음(CoT:14%, ReAct:6%)
    
2. Failure(Type:할루시네이션) :
    
    CoT는 할루시네이션이 56%, ReAct는 외부 지식에 접근할 수 있기에 왜곡문제 줄어듦
    

### **ReAct의 한계점**

1. **Reasoning error**
- ReAct는 Thought → Action → Observation 구조를 통해
    
    → 사실 기반성과 신뢰도는 높음
    
- 하지만 구조적 한계로 추론 유연성을 떨어뜨림
    
    → **CoT보다 추론 오류 발생률이 높아질 수 있음**
    

대표적인 오류 패턴

- 모델이 이전 Thought와 Action을 반복 생성
- 다음 행동을 제대로 추론하지 못하고 루프에 빠짐

e.g.

<img width="705" height="215" alt="Reasoning_Error" src="https://github.com/user-attachments/assets/17494a69-9fb2-415e-9547-5d30edc7d06d" />

- Hairspray에 출연한 여러 배우들과 함께 출연한 "여배우"의 미들 네임을 묻는 질문
    
    질문의 핵심 대상(=해당 여배우)을 먼저 식별하지 않고, 등장한 모든 배우를 하나씩 검색하며 접근함
    
    → 비효율적 추론 흐름, 루프 가능성 존재
  
    
1. **Search result error**
- ReAct는 검색(Search)을 통해
    
    정보성 있는 지식을 가져와 사실성을 높임
    
- 그러나 전체 오류 중 **23%는 비정보성 검색(non-informative search)** 때문에
    
    → **모델의 추론 흐름이 흐트러지고, 회복도 어려움**
    
- ReAct + CoT-SC 결합 전략으로 어느정도 해결

<img width="709" height="157" alt="Search_Error" src="https://github.com/user-attachments/assets/6f6383c0-6617-4652-8c2f-5b2b8ac5eeb0" />

    

A) ReAct → CoT-SC:

ReAct가 정해진 단계 수 안에 답을 반환하지 못하면, CoT-SC로 되돌아가기(back off).

실험 결과, HotpotQA에서는 7단계, FEVER에서는 5단계 이상으로 늘려도 성능이 더 이상 개선되지 않아 이 수치를 기준으로 설정

B) CoT-SC → ReAct:

CoT-SC에서 n개의 샘플 중 과반수(n/2)보다 적은 수의 샘플만이 동일한 최종 답을 낸 경우,

이는 내부 지식(internal knowledge)만으로는 확신을 갖기 어렵다는 뜻이므로, ReAct 방식으로 되돌아감



## **실험**

Gpt 4.1-mini로 실험

<img width="820" height="976" alt="Architecture" src="https://github.com/user-attachments/assets/4177210d-39d0-4d15-804a-e26cca831125" />

### **피드백 방법**

Prompt로 In-Context learning을 통해 피드백 하도록 요청

```
2. 질문이 명확하지 않거나, 추가 정보가 필요하다고 판단되면 바로 구체적인 질문을 요청하세요.:
   {
      "thought": "다음에 무엇을 해야할지와 질문이 명확하지 않다고 판단한 이유",
      "answer": "이유와 함께 질문을 더 구체적으로 해달라는 요청"
    }
[예시]
1) 명확하지 않은 질문
질문: 의사는 어디?
응답:
  {
    "thought": "질문이 모호하므로, 사용자가 어떤 의사를 찾는지 추가 정보를 요청해야겠다."",
    "answer": "어떤 의사를 찾으시나요? 증상을 알려주세요."
  }
```

### **Tool 호출 방법**

Prompt로 In-Context learning을 통해 Tool 요청

```
3. 질문이 명확하다면, 아래의 순서로 행동하세요:
   - 대화 기록과 도구만으로 최종 답변이 가능하면 바로 답변하세요.
   - 답변이 불충분하거나, 추가 정보가 필요하다고 판단되면, 도구를 적극적으로 사용하세요.
   - 도구 사용은 한 번에 하나씩만이 아니라, 필요하다면 여러 번 반복적으로 사용해도 됩니다.
   - 도구 사용의 목적, 선택 이유, 입력값을 명확히 하세요.
   - 도구 사용 후에도 답이 불명확하면, 추가 도구 사용을 반복하세요.
4. 도구 사용 시에는 아래와 같은 JSON 포맷을 반드시 지키세요.
     {
         "thought": "다음에 무엇을 해야 할지에 대한 상세한 추론",
         "action": {
             "name": "도구 이름 (wikipedia, google, 또는 none)",
             "reason": "이 도구를 선택한 이유에 대한 설명",
             "input": "원래 질문과 다를 경우 사용할 구체적인 입력값"
         }
     }
[예시]
2) 도구 사용
질문: 파이썬이 뭐야?
응답:
{
  "thought": "질문에 대한 정의를 제공하려면 신뢰할 수 있는 백과사전 정보가 필요하다. wikipedia 도구를 사용해 파이썬에 대해 검색해야겠다.",
  "action": {
    "name": "wikipedia",
    "reason": "파이썬의 정의를 얻기 위해",
    "input": "파이썬"
  }
}
```

### **전체적 흐름**

### **모호한 질문(예시1)**

Q: 치킨

A: 치킨에 관해 어떤 정보를 원하시는지 구체적으로 알려주세요. 예를 들어 치킨의 종류, 조리법, 역사, 또는 인기 있는 치킨 브랜드 등에 대해 알고 싶으신가요?

1. 질문입력

```
질문을 입력하세요: 치킨
```

1. Thought
    
    이전 대화 기록들을 통해 추론
    

```
# Prompt
프롬프트내용들 당신은 ReAct(Reasoning and Acting) 에이전트입니다. 아래의 질의에 답하는 것이 임무입니다.
질문: 치킨
이전 추론 및 관찰 기록: user: 치킨
사용 가능한 도구: wikipedia, google
[지침]
1. 질문, 이전 추론, 관찰 기록을 분석하세요.
2. **질문이 명확하지 않거나, 추가 정보가 필요하다고 판단되면** 바로 구체적인 질문을 요청하세요.:
{"thought": "다음에 무엇을 해야할지와 질문이 명확하지 않다고 판단한 이유", "answer": "이유와 함께 질문을 더 구체적으로 해달라는 요청"}
3. **질문이 명확하다면**, 아래의 순서로 행동하세요:
    - 대화 기록과 도구만으로 최종 답변이 가능하면 바로 답변하세요.
    - 답변이 불충분하거나, 추가 정보가 필요하다고 판단되면, 도구를 적극적으로 사용하세요.
    - 도구 사용은 한 번에 하나씩만이 아니라, 필요하다면 여러 번 반복적으로 사용해도 됩니다.
    - 도구 사용의 목적, 선택 이유, 입력값을 명확히 하세요.
    - 도구 사용 후에도 답이 불명확하면, 추가 도구 사용을 반복하세요.
4. 도구 사용 시에는 아래와 같은 JSON 포맷을 반드시 지키세요.
{
  "thought": "다음에 무엇을 해야 할지에 대한 상세한 추론",
  "action": {
  "name": "도구 이름 (wikipedia, google, 또는 none)",
  "reason": "이 도구를 선택한 이유에 대한 설명",
  "input": "원래 질문과 다를 경우 사용할 구체적인 입력값"
  }
}
5. 최종 답변이 가능할 때는 아래와 같은 JSON 포맷을 사용하세요.
{
  "thought": "최종적으로 도출한 추론 과정",
  "answer": "질문에 대한 종합적인 답변"
}
6. 도구 사용 결과가 없거나 실패하면, 이를 인지하고 다른 도구나 접근법을 고려하세요.
7. 여러 번 도구를 사용해도 답을 찾지 못하면, 솔직하게 답변이 어렵다고 인정하세요.
```

1. Action

모호한 질문에 대해 구체적으로 답변 해달라고 요구

```
{
"thought": "질문 '치킨'은 매우 간단하고 포괄적이어서 구체적으로 어떤 정보를 원하는지 알기 어렵다. 치킨의 종류, 역사, 조리법, 또는 치킨 관련 특정 정보인지 명확하지 않기 때문에 추가 질문을 해야겠다.",
"answer": "치킨에 관해 어떤 정보를 원하시는지 구체적으로 알려주세요. 예를 들어 치킨의 종류, 조리법, 역사, 또는 인기 있는 치킨 브랜드 등에 대해 알고 싶으신가요?"
}
```

1. 결과

```
최종 답변:
치킨에 관해 어떤 정보를 원하시는지 구체적으로 알려주세요.
예를 들어 치킨의 종류, 조리법, 역사, 또는 인기 있는 치킨 브랜드 등에 대해 알고 싶으신가요?
```

### **질문 보완(예시1, Tool 활용)**

Q: 치킨이 뭐야

A: 치킨은 닭 또는 닭고기를 가리키며, 특히 닭을 튀기거나 조리한 음식을 뜻합니다. 한국에서는 주로 닭튀김 요리를 치킨이라고 부르며, 미국식 프라이드 치킨이 대표적입니다.

1. 질문

```
질문을 입력하세요: 치킨이 뭐야
```

1. Thought

```
프롬프트내용들 당신은 ReAct(Reasoning and Acting) 에이전트입니다. 아래의 질의에 답하는 것이 임무입니다.
질문: 치킨이 뭐야
이전 추론 및 관찰 기록: user: 치킨이 뭐야
사용 가능한 도구: wikipedia, google
[지침]
1. 질문, 이전 추론, 관찰 기록을 분석하세요.
2. **질문이 명확하지 않거나, 추가 정보가 필요하다고 판단되면** 바로 구체적인 질문을 요청하세요.:
{"thought": "다음에 무엇을 해야할지와 질문이 명확하지 않다고 판단한 이유", "answer": "이유와 함께 질문을 더 구체적으로 해달라는 요청"}
3. **질문이 명확하다면**, 아래의 순서로 행동하세요:
    - 대화 기록과 도구만으로 최종 답변이 가능하면 바로 답변하세요.
    - 답변이 불충분하거나, 추가 정보가 필요하다고 판단되면, 도구를 적극적으로 사용하세요.
    - 도구 사용은 한 번에 하나씩만이 아니라, 필요하다면 여러 번 반복적으로 사용해도 됩니다.
    - 도구 사용의 목적, 선택 이유, 입력값을 명확히 하세요.
    - 도구 사용 후에도 답이 불명확하면, 추가 도구 사용을 반복하세요.
4. 도구 사용 시에는 아래와 같은 JSON 포맷을 반드시 지키세요.
{
  "thought": "다음에 무엇을 해야 할지에 대한 상세한 추론",
  "action": {
  "name": "도구 이름 (wikipedia, google, 또는 none)",
  "reason": "이 도구를 선택한 이유에 대한 설명",
  "input": "원래 질문과 다를 경우 사용할 구체적인 입력값"
  }
}
5. 최종 답변이 가능할 때는 아래와 같은 JSON 포맷을 사용하세요.
{
  "thought": "최종적으로 도출한 추론 과정",
  "answer": "질문에 대한 종합적인 답변"
}
6. 도구 사용 결과가 없거나 실패하면, 이를 인지하고 다른 도구나 접근법을 고려하세요.
7. 여러 번 도구를 사용해도 답을 찾지 못하면, 솔직하게 답변이 어렵다고 인정하세요.
```

1. Action

```
{
  "thought": "질문 '치킨이 뭐야'는 일반적인 용어에 대한 정의를 묻는 것으로 명확하다.
              치킨은 음식으로서의 닭고기 조리법을 의미하는 경우가 많으므로, 신뢰할 수 있는 백과사전 정보가 필요하다.
              wikipedia 도구를 사용하여 치킨에 대한 포괄적 정의와 배경 정보를 얻는 것이 적절하다.",
  "action": {
  "name": "wikipedia",
  "reason": "치킨에 대한 정의와 정보를 얻기 위해",
  "input": "치킨"
  }
}
```

1. Obs(Tool: Wiki search)

```
wikipedia 결과: 치킨(영어: chicken)은 닭, 닭고기 또는 닭튀김을 뜻한다.
치킨(한국)
프라이드 치킨(미국)
```

1. Thought

```
당신은 ReAct(Reasoning and Acting) 에이전트입니다. 아래의 질의에 답하는 것이 임무입니다.
질문: 치킨이 뭐야
이전 추론 및 관찰 기록:
user: 치킨이 뭐야
assistant: Thought: {
  "thought": "질문 '치킨이 뭐야'는 일반적인 용어에 대한 정의를 묻는 것으로 명확하다. 치킨은 음식으로서의 닭고기 조리법을 의미하는 경우가 많으므로, 신뢰할 수 있는 백과사전 정보가 필요하다. wikipedia 도구를 사용하여 치킨에 대한 포괄적 정의와 배경 정보를 얻는 것이 적절하다.",
  "action": {
  "name": "wikipedia",
  "reason": "치킨에 대한 정의와 정보를 얻기 위해",
  "input": "치킨"
  }
}
assistant: Action: wikipedia 도구 사용
system: wikipedia 결과: 치킨(영어: chicken)은 닭, 닭고기 또는 닭튀김을 뜻한다.
치킨(한국)
프라이드 치킨(미국)
사용 가능한 도구: wikipedia, google
[지침]
1. 질문, 이전 추론, 관찰 기록을 분석하세요.
2. **질문이 명확하지 않거나, 추가 정보가 필요하다고 판단되면** 바로 구체적인 질문을 요청하세요.:
{"thought": "다음에 무엇을 해야할지와 질문이 명확하지 않다고 판단한 이유", "answer": "이유와 함께 질문을 더 구체적으로 해달라는 요청"}
...
```

1. Action

```
{
"thought": "wikipedia 검색 결과에 따르면 '치킨'은 닭, 닭고기 또는 닭튀김을 뜻한다.
            일반적으로 음식으로서의 치킨은 닭고기를 재료로 한 요리를 의미하며, 프라이드 치킨처럼 튀긴 닭고기를 가리키는 경우가 많다.
            따라서 이를 종합하여 치킨에 대한 기본 정의와 관련된 정보를 답변으로 제시한다.",
"answer": "치킨은 닭 또는 닭고기를 가리키며, 특히 닭을 튀기거나 조리한 음식을 뜻합니다.
           한국에서는 주로 닭튀김 요리를 치킨이라고 부르며, 미국식 프라이드 치킨이 대표적입니다."
}
```

1. 결과

```
최종 답변:
치킨은 닭 또는 닭고기를 가리키며, 특히 닭을 튀기거나 조리한 음식을 뜻합니다.
한국에서는 주로 닭튀김 요리를 치킨이라고 부르며, 미국식 프라이드 치킨이 대표적입니다.
```

### **모호한 질문(예시2)**

Q: select 어떻게 써?

A: select가 무엇을 의미하는지, 예를 들어 SQL의 select 문인지, 프로그래밍 언어에서의 select인지, 아니면 다른 맥락인지 구체적으로 알려주시면 더 정확하게 설명해 드릴 수 있습니다.

1. 질문

```
질문을 입력하세요: select 어떻게 써?
```

1. Thought

```
질문: select 어떻게 써?
이전 추론 및 관찰 기록: user: select 어떻게 써?
사용 가능한 도구: wikipedia, google
[지침]
1. 질문, 이전 추론, 관찰 기록을 분석하세요.
2. **질문이 명확하지 않거나, 추가 정보가 필요하다고 판단되면** 바로 구체적인 질문을 요청하세요.:
{"thought": "다음에 무엇을 해야할지와 질문이 명확하지 않다고 판단한 이유", "answer": "이유와 함께 질문을 더 구체적으로 해달라는 요청"}
...
```

1. Action

```
{
"thought": "질문이 'select 어떻게 써?'라고 매우 간단하게 표현되어 있지만, 'select'가 무엇을 의미하는지 명확하지 않다.
            예를 들어 SQL의 select 문인지, 프로그래밍 언어의 특정 함수인지, 아니면 다른 분야의 용어인지 추가 정보가 필요하다.
            따라서 사용자가 원하는 'select'의 맥락이나 분야에 대해 구체적으로 질문할 필요가 있다.",
"answer": "select가 무엇을 의미하는지, 예를 들어 SQL의 select 문인지, 프로그래밍 언어에서의 select인지,
           아니면 다른 맥락인지 구체적으로 알려주시면 더 정확하게 설명해 드릴 수 있습니다."
}
```

1. 결과

```
최종 답변:
select가 무엇을 의미하는지, 예를 들어 SQL의 select 문인지, 프로그래밍 언어에서의 select인지,
아니면 다른 맥락인지 구체적으로 알려주시면 더 정확하게 설명해 드릴 수 있습니다.
```

### **질문 보완(예시2 내부추론)**

Q: SQL의 select 문 어떻게 써

A: SQL의 select 문은 데이터베이스에서 데이터를 조회할 때 사용하는 명령어입니다. 기본적인 문법은 다음과 같습니다: …

1. 질문

```
질문을 입력하세요: SQL의 select 문 어떻게 써
```

1. Thought

```
프롬프트내용들 당신은 ReAct(Reasoning and Acting) 에이전트입니다. 아래의 질의에 답하는 것이 임무입니다.
질문: SQL의 select 문 어떻게 써
이전 추론 및 관찰 기록: user: SQL의 select 문 어떻게 써
사용 가능한 도구: wikipedia, google
[지침]
1. 질문, 이전 추론, 관찰 기록을 분석하세요.
2. **질문이 명확하지 않거나, 추가 정보가 필요하다고 판단되면** 바로 구체적인 질문을 요청하세요.:
{"thought": "다음에 무엇을 해야할지와 질문이 명확하지 않다고 판단한 이유", "answer": "이유와 함께 질문을 더 구체적으로 해달라는 요청"}
3. **질문이 명확하다면**, 아래의 순서로 행동하세요:
...
```

1. Action

```
{
"thought": "사용자가 SQL의 select 문 작성 방법에 대해 일반적인 정보를 요청하고 있다고 판단하여,
            기본적인 select 문 구조와 사용법을 설명하는 것이 적절하다.",
"answer": "SQL의 select 문은 데이터베이스에서 데이터를 조회할 때 사용하는 명령어입니다.
            기본적인 문법은 다음과 같습니다:\n\n\\nSELECT 컬럼명1, 컬럼명2, ...\\nFROM 테이블명\\nWHERE 조건식;\\n\n\n예를 들어, 'employees' 테이블에서 이름(name)과 나이(age)를 조회하려면:\n\n\\nSELECT name, age\\nFROM employees;\\n\n\n특정 조건을 걸고 싶으면 WHERE 절을 사용합니다:\n\n\\nSELECT name, age\\nFROM employees\\nWHERE age > 30;\\n\n\n이 외에도 정렬(ORDER BY), 그룹화(GROUP BY) 등 다양한 옵션이 있습니다. 더 구체적인 예시나 기능을 원하시면 알려주세요."
}
```

1. 결과


```
최종 답변:
SQL의 select 문은 데이터베이스에서 데이터를 조회할 때 사용하는 명령어입니다. 기본적인 문법은 다음과 같습니다:

SELECT 컬럼명1, 컬럼명2, ...
FROM 테이블명
WHERE 조건식;

예를 들어, 'employees' 테이블에서 이름(name)과 나이(age)를 조회하려면:

SELECT name, age
FROM employees;

특정 조건을 걸고 싶으면 WHERE 절을 사용합니다:

SELECT name, age
FROM employees
WHERE age > 30;

이 외에도 정렬(ORDER BY), 그룹화(GROUP BY) 등 다양한 옵션이 있습니다. 더 구체적인 예시나 기능을 원하시면 알려주세요.
[최종 답변]
최종 답변: SQL의 select 문은 데이터베이스에서 데이터를 조회할 때 사용하는 명령어입니다. 기본적인 문법은 다음과 같습니다:

SELECT 컬럼명1, 컬럼명2, ...
FROM 테이블명
WHERE 조건식;

예를 들어, 'employees' 테이블에서 이름(name)과 나이(age)를 조회하려면:

SELECT name, age
FROM employees;

특정 조건을 걸고 싶으면 WHERE 절을 사용합니다:

SELECT name, age
FROM employees
WHERE age > 30;

이 외에도 정렬(ORDER BY), 그룹화(GROUP BY) 등 다양한 옵션이 있습니다. 더 구체적인 예시나 기능을 원하시면 알려주세요.
```


## **결론**

1. 성능 좋은 모델을 활용 한다면 사용 가능해 보임
    - Qwen3-1.4b의 경우 Thinking mode를 사용해도 결과가 좋지 않음
2. 피드백 기능의 한계가 있어 보임
    - 따로 피드백 Tool을 만들어 처리해야 할 듯
3. 다양한 디자인 패턴으로 시도해볼만 한 듯
