from typing import TypedDict, Sequence
from langgraph.graph import StateGraph
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import START, END
from random_weather import  generate_weather_data

class AllState(TypedDict):
    messages: Sequence[str]

# 使用 Ollama 初始化模型
model = OllamaLLM(
    model="llama3.1",  # 替換為您需要的模型名稱，例如 'llama2-13b' 或其他 Ollama 支持的模型
    temperature=0,
    max_tokens=None,  # 默認不設限
    timeout=None,     # 根據需要設置請求超時
    max_retries=2     # 設置重試次數
)


# Agent 1 : 用於判斷使用者的問題是否跟天氣相關, 回答yes or no 利於流程進行
prompt_str = """
You are given one question. Determine if the question is weather-related.
Only reply with "yes" or "no".
Question: {user_query}
"""
prompt = ChatPromptTemplate.from_template(prompt_str)
classification_chain = prompt | model



# Agent 2  : 用於判斷地理位置  只會輸出單一位置消除其他不必要訊息 例如：「台北」、「高雄」, 
# 如果地理位置沒有在資料庫中回答「no_response」
weather_prompt_str = """
請從以下問題中提取台灣的城市名稱。
請僅回覆城市名稱，例如：「台北」、「高雄」。
如果問題中沒有城市名稱，請回覆「no_response」。
請勿輸出多餘的文字或解釋。

問題：{user_query}
"""



weather_prompt = ChatPromptTemplate.from_template(weather_prompt_str)
weather_chain = weather_prompt | model


# 輸入假設資料可以替換為真正天氣API
def get_taiwan_weather(city: str) -> str:
    """查詢台灣特定城市的天氣狀況。"""
    weather_data = generate_weather_data()
    print(weather_data)
    # 初始回覆都是暫時無資料, 除非輸入地點存在就會替換初始字串
    return f"{city}的天氣：{weather_data.get(city, '暫無資料')}"


def classify_question(state: AllState):
    """判斷問題是否與天氣相關，並輸出中間結果。"""
    user_query = state["messages"][-1]
    # 使用Ａgent 1 ： 判斷是否跟天氣相關 output : yes or no 
    response = classification_chain.invoke({"user_query": user_query})
    print(f"【Stage: classify_question】User Query: {user_query}")
    print(f"【Stage: classify_question】Classification Response: {response.strip()}")
    return {"messages": state["messages"] + [response.strip()]}


def is_weather_related(state: AllState):
    """根據分類結果決定邏輯分支，並輸出中間結果。"""
    # 取得 classify_question() 分類結果 yes or no, 如果是yes就extract_city,  no 就到結束節點
    classification_result = state["messages"][-1].strip().lower()
    next_step = "extract_city" if classification_result == "yes" else "end"
    print(f"【Stage: is_weather_related】Classification Result: {classification_result}")
    print(f"【Stage: is_weather_related】Next Step: {next_step}")
    return {"messages": state["messages"], "next_step": next_step}

    
def query_next_step(state: AllState):
    """根據 'next_step' 字段決定下一步。"""
    next_step = state.get("next_step", None)
    if not next_step or next_step not in ["extract_city", "end"]:
        next_step = "end"  # 確保默認值為 'end'
    return next_step



def extract_city(state: AllState):
    """從問題中提取城市名稱，並輸出中間結果。"""
    # 經過流程處理後將user的問題萃取出地理位置
    user_query = state["messages"][0]
    print(f"【Stage: extract_city】User Query: {user_query}")

    # 檢查地理位置是否在數據中有就輸出位置 沒有就回no_response
    response = weather_chain.invoke({"user_query": user_query}).strip()
    print(f"【Stage: extract_city】Extracted Response: {response}")
    if not response or response.lower() in ["", "no_response"]:
        response = "no_response"
    return {"messages": state["messages"] + [response]}


def provide_weather(state: AllState):
    """提供天氣資訊，並輸出中間結果。"""
    city_name = state["messages"][-1].strip()
    print(f"【Stage: provide_weather】City Name: {city_name}")
    if city_name == "no_response":
        response_message = "無法識別城市名稱，請提供有效的問題。"
    else:
        # 經過判斷是否跟天氣有關 然後地名萃取  地名是否存在數據中 最後回覆
        response_message = get_taiwan_weather(city_name)
    print(f"【Stage: provide_weather】Weather Info: {response_message}")
    return {"messages": state["messages"] + [response_message]}



# 更新 StateGraph 邏輯
graph_builder = StateGraph(AllState)
graph_builder.add_node("classify_question", classify_question)
graph_builder.add_node("is_weather_related", is_weather_related)
graph_builder.add_node("extract_city", extract_city)
graph_builder.add_node("provide_weather", provide_weather)

# 添加新的分支決策邏輯
graph_builder.add_edge(START, "classify_question")
graph_builder.add_edge("classify_question", "is_weather_related")
graph_builder.add_conditional_edges(
    "is_weather_related",
    query_next_step,  # 根據返回值決定跳轉
    {"extract_city": "extract_city", "end": END}  # 定義有效分支
)

graph_builder.add_edge("extract_city", "provide_weather")
graph_builder.add_edge("provide_weather", END)

graph_builder.set_entry_point("classify_question")
graph_builder.set_finish_point("provide_weather")

app = graph_builder.compile()

# 繪製圖表
    # image_data = app.get_graph().draw_mermaid_png()  # 二進制資料
    # with open('LangGraph_workflow.png', 'wb') as f:
    #     f.write(image_data)



test_inputs = [
    "我想去台北旅遊 想知道天氣"
]

for query in test_inputs:
    init_state = {"messages": [query]}
    response = app.invoke(init_state)
    print(f"Input: {query}")
    for message in response["messages"]:
        print(f" -> {message}")

