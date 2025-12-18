from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.runtime import get_runtime
from langchain_openai import ChatOpenAI
from elasticsearch import Elasticsearch
from openai import OpenAI
from pathlib import Path
from types import SimpleNamespace

from es_tools.tools import ESVectorSearchTool

# ES配置项，地址、接口、仓库
ES_HOST = "http://localhost:9200"
es = Elasticsearch(f"{ES_HOST}")
index_name = "audit_2025_cases"

def load_llm_api_key():
    """从配置文件中加载API密钥"""
    config_path = Path("config/qwen_long_api_key.txt")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            api_key = f.read().strip()
        if not api_key:
            raise ValueError("API密钥文件为空")
        return api_key
    except FileNotFoundError:
        raise FileNotFoundError(f"API密钥文件不存在: {config_path}")
    except Exception as e:
        raise Exception(f"读取API密钥失败: {e}")

api_key = load_llm_api_key()
# def load_embed_api_key():
#     """从配置文件中加载API密钥"""
#     config_path = Path("config/guiji.txt")
#     try:
#         with open(config_path, 'r', encoding='utf-8') as f:
#             api_key = f.read().strip()
#         if not api_key:
#             raise ValueError("API密钥文件为空")
#         return api_key
#     except FileNotFoundError:
#         raise FileNotFoundError(f"API密钥文件不存在: {config_path}")
#     except Exception as e:
#         raise Exception(f"读取API密钥失败: {e}")
        
# em_api_key = load_embed_api_key()
# OpenAI的embedding接口
client = OpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=api_key,
)

# 文本转向量
def text2vec(case: str) -> dict:
    completion = client.embeddings.create(
        model="text-embedding-v3", input=case, dimensions=1024
    )
    return completion.data[0].embedding

# 大模型接口
llm = ChatOpenAI(
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=api_key,
)

# 固化信息
class Ctx(TypedDict):
    """
    不更改的信息，包括客户信息、检索topk、检索阈值
    """

    user_name: str
    user_province: str = ""
    user_city: str = ""
    user_district: str = ""
    user_subcompany: str = ""
    topk: int = 3
    threshold: float = 0.0


# 状态信息
class State(TypedDict):
    """
    保持更新的状态定义
    """

    messages: Annotated[list, add_messages]
    memory: list
    query: str
    reject: str
    ask_further: str
    query_rewrite: str
    cases: list
    response: str


# 拒答判断
def reject(state: State):
    """
    判断是否拒绝回答，更新在状态的"reject"字段
    1. 返回“回答”：目标明确且不违规
    2. 返回“拒答”：目标不明确或违规
    """
    # print(f"\n=====我们看看进入到reject的状态是啥样：=====\n{state}\n")
    query = state["query"]
    messages = state["messages"]
    prompt = f"请综合历史上下文，判断当前问题的目标是否明确和不违规，如果是能通过查询案例得到答案的、返回“回答”，否则返回“拒答”。请不要轻易拒答，只有在问题非常不明确、上下文完全没信息的时候才返回“拒答”。用户当前的问题是：{query}。"
    messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    response = llm.invoke(messages)
    messages.pop()  # 剔除中间判断信息

    # 过程记录
    state["memory"].extend(
        [
            {
                "role": "reject",
                "content": prompt,
            },
            {
                "role": "assistant",
                "content": response.content,
            },
        ]
    )
    return {"reject": response.content}


# 分析条件
def should_analyse(state: State):
    """
    路由模块，判断是否需要分析回答
    """
    # print(f"\n=====我们看看进入到should_analyse的状态是啥样：=====\n{state}\n")
    # 目标明确且不违规
    print(f"\n👉是否拒答：{state['reject']}")
    if "回答" in state["reject"]:
        return "回答"
    else:
        # 拒答直接结束，所以补上原始查询和拒答话术
        state["messages"].extend(
            [
                {
                    "role": "user",
                    "content": state["query"],
                },
                {
                    "role": "assistant",
                    "content": "很抱歉，这个问题不是审计案例相关问题，我不作回答。",
                },
            ]
        )
        state["response"] = "很抱歉，这个问题不是审计案例相关问题，我不作回答。"

        return "拒答"


# 追问判断
def ask_further(state: State):
    # print(f"\n=====我们看看进入到ask_further的状态是啥样：=====\n{state}\n")
    """
    分析是否是一个新问题，还是旧问题的追问。
    新问题，后面要走改写+es检索
    旧问题追问，直接走分析
    """
    query = state["query"]
    # 汇总意图+术语
    prompt = f"请判断用户的问题是否是追问。如果是一个新问题、需要进一步从数据库查找案例，则不是追问，返回“否”。如果该问题是延续之前上下文的补充提问，则是追问，返回“是”。请谨慎回答“是”，因为追问将不再检索案例库，直接基于上下文回答。用户本轮的问题是：{query}，"

    message = state["messages"]
    message.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    # 追问判断
    new_or_old = llm.invoke(message).content
    state["memory"].extend(
        [
            {
                "role": "ask_further",
                "content": prompt,
            },
            {
                "role": "assistant",
                "content": new_or_old,
            },
        ]
    )
    print(f"\n👉是否追问：{new_or_old}")

    message.pop()  # 剔除过程交互记录
    return {"ask_further": new_or_old}


def should_es_search(state: State):
    """
    路由模块，判断是否需要从ES搜索案例
    """
    # 是追问，无需从ES搜索，直接补上原始查询
    if "是" in state["ask_further"]:
        # 追问的，就在消息中补上原始查询
        state["messages"].append(
            {
                "role": "user",
                "content": state["query"],
            }
        )
        return "是"
    else:
        return "否"


# 模糊词表
illdefined_schema = {
    "近期": "近三个月",
}
# 专业术语表
professional_schema = {
    "销售适当性": "销售适当性是指在销售活动中，销售的保险产品是否符合客户风险需求。",
}


# 指令理解
def rewrite(state: State):
    """
    1. 意图理解
    2. 专业术语搜索
    3. 模糊词语搜索
    4. 指令改写，更新在状态的"query_rewrite"字段
    """
    query = state["query"]
    # 专业术语搜索
    professional_terms = {}
    for term in professional_schema.keys():
        if term in query:
            professional_terms.update({term: professional_schema[term]})
    # 模糊词语搜索
    illdefined_terms = {}
    for term in illdefined_schema.keys():
        if term in query:
            illdefined_terms.update({term: illdefined_schema[term]})
    # 汇总意图+术语
    prompt = f"你是一个指令理解模型，你的任务是参考上下文并结合用户当前的指令，将其改写为信息全面的指令，只需要输出你改写后的指令。用户的指令是：{query}。其中相关专业术语定义如下：{professional_terms}。模糊表述定义如下：{illdefined_terms}。"

    message = state["messages"]
    message.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    # 指令改写
    query_rewrite = llm.invoke(message).content
    print(f"\n👉指令改写：{query_rewrite}")
    message.pop()  # 剔除过程交互记录，后续将查询案例库形成完整的分析指令

    state["memory"].extend(
        [
            {
                "role": "professional_terms",
                "content": professional_terms,
            },
            {
                "role": "illdefined_terms",
                "content": illdefined_terms,
            },
            {
                "role": "query_rewrite",
                "content": query_rewrite,
            },
        ]
    )

    return {"query_rewrite": query_rewrite}


# ES案例搜索
def es_search(state: State):
    """
    ES案例取数模块
    1. 根据用户权限做规则筛选
    2. 根据查询做语义筛选
    """
    # 获取固化的信息
    rt = get_runtime(Ctx)
    user_province = rt.context.get("user_province", "")
    user_city = rt.context.get("user_city", "")
    user_district = rt.context.get("user_district", "")
    user_subcompany = rt.context.get("user_subcompany", "")
    topk = rt.context.get("topk", 5)
    threshold = rt.context.get("threshold", 0.0)

    query_rewrite = state["query_rewrite"]
    query_vec = text2vec(query_rewrite)

    if (
        user_province == ""
        and user_city == ""
        and user_district == ""
        and user_subcompany == ""
    ):  # 审计总部权限
        query_es = {"match_all": {}}
    else:
        # 机构权限，只能查看对应的案例
        must_clauses = [{"term": {"子公司": user_subcompany}}]
        if user_province != "":
            must_clauses.append({"term": {"省": user_province}})
        if user_city != "":
            must_clauses.append({"term": {"市": user_city}})
        if user_district != "":
            must_clauses.append({"term": {"分支机构": user_district}})
        query_es = {
            "bool": {
                "must": must_clauses,
            },
        }
    tool_config = SimpleNamespace(
        index_allowlist=None,
        request_timeout=10.0,
        max_hits_cap=200,
    )
    tool = ESVectorSearchTool(
        es=es,
        config=tool_config,
        embeddings=None,
        embedding_fn=text2vec,
    )
    result = tool._run(
        index=index_name,
        query_text=query_rewrite,
        vector_field="缺陷内容向量_qwen",
        k=topk,
        num_candidates=max(topk * 5, 50),
        filter=query_es,
        source_includes=["子公司", "省", "市", "分支机构", "缺陷内容"],
    )
    hits = result.get("hits", {}).get("hits", [])
    print(f"\nES命中: {len(hits)}")
    for i, hit in enumerate(hits[: min(topk, 3)], 1):
        src = hit.get("_source", {})
        score = hit.get("_score", 0.0)
        region = f"{src.get('省', '')}/{src.get('市', '')}/{src.get('分支机构', '')}"
        company = src.get("子公司", "")
        text = src.get("缺陷内容", "")
        snippet = (text[:60] + "…") if isinstance(text, str) and len(text) > 60 else text
        print(f"{i}. {company} {region} | {score:.3f} | {snippet}")

    # 汇总案例和相似度得分
    cases = []
    for case_info in result["hits"]["hits"]:
        case = case_info["_source"]
        score = case_info.get("_score", 0.0)
        if score > threshold:
            cases.append(case)  # 先不考虑阈值

    return {"cases": cases}


# 案例分析
def analyse(state: State):
    """
    分析回答
    """
    # print(f"\n=====我们看看进入到analyse的状态是啥样：=====\n{state}\n")
    messages = state["messages"]

    # 追问就用原始问题。非追问用问题改写和case内容。
    if state["ask_further"] == "是":
        messages.append(
            {
                "role": "user",
                "content": f"请根据上下文信息，回答用户问题。用户问题为：{state['query']}。",
            }
        )
    else:
        messages.append(
            {
                "role": "user",
                "content": f"请根据上下文信息，结合相关案例回答用户问题。相关案例为：{state['cases']}，用户问题为：{state['query_rewrite']}",
            }
        )

    response = llm.invoke(messages)
    messages.append(
        {
            "role": "assistant",
            "content": response.content,
        }
    )

    state["memory"].extend(
        [
            {
                "role": "analyse",
                "content": messages[-1]["content"],
            },
        ]
    )

    return {"response": messages[-1]["content"]}


# 建立静态图
graph = StateGraph(State)
graph.add_node("拒答判断", reject)
graph.add_node("追问判断", ask_further)
graph.add_node("指令改写", rewrite)
graph.add_node("案例检索", es_search)
graph.add_node("分析回答", analyse)
graph.add_edge(START, "拒答判断")
graph.add_conditional_edges(
    "拒答判断", should_analyse, {"拒答": END, "回答": "追问判断"}
)
graph.add_conditional_edges(
    "追问判断", should_es_search, {"是": "分析回答", "否": "指令改写"}
)
graph.add_edge("指令改写", "案例检索")
graph.add_edge("案例检索", "分析回答")
graph.add_edge("分析回答", END)

# 编译静态图
app = graph.compile()

# 粗略可视化
app.get_graph().print_ascii()


if __name__ == "__main__":
    """
    请帮我查一下农险近期缺陷案例有哪些？
    这些案例都是哪个省的？
    """
    # 固话agent状态，也可用于后续保存和载入
    state = {
        "messages": [],
        "memory": [],
        "query": "",
        "reject": "",
        "query_rewrite": "",
        "response": "",
    }
    context = {
        "user_province": "山东省",
        "user_city": "",
        "user_district": "",
        "user_subcompany": "产险",
        "topk": 3,
        "threshold": 0.0,
    }

    print("\n🤖 机器人：你好呀！")

    while True:
        # 获取用户输入
        user_input = input("\n👤 你: ")

        # 如果输入 exit 就结束
        if user_input.lower() == "exit":
            print("\n🤖 机器人: 再见！")
            break

        # 更新agent状态，替换为本轮用户查询
        state["query"] = user_input
        # print(state)
        state = app.invoke(state, context=context)
        print(f"\n🤖 机器人：{state.get('response', '本轮拒答')}")
        # print(f"\n=====我们看看结束的状态是啥样：=====\n{state}\n")




