# Audit_Case_QA

该项目是用langgraph搭建一个审计案例问答机器人，以满足针对用户权限下的案例查询和分析需求。

🟦 基础环境：python = 3.10
🟨 知识库：ElasticSearch = 9.2.2
🟩 Agent框架：langgraph = 1.0.4

## 快速执行
1. 获取术语字典：```python get_keywords_batch.py```
2. 合成案例：```python ./make_case/es_write.py```
3. ES入库：```python ./make_case/case_create.py```
4. 搭建智能体并运行：```python ./Case_QA.py```

## 1 背景简介
- 目前有一个审计案例库，存放了每个缺陷案例发生的时间、案例归属机构和缺陷案例内容
- 使用人员是集团和子公司、分支机构的审计人员，每个人员有自己的权限，例如集团可以查看全部案例，A省a市的只能看到对应的案例
- 希望建立一个问答机器人，能通过问答界面完成案例的查询和分析，并支持多轮问答。
  
## 2 术语处理
- 考虑到业务数据敏感性，这里用了大模型合成的查询，见```./data/用户查询指令集.xlsx```
- 根据分析，我们发现指令中存在审计专业术语和模糊表述，大模型在这两方面回答的不好，因此需要针对性构建术语词典。
- 基于上述内容，让大模型自主挖掘指令中的审计专业术语和模糊表述，然后根据这些内容生成术语定义。

执行命令生成术语。
```
python get_keywords_batch.py
```
具体逻辑如下：
- 使用```pandas```读取用户查询指令

    | id   | 问题	     | 类别     |
    | ---------- | ------------------------------------ | ------ |
    | 1 | 产险车险承保环节近年来有哪些虚增保费相关的典型审计发现？ | 车险相关 |
    | 2 | 近年来健康险未按规定划分保障责任的审计发现有哪些合规风险？ | 健康险相关 |
    | 3 | 近年来寿险业务系统与财务系统保费数据不一致的审计发现？ | 寿险相关 |
  
- 调用大模型（我是阿里公有云的qwen-plus），分析指令中的审计专业术语和模糊表述。
    ```
    # 这里修改大模型调用的url和api_key
    async def main():
        base_url = "your_base_url"
        api_key = "your_api_key"
    ```
- 保存成词表供后续使用，包括专业术语和模糊词汇及对应的定义。
  - 默认保存在```"./output/terms_dict.json"```和```"./output/illdefined_dict.json"```
  - 这里仅是大模型自己的定义，实际需要业务确认。
    ```
    {
        "销售适当性": "销售适当性是指金融机构在销售过程中必须确保其产品和服务符合客户的风险承受能力和其他相关情况。",
        "内部问责": "指企业对员工或机构在经营管理过程中违反法律法规、监管规定或内部制度的行为，所采取的内部纪律处分、经济处罚或其他处理措施。",
        "内勤": "指寿险公司内部从事管理、运营、支持等职能的正式员工，不直接参与保险销售。",
        "外勤": "指寿险公司中从事保险产品销售的代理人或营销人员，通常为合同制或代理制，直接面向客户开展业务。",
        ...
    }

    ```
## 3 案例入库
### 3.1 安装ES
- 详见ElasticSearch安装教程：<https://blog.csdn.net/qq_29539827/article/details/154174494>
- 下载ES安装包：我是当前最新版9.2.2，<https://www.elastic.co/downloads/elasticsearch>
- 下载分词器：配套9.2.2的分词器，<https://release.infinilabs.com/analysis-ik/stable/>。剪切```elasticsearch-analysis-ik-9.2.2```文件夹到```elasticsearch-9.2.2/plugins```下粘贴
- 回到```elasticsearch-9.2.2/bin```目录，双击```elasticsearch.bat```进行启动。
  - 如果报错，请确认文件夹路径无空格。
  - 还不行修改```elasticsearch-9.2.2/config/elasticsearch.yml```，设置```xpack.security.enabled: false```
- 安装成功后运行地址：<http://localhost:9200/>
    ```
    {
    "name" : "your pc name",
    "cluster_name" : "elasticsearch",
    "cluster_uuid" : "xxxxxxxxxxxxxx",
    "version" : {
        "number" : "9.2.2",
        "build_flavor" : "default",
        "build_type" : "zip",
        "build_hash" : "xxxxxxxxxxxxxx",
        "build_date" : "2025-xx-xxT08:06:51.614397514Z",
        "build_snapshot" : false,
        "lucene_version" : "10.3.2",
        "minimum_wire_compatibility_version" : "8.19.0",
        "minimum_index_compatibility_version" : "8.0.0"
    },
    "tagline" : "You Know, for Search"
    }
    ```

### 3.2 生成案例
```
python ./make_case/case_create.py
```
用大模型生成了一批，见```./data/保险行业审计缺陷案例数据集.xlsx```
| 发生时间   | 省     | 市     | 子公司 | 分支结构 | 缺陷内容                                                                   |
| ---------- | ------ | ------ | ------ | -------- | -------------------------------------------------------------------------- |
| 2023-09-14 | 浙江省 | 杭州市 | 健康险 | 余杭区   | 医疗险理赔档案未归档，34笔2023年理赔案未在30天内归档，违反档案管理规定     |
| 2023-10-21 | 湖北省 | 襄阳市 | 寿险   | 武昌区   | 重疾险理赔未审核治疗记录，16笔理赔案无完整治疗病历，存在虚假理赔风险       |
| 2023-03-07 | 广东省 | 珠海市 | 产险   | 宝安区   | 农业险承保未核实投保标的数量，27笔农险保单存在虚增种植面积，涉保费12.8万元 |

### 3.3 ES写入案例
```
python ./make_case/es_write.py
```
代码内主要步骤说明：

#### 3.3.1 创建ES索引
```
index_mapping = {
    "mappings": {
        "properties": {
            "发生时间": {
                "type": "date",
                "format": "yyyy-MM-dd||yyyy-MM-dd HH:mm:ss||epoch_millis",
            },
            "省": {"type": "keyword"},
            "市": {"type": "keyword"},
            "子公司": {"type": "keyword"},
            "分支结构": {"type": "keyword"},
            "缺陷内容": {"type": "text"},
        }
    }
}
```
#### 3.3.2 定义索引名称并写入
```
def create_index(index_name):
    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name, body=index_mapping)
        print(f"✅ 索引 {index_name} 创建成功")
    else:
        print(f"索引 {index_name} 已存在，无需重复创建。")


index_name = "audit_2025_cases"
create_index(index_name)
```
#### 3.3.3 读取数据转为ES写入格式
```
from elasticsearch.helpers import bulk

# 格式化数据为 ES Bulk 要求的格式
def format_es_actions(df, index_name):
    ...

    return actions


# 2.3 生成批量写入数据
es_actions = format_es_actions(df, index_name)
print(f"✅ 生成 {len(es_actions)} 条 ES 写入数据")
```
#### 3.3.4 批量写入ES
```
def bulk_write_to_es(client=es, actions=es_actions):
    try:
        # 批量写入（chunk_size：每批写入条数，max_retries：失败重试次数）
        success, failed = bulk(
            client,
            actions,
            chunk_size=100,  # 每批 100 条（根据 ES 性能调整，建议 500-2000）
            max_retries=3,  # 失败重试 3 次
        )
        print(f"✅ ES 批量写入成功：{success} 条")
        if failed:
            print(f"❌ 写入失败：{len(failed)} 条，失败详情：{failed}")
    except Exception as e:
        print(f"❌ ES 写入异常：{str(e)}")


# 执行批量写入
bulk_write_to_es(es, es_actions)
```
#### 3.3.5 检查ES数据
```
# 查查总数
count_resp = es.count(index=index_name)
print("文档总数：", count_resp)

# 看几条样例
search_resp = es.search(index=index_name, body={"size": 2, "query": {"match_all": {}}})
print("示例文档：")
for hit in search_resp["hits"]["hits"]:
    print(hit["_source"])
```
#### 3.3.6 更新ES向量化索引字段
```
mapping_update = {
    "properties": {"缺陷内容向量_qwen": {"type": "dense_vector", "dims": 1024}}
}

# 执行结构更新
resp = es.indices.put_mapping(index=index_name, body=mapping_update)
print("mapping 更新结果：", resp)
```
#### 3.3.7 案例向量化
调用公网嵌入模型进行向量化，请修改为自己的url和api_key
```
client = AsyncOpenAI(
    base_url="your_base_url",
    api_key="your_api_key",
)

async def process_single_case(case: str) -> dict:
    completion = await client.embeddings.create(
        model="text-embedding-v3", input=case, dimensions=1024
    )
    return completion.data[0].embedding
```

#### 3.3.8 批量向量化写入ES
```
updater = ESUpdater(
    es_client=es,
    index_name=index_name,
    embedding_func=process_batch_cases,
    vector_field="缺陷内容向量_qwen",  # 与ES mapping中定义的向量字段名一致
)

# 执行更新
asyncio.run(updater.batch_update_vectors())
```

#### 3.3.9 检查向量化更新情况
```
# 检查更新情况
search_resp = es.search(
    index=index_name,
    body={
        "size": 2,
        "_source": ["缺陷内容", "缺陷内容向量_qwen"],
        "query": {"match_all": {}},
    },
)
print("示例文档：")
for hit in search_resp["hits"]["hits"]:
    print(
        hit["_source"].get("缺陷内容", ""),
        hit["_source"].get("缺陷内容向量_qwen", "")[:4],
    )
```
## 4 搭建多轮问答机器人
基于langgraph搭建多轮问答机器人，主要包括指令理解（拒答判断、追问判断、意图识别、术语识别、指令改写）、信息检索、案例分析，核心在于**模块间的流转**和**上下文记忆**。
可以直接运行```python Case_QA.py```

### 4.1 配置项准备
主要包括ES库读取、大模型调用、嵌入调用（把用户查询转为向量）
```
# ES配置项，地址、接口、仓库
ES_HOST = "http://localhost:9200"
es = Elasticsearch(f"{ES_HOST}")
index_name = "audit_2025_cases"

# OpenAI的embedding接口
client = OpenAI(
    base_url="your_base_url",
    api_key="your_api_key",
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
    openai_api_key="your_api_key",
    openai_api_base="your_base_url",
)
```
### 4.2 定义agent状态
一个是不用修改的，如用户的个人信息、检索的配置等。一个是需要更新的agent对话状态管理。
```
# 固化信息
class Ctx(TypedDict):
    user_name: str
    user_province: str = ""
    user_city: str = ""
    user_district: str = ""
    user_subcompany: str = ""
    topk: int = 3
    threshold: float = 0.0

# 状态信息
class State(TypedDict):
    messages: Annotated[list, add_messages]
    memory: list
    query: str
    reject: str
    ask_further: str
    query_rewrite: str
    cases: list
    response: str
```

### 4.3 拒答模块
这个模块是输入历史上下文+本轮用户问题，判断是否拒答。**重点是需要在message中剔除掉本轮记录，因为这个模块是一次性的中间信息不属于上下文**。然后通过路由分流，拒答直接结束，不拒答就到下一个追问判断。
```
# 拒答判断
def reject(state: State):
    ...
    return {"reject": response.content}

# 路由模块
def should_analyse(state: State):
    if "回答" in state["reject"]:
        return "回答"
    else:
        ...
        state["response"] = "很抱歉，这个问题不是审计案例相关问题，我不作回答。"

        return "拒答"
```

### 4.4 追问模块
这个模块是输入历史上下文+本轮用户问题，判断是否是追问。这个模块的作用是判断是否为单纯的追问而不需要调ES检索，**需要在message中剔除掉本轮记录，因为这个模块是一次性的中间信息不属于上下文**。然后通过路由分流，追问的直接到分析，不是追问的就到下一个指令改写。
```
# 追问判断
def ask_further(state: State):
    ...
    return {"ask_further": new_or_old}

# 路由模块
def should_es_search(state: State):
    ...

    if "是" in state["ask_further"]:
        return "是"
    else:
        return "否"
```

### 4.5 指令改写模块
这个模块会先对用户查询，根据规则匹配出提及的专业术语和模糊表述。然后结合上下文、术语查询结果，将问题改写为一个信息全面的问题，作为后续的新用户查询。即ill-defined到well-defined。
```
def rewrite(state: State):
    ...
    return {"query_rewrite": query_rewrite}
```

### 4.6 ES检索模块
这个模块会根据用户权限做规则筛选，筛选出符合权限的案例。然后根据查询改写和案例向量相似度，做语义匹配召回topk个案例。
```
def es_search(state: State):
    ...
    return {"cases": cases}
```

### 4.7 案例分析模块
这个模块会输入历史上下文、查询改写和案例，让大模型进行分析作答。
```
def analyse(state: State):
    ...
    return {"response": response.content}
```

### 4.8 构建langgraph静态图
根据每个模块的功能，我们就比较清晰的构建出静态图。

```
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
              +-----------+
              | __start__ |
              +-----------+
                    *
                    *
                    *
                +------+
                | 拒答判断 |.
                +------+ ..
                .          ...
              ..              ...
             .                   ..
        +------+                   ..
        | 追问判断 |                    .
        +------+                    .
        .       .                   .
      ..         .                  .
     .            ..                .
+------+            .               .
| 指令改写 |            .               .
+------+            .               .
+------+            .               .
+------+            .               .
    *               .               .
    *               .               .
    *               .               .
    *               .               .
    *               .               .
+------+            .               .
| 案例检索 |          ..                .
+------+         .                  .
        *       .                   .
         **   ..                    .
           * .                      .
        +------+                   ..
        | 分析回答 |                 ..
        +------+              ...
                *          ...
                 **      ..
                   *   ..
              +---------+
              | __end__ |
              +---------+
```

## 5 实际效果

### 固定信息
```
context = {
    "user_province": "山东省",
    "user_city": "",
    "user_district": "",
    "user_subcompany": "产险",
    "topk": 3,
    "threshold": 0.0,
}
```

### 状态更新
langgraph自带的MemorySave仅支持state内部只有message，所以我们需要在每一轮自己维护好state传进去（仅需更新当前state的query），同时也可以将state以json的格式保存来长期固化。

### 问答效果
🤖 机器人：你好呀！

👤 你: 请帮我查一下农险近期缺陷案例有哪些？

👉是否拒答：回答

👉是否追问：否

👉指令改写：请帮我查询近三个月内农业保险相关的缺陷案例有哪些？

👉ES搜索结果：
{'took': 16, 'timed_out': False, '_shards': {'total': 1, 'successful': 1, 'skipped': 0, 'failed': 0}, 'hits': {'total': {'value': 10, 'relation': 'eq'}, 'max_score': 0.7024403, 'hits': [{'_index': 'audit_2025_cases', '_id': '25', '_score': 0.7024403, '_source': {'省': '山东省', '市': '潍坊市', '子公司': '产险', '缺陷内容': '农业险理赔未公示损失信息，22笔农险理 赔未按要求在村社公示，违反农险监管规定'}}, {'_index': 'audit_2025_cases', '_id': '270', '_score': 0.6900946, '_source': {'省': '山东省', '市': '淄博市', '子公司': '产险', '缺陷内容': '农业险承保未签订正式投保协议，19笔保单仅口头约定，无书面协议，违反承保流程'}}, {'_index': 'audit_2025_cases', '_id': '263', '_score': 0.6629813, '_source': {'省': '山东省', '市': '淄博市', '子公司': '产险', '缺陷内容': '短期意外险（产险）销售未提示免责条款，35名客户因未知晓免责拒赔投诉，违反销售规范'}}]}}

🤖 机器人：根据您提供的案例信息，近三个月内与农业保险相关的缺陷案例有以下两起：

1. **山东省潍坊市**（子公司：产险）
   缺陷内容：农业险理赔未公示损失信息。共发现22笔农险理赔未按监管要求在村社进行公示，违反了农业保险的相关监管规定。      

2. **山东省淄博市**（子公司：产险）
   缺陷内容：农业险承保未签订正式投保协议。涉及19笔保单，仅通过口头约定完成承保，缺乏书面投保协议，违反了农业保险的承保流程规范。

以上案例均涉及农业保险在承保或理赔环节的操作不合规问题，反映出在信息披露和流程规范化方面存在薄弱环节。

👤 你: 这些案例都是哪个市的？

👉是否拒答：回答

👉是否追问：是

🤖 机器人：这些农业保险相关的缺陷案例分别涉及以下两个市：

- **潍坊市**
- **淄博市**

其中，潍坊市和淄博市均属于山东省。

## 6 后续优化方向
- 为了确保检索效果，目前是基于ES模板，查询类型比较有限。后续应通过指令更细致的理解，让大模型自主撰写ES查询代码并执行；
- 缺乏闭环迭代，需要引入一个反思验证模块，针对当前模块调用失败、执行不好的，需要重复执行；
- 考虑到业务流程可控，目前只能算是一个带有分支的工作流。后续可以考虑把每个模块封装为mcp工具来调用，真正的实现plan-action-verify的自主思考agent；