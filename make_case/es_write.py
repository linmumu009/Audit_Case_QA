from elasticsearch import Elasticsearch

ES_HOST = "http://localhost:9200"
# ============= 1 创建ES索引 =============

# 1.1 连接本地 ES
es = Elasticsearch(f"{ES_HOST}")

# 1.2 定义索引结构（mapping）
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


# 1.3 定义索引名称
def create_index(index_name):
    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name, body=index_mapping)
        print(f"✅ 索引 {index_name} 创建成功")
    else:
        print(f"索引 {index_name} 已存在，无需重复创建。")


index_name = "audit_2025_cases"
create_index(index_name)

# ============= 2 原始案例数据改造 =============

# 2.1 读取 Excel 文件
import pandas as pd

df = pd.read_excel("./output/保险行业审计缺陷案例数据集.xlsx")

from elasticsearch.helpers import bulk


# 2.2 格式化数据为 ES Bulk 要求的格式
def format_es_actions(df, index_name):
    actions = []
    for idx, row in df.iterrows():
        # 构建单条数据（_id 用 order_id 确保唯一，避免重复写入）
        action = {
            "_index": index_name,
            "_id": idx,  # 可选：用业务唯一 ID 作为 ES 文档 ID
            "_source": row.to_dict(),  # 一行数据转为字典（字段名需与 ES 映射一致）
        }
        actions.append(action)
    return actions


# 2.3 生成批量写入数据
es_actions = format_es_actions(df, index_name)
print(f"✅ 生成 {len(es_actions)} 条 ES 写入数据")


# ============= 3 写入ES =============
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

# ============= 4 检查ES数据 =============
# 4.1 查查总数
count_resp = es.count(index=index_name)
print("文档总数：", count_resp)

# 4.2看几条 sample
search_resp = es.search(index=index_name, body={"size": 2, "query": {"match_all": {}}})
print("示例文档：")
for hit in search_resp["hits"]["hits"]:
    print(hit["_source"])

# ============= 5 补充向量化 =============
# 5.1 更新 mapping 增加向量字段
mapping_update = {
    "properties": {"缺陷内容向量_qwen": {"type": "dense_vector", "dims": 1024}}
}

# 5.2 执行结构更新
resp = es.indices.put_mapping(index=index_name, body=mapping_update)
print("mapping 更新结果：", resp)

# 5.3 案例向量化
from openai import AsyncOpenAI
import asyncio

client = AsyncOpenAI(
    base_url="your_base_url",
    api_key="your_api_key",
)

# 测试下
# from openai import OpenAI
# input_text = "衣服的质量杠杠的"
# completion = client.embeddings.create(
#     model="text-embedding-v3",
#     input=input_text,
#     dimensions=1024
# )


async def process_single_case(case: str) -> dict:
    completion = await client.embeddings.create(
        model="text-embedding-v3", input=case, dimensions=1024
    )
    return completion.data[0].embedding


async def process_batch_cases(cases: list[str]) -> list[dict]:
    semaphore = asyncio.Semaphore(10)  # 限制并发数量为10

    async def bounded_task(case: str) -> dict:
        async with semaphore:
            return await process_single_case(case)

    tasks = [bounded_task(case) for case in cases]
    results = await asyncio.gather(*tasks)

    print(f"✅ 生成 {len(results)} 条向量数据")
    # print(results[0][:3])

    return results


# 5.4 更新ES

from elasticsearch.helpers import bulk
import asyncio


class ESUpdater:
    def __init__(self, es_client, index_name, embedding_func, vector_field="案例向量"):
        """
        初始化ES更新器
        :param es_client: Elasticsearch客户端实例
        :param index_name: 索引名称
        :param embedding_func: 生成向量的异步函数（如process_batch_cases）
        :param vector_field: 存储向量的字段名
        """
        self.es = es_client
        self.index_name = index_name
        self.embedding_func = embedding_func
        self.vector_field = vector_field
        self.batch_size = 10  # 可根据ES性能调整

    def batch_get_docs(self, scroll_time="5m"):
        """
        批量滚动获取ES文档（含_id和案例字段）
        :param scroll_time: 滚动会话有效期
        """
        # 初始查询获取第一批数据
        query = {
            "query": {"match_all": {}},
            "size": self.batch_size,
            "_source": ["缺陷内容"],  # 只获取需要的字段
        }
        response = self.es.search(index=self.index_name, body=query, scroll=scroll_time)
        scroll_id = response["_scroll_id"]
        hits = response["hits"]["hits"]

        while hits:
            # 提取批次数据（id + 案例文本）
            batch = [
                {
                    "doc_id": hit["_id"],
                    "case_text": hit["_source"].get(
                        "缺陷内容", ""
                    ),  # 处理字段可能不存在的情况
                }
                for hit in hits
            ]
            yield batch

            # 获取下一批数据
            response = self.es.scroll(scroll_id=scroll_id, scroll=scroll_time)
            scroll_id = response["_scroll_id"]
            hits = response["hits"]["hits"]

    async def batch_update_vectors(self):
        """批量更新向量到ES"""
        total_updated = 0

        for batch in self.batch_get_docs():
            # 过滤空文本
            valid_batch = [item for item in batch if item["case_text"].strip()]
            if not valid_batch:
                continue

            # 批量获取向量
            case_texts = [item["case_text"] for item in valid_batch]
            vectors = await self.embedding_func(case_texts)
            print("case_texts", len(case_texts))
            print(vectors[0][:4])

            # 构建批量更新操作
            actions = []
            for i, item in enumerate(valid_batch):
                action = {
                    "_op_type": "update",
                    "_index": self.index_name,
                    "_id": item["doc_id"],
                    "doc": {self.vector_field: vectors[i]},
                }
                actions.append(action)

            # 执行批量更新
            success, failed = bulk(
                self.es, actions, refresh="wait_for", raise_on_error=False
            )
            total_updated += success

            print(
                f"批次更新完成 - 成功: {success}, 失败: {len(failed)}, 累计成功: {total_updated}"
            )

            if failed:
                print(f"失败详情: {failed}")

        print(f"所有文档向量更新完成，累计成功更新 {total_updated} 条")


updater = ESUpdater(
    es_client=es,
    index_name=index_name,
    embedding_func=process_batch_cases,
    vector_field="缺陷内容向量_qwen",  # 与ES mapping中定义的向量字段名一致
)

# 执行更新
asyncio.run(updater.batch_update_vectors())

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


# ============= 6 ES检索 =============
def es_search(query):
    query_vec = asyncio.run(process_single_case(query))
    search_body = {
        "size": 1,  # 只取相似度最高的1条
        "_source": [
            "缺陷内容",
            "缺陷内容向量_qwen",
            "_score",
        ],  # 只返回需要的字段，减少数据传输
        "query": {
            "script_score": {
                "query": {"match_all": {}},  # 必需：基础查询（匹配所有文档）
                "script": {
                    "source": "cosineSimilarity(params.query_vec, '缺陷内容向量_qwen')",
                    "params": {
                        "query_vec": query_vec,  # 传入查询向量
                    },
                },
            }
        },
        "sort": [{"_score": {"order": "desc"}}],  # 按相似度得分降序
    }
    resp = es.search(index=index_name, body=search_body)
    return resp["hits"]["hits"]

# 测试下向量相似度检索
query = "万能险初始费用扣除比例错误，24笔保单多扣初始费用，合计5.1万元"
result = es_search(query)
for hit in result:
    print(
        hit["_source"].get("缺陷内容", ""),
        hit["_source"].get("缺陷内容向量_qwen", "")[:4],
        hit["_score"],
    )
