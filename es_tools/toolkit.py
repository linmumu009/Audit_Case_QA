"""ES 工具包配置与装配
定义 `ElasticsearchToolkitConfig`（工具行为配置）与 `ElasticsearchDatabaseToolkit`（生产 ES 工具集合）。

该工具包对标 LangChain 的 SQLDatabaseToolkit：负责把一组只读 ES 工具装配成可注入 Agent 的 tools 列表。
核心设计目标：
- READ-ONLY：不提供任何写入能力
- SAFE：支持 index_allowlist、超时、返回条数上限等约束
- OPTIONAL-LLM: 若提供 llm，则追加 LLM-based DSL 检查工具（对齐 QuerySQLCheckerTool 的定位）
- OPTIONAL-EMBEDDINGS: 若提供 embeddings/embedding_fn，则追加“查询向量生成”工具与“向量检索”工具（推荐给 Agent 使用）

注意：所有工具均只读；若你们想启用 explain 等高级能力，需要显式打开配置开关。
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Any

from elasticsearch import Elasticsearch
from langchain_core.tools import BaseTool

from .tools import (
    ESListIndicesTool,
    ESGetMappingTool,
    ESFieldCapsTool,
    ESSampleDocsTool,
    ESSearchDSLTool,
    ESCountTool,
    ESValidateQueryTool,
    ESExplainTool,
    QueryESDSLCheckerTool,
    ESQueryVectorTool,
    ESVectorSearchTool,
)


@dataclass
class ElasticsearchToolkitConfig:
    """Elasticsearch 工具配置

    字段：
    - index_allowlist：允许访问的索引白名单（None 表示不限制）
    - request_timeout：ES 请求超时（秒）
    - max_hits_cap：查询返回的最大命中数上限（全局兜底）
    - max_sample_docs：示例文档最大数量（采样上限）
    - enable_validate：是否启用 `_validate/query` 校验工具
    - enable_explain：是否启用 `_explain` 工具（需要 doc_id）
    - enable_llm_dsl_checker：是否启用 LLM DSL Checker（需同时提供 llm）
    - enable_query_vector：是否启用“查询向量生成”工具（需同时提供 embeddings/embedding_fn）
    """

    index_allowlist: Optional[Sequence[str]] = None
    request_timeout: float = 10.0
    max_hits_cap: int = 200
    max_sample_docs: int = 20
    enable_validate: bool = True
    enable_explain: bool = False
    enable_llm_dsl_checker: bool = True
    enable_query_vector: bool = True
    enable_vector_search: bool = True


class ElasticsearchDatabaseToolkit:
    """对标 SQLDatabaseToolkit：负责生产一组 ES Tools（只读）。"""

    def __init__(
        self,
        es_client: Elasticsearch,
        llm=None,
        embeddings=None,
        embedding_fn: Optional[Callable[[str], List[float]]] = None,
        config: ElasticsearchToolkitConfig | None = None,
    ):
        """初始化工具包

        参数：
        - es_client：Elasticsearch 客户端实例
        - llm：可选，用于注入基于 LLM 的 DSL 检查工具
        - embeddings：可选，LangChain Embeddings（需实现 embed_query）
        - embedding_fn：可选，纯函数式 embedding(text)->vector（与 embeddings 二选一即可）
        - config：工具配置（为空时使用默认值）
        """
        self.es = es_client
        self.llm = llm
        self.embeddings = embeddings
        self.embedding_fn = embedding_fn
        self.config = config or ElasticsearchToolkitConfig()

    def get_tools(self) -> List[BaseTool]:
        """返回已装配的 ES 工具列表（只读）"""
        tools: List[BaseTool] = [
            ESListIndicesTool(self.es, self.config),
            ESGetMappingTool(self.es, self.config),
            ESFieldCapsTool(self.es, self.config),
            ESSampleDocsTool(self.es, self.config),
            ESSearchDSLTool(self.es, self.config),
            ESCountTool(self.es, self.config),
        ]

        if self.config.enable_validate:
            tools.append(ESValidateQueryTool(self.es, self.config))

        if self.config.enable_explain:
            tools.append(ESExplainTool(self.es, self.config))

        # 可选：如果提供 llm，则加一个 LLM-based DSL checker（对齐 QuerySQLCheckerTool 的定位）
        if self.config.enable_llm_dsl_checker and self.llm is not None:
            tools.append(QueryESDSLCheckerTool(self.es, self.llm, self.config))

                # 可选：如果提供 embeddings/embedding_fn，则增加“查询向量生成”工具
        if self.config.enable_query_vector and (self.embeddings is not None or self.embedding_fn is not None):
            tools.append(ESQueryVectorTool(self.embeddings, self.embedding_fn))

        # 可选：如果提供 embeddings/embedding_fn，则增加“向量检索”工具（推荐给 Agent 使用）
        if self.config.enable_vector_search and (self.embeddings is not None or self.embedding_fn is not None):
            tools.append(ESVectorSearchTool(self.es, self.config, self.embeddings, self.embedding_fn))

        return tools
