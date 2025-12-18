"""ES 工具包配置与装配
定义 `ElasticsearchToolkitConfig`（工具行为配置）与 `ElasticsearchDatabaseToolkit`（生产 ES 工具集合）。
该工具包对标 SQLDatabaseToolkit，用于为 Agent 注入一组只读的 ES 操作工具。
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from elasticsearch import Elasticsearch
from langchain_core.tools import BaseTool

@dataclass
class ElasticsearchToolkitConfig:
    """Elasticsearch 工具配置
    字段：
    - index_allowlist：允许访问的索引白名单（None 表示不限制）
    - request_timeout：ES 请求超时（秒）
    - max_hits_cap：查询返回的最大命中数上限（全局兜底）
    - max_sample_docs：示例文档最大数量（采样上限）
    - enable_validate：是否启用 `_validate/query` 校验工具
    - enable_explain：是否启用 `_explain` 工具（需额外实现）
    """
    index_allowlist: Optional[Sequence[str]] = None
    request_timeout: float = 10.0
    max_hits_cap: int = 200
    max_sample_docs: int = 20
    enable_validate: bool = True
    enable_explain: bool = False

class ElasticsearchDatabaseToolkit:
    """对标 SQLDatabaseToolkit：负责生产一组 ES Tools（只读）
    根据配置动态组装索引/映射/字段能力/采样/查询/统计/校验等工具。
    若 `enable_explain` 或 LLM 检查启用，则需在工程中提供对应工具实现。
    """

    def __init__(self, es_client: Elasticsearch, llm=None, config: ElasticsearchToolkitConfig | None = None):
        """初始化工具包
        参数：
        - es_client：Elasticsearch 客户端实例
        - llm：可选，用于注入基于 LLM 的 DSL 检查工具
        - config：工具配置（为空时使用默认值）
        """
        self.es = es_client
        self.llm = llm
        self.config = config or ElasticsearchToolkitConfig()

    def get_tools(self) -> List[BaseTool]:
        """返回已装配的 ES 工具列表（只读）
        包含：索引列表、映射、字段能力、示例文档、查询、计数；
        可选：校验、解释、LLM DSL 检查（需实现）。
        """
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
        # 可选：如果提供 llm，则加一个 LLM-based DSL checker（对齐 QuerySQLCheckerTool 思路）
        if self.llm is not None:
            tools.append(QueryESDSLCheckerTool(self.llm, self.config))
        return tools
