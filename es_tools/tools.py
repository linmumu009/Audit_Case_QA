"""Elasticsearch 只读工具集合（LangChain BaseTool）
提供如下工具：
- es_list_indices：列出可访问索引
- es_get_mapping：查看索引映射
- es_field_caps：查看字段能力（searchable/aggregatable）
- es_sample_docs：采样若干文档以理解字段/值
- es_search_dsl：执行只读的 Query DSL（带 size 与全局 cap 限制）
- es_count：执行只读计数
- es_validate_query：执行 `_validate/query`（可用时）
所有工具均遵循 allowlist 与请求超时等配置，面向 Agent 的安全可控查询。
"""
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from langchain_core.tools import BaseTool

def _assert_index_allowed(index: str, allowlist: Optional[List[str]]):
    """校验索引是否在允许列表中
    参数：
    - index：目标索引名
    - allowlist：允许访问的索引列表（None 表示不限制）
    异常：
    - ValueError：索引不在 allowlist 中时抛出
    """
    if allowlist and index not in allowlist:
        raise ValueError(f"Index '{index}' not allowed by allowlist.")

class ListIndicesInput(BaseModel):
    pattern: Optional[str] = Field(default=None)

class ESListIndicesTool(BaseTool):
    name = "es_list_indices"
    description = "List available indices (read-only). Use first."
    args_schema = ListIndicesInput
    # 构造：注入 ES 客户端与配置（只读）
    def __init__(self, es, config): super().__init__(); self.es=es; self.config=config
    def _run(self, pattern: Optional[str]=None) -> List[str]:
        """列出索引名称
        参数：
        - pattern：可选的通配符或前缀，用于筛选索引
        返回：
        - 索引名称列表（若设置 allowlist，则已过滤）
        """
        resp = self.es.cat.indices(index=pattern or "*", format="json", request_timeout=self.config.request_timeout)
        indices = [r["index"] for r in resp]
        if self.config.index_allowlist:
            indices = [i for i in indices if i in self.config.index_allowlist]
        return indices

class GetMappingInput(BaseModel):
    index: str

class ESGetMappingTool(BaseTool):
    name = "es_get_mapping"
    description = "Get mapping for an index."
    args_schema = GetMappingInput
    # 构造：注入 ES 客户端与配置（只读）
    def __init__(self, es, config): super().__init__(); self.es=es; self.config=config
    def _run(self, index: str) -> Dict[str, Any]:
        """获取索引的映射信息（mapping）
        参数：
        - index：目标索引名（需在 allowlist 内）
        返回：
        - ES `indices.get_mapping` 的原始响应字典
        """
        _assert_index_allowed(index, list(self.config.index_allowlist) if self.config.index_allowlist else None)
        return self.es.indices.get_mapping(index=index, request_timeout=self.config.request_timeout)

class FieldCapsInput(BaseModel):
    index: str
    fields: List[str] = Field(default_factory=lambda: ["*"])

class ESFieldCapsTool(BaseTool):
    name = "es_field_caps"
    description = "Get field capabilities (searchable/aggregatable). Useful for building correct aggregations."
    args_schema = FieldCapsInput
    # 构造：注入 ES 客户端与配置（只读）
    def __init__(self, es, config): super().__init__(); self.es=es; self.config=config
    def _run(self, index: str, fields: List[str]) -> Dict[str, Any]:
        """获取字段能力信息
        参数：
        - index：目标索引名（需在 allowlist 内）
        - fields：字段列表（支持通配符），内部拼接为逗号分隔
        返回：
        - ES `field_caps` 的原始响应字典
        """
        _assert_index_allowed(index, list(self.config.index_allowlist) if self.config.index_allowlist else None)
        return self.es.field_caps(index=index, fields=",".join(fields), request_timeout=self.config.request_timeout)

class SampleDocsInput(BaseModel):
    index: str
    size: int = Field(default=3, ge=1, le=20)

class ESSampleDocsTool(BaseTool):
    name = "es_sample_docs"
    description = "Sample a few documents (match_all) to understand fields/values."
    args_schema = SampleDocsInput
    # 构造：注入 ES 客户端与配置（只读）
    def __init__(self, es, config): super().__init__(); self.es=es; self.config=config
    def _run(self, index: str, size: int=3) -> List[Dict[str, Any]]:
        """采样文档（match_all）
        参数：
        - index：目标索引名（需在 allowlist 内）
        - size：采样数量（受 `max_sample_docs` 限制）
        返回：
        - 文档 `_source` 列表（截取至 size）
        """
        _assert_index_allowed(index, list(self.config.index_allowlist) if self.config.index_allowlist else None)
        size = min(size, self.config.max_sample_docs)
        resp = self.es.search(index=index, body={"query":{"match_all":{}}, "size": size},
                              request_timeout=self.config.request_timeout)
        return [h.get("_source", {}) for h in resp.get("hits", {}).get("hits", [])]

class SearchDSLInput(BaseModel):
    index: str
    body: Dict[str, Any]
    top_k: int = Field(default=10, ge=1, le=200)

class ESSearchDSLTool(BaseTool):
    name = "es_search_dsl"
    description = "Execute _search with Query DSL (read-only). Enforces size <= top_k and global cap."
    args_schema = SearchDSLInput
    # 构造：注入 ES 客户端与配置（只读）
    def __init__(self, es, config): super().__init__(); self.es=es; self.config=config
    def _run(self, index: str, body: Dict[str, Any], top_k: int=10) -> Dict[str, Any]:
        """执行只读查询（_search）
        安全策略：
        - 对 `size` 进行限制：`min(body.size, top_k, max_hits_cap)`
        - 校验索引访问权限
        参数：
        - index：目标索引名
        - body：Query DSL 请求体
        - top_k：本次请求的显示上限（最终受全局 cap 约束）
        返回：
        - ES `search` 的原始响应字典
        """
        _assert_index_allowed(index, list(self.config.index_allowlist) if self.config.index_allowlist else None)
        cap = min(int(top_k), self.config.max_hits_cap)
        safe_body = dict(body)
        safe_body["size"] = min(int(safe_body.get("size", cap)), cap)
        return self.es.search(index=index, body=safe_body, request_timeout=self.config.request_timeout)

class CountInput(BaseModel):
    index: str
    body: Dict[str, Any]

class ESCountTool(BaseTool):
    name = "es_count"
    description = "Execute _count with Query DSL (read-only)."
    args_schema = CountInput
    # 构造：注入 ES 客户端与配置（只读）
    def __init__(self, es, config): super().__init__(); self.es=es; self.config=config
    def _run(self, index: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """执行只读计数（_count）
        参数：
        - index：目标索引名（需在 allowlist 内）
        - body：Query DSL 请求体（通常只需包含 `query`）
        返回：
        - ES `count` 的原始响应字典
        """
        _assert_index_allowed(index, list(self.config.index_allowlist) if self.config.index_allowlist else None)
        return self.es.count(index=index, body=body, request_timeout=self.config.request_timeout)

class ValidateInput(BaseModel):
    index: str
    body: Dict[str, Any]

class ESValidateQueryTool(BaseTool):
    name = "es_validate_query"
    description = "Validate query via _validate/query if available."
    args_schema = ValidateInput
    # 构造：注入 ES 客户端与配置（只读）
    def __init__(self, es, config): super().__init__(); self.es=es; self.config=config
    def _run(self, index: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """校验 Query DSL（_validate/query）
        参数：
        - index：目标索引名（需在 allowlist 内）
        - body：Query DSL 请求体（一般仅需 `{"query": ...}`）
        返回：
        - ES `indices.validate_query` 的原始响应字典
        """
        _assert_index_allowed(index, list(self.config.index_allowlist) if self.config.index_allowlist else None)
        # 这里 body 通常只需要 {"query": ...}
        return self.es.indices.validate_query(index=index, body=body, request_timeout=self.config.request_timeout)
