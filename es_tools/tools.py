"""ES Tools（只读）
该文件提供与 Elasticsearch 交互的一组 BaseTool，实现“列索引 / 看 mapping / field_caps / 采样 / DSL 查询 / count / validate / explain”等能力，
并补齐：
- QueryESDSLCheckerTool：参考 LangChain QuerySQLCheckerTool 的定位，使用 LLM 进行 DSL 纠错/规范化（只返回 JSON）
- ESQueryVectorTool：生成“查询向量”（默认不回传全量向量，避免 token 爆炸；需要时可 return_vector=True）
- ESVectorSearchTool：推荐给 Agent 使用的向量检索工具（内部完成向量生成 + ES kNN 查询）
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, Type

from elasticsearch import Elasticsearch

try:
    from langchain_core.pydantic_v1 import BaseModel, Field  # type: ignore
except Exception:  # noqa: BLE001
    try:
        from pydantic import BaseModel, Field  # type: ignore
    except Exception:  # noqa: BLE001
        class BaseModel:  # type: ignore
            pass

        def Field(default=None, description: str | None = None):  # type: ignore
            return default


class BaseTool:  # type: ignore
    name: str = ""
    description: str = ""
    args_schema: Any | None = None

    def __init__(self, *args, **kwargs):
        pass

    def _run(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError

try:
    from langchain_core.prompts import ChatPromptTemplate  # type: ignore
    from langchain_core.output_parsers import StrOutputParser  # type: ignore
except Exception:  # noqa: BLE001
    ChatPromptTemplate = None  # type: ignore
    StrOutputParser = None  # type: ignore


# -----------------------------
# shared helpers
# -----------------------------

def _assert_index_allowed(index: str, allowlist: Optional[Sequence[str]]):
    if allowlist is None:
        return
    if index not in set(allowlist):
        raise ValueError(f"Index '{index}' is not allowed. Allowed: {list(allowlist)}")


def _flatten_mapping_fields(properties: Dict[str, Any], prefix: str = "", limit: int = 400) -> Dict[str, str]:
    """将 mapping.properties 展平成 {field_path: type} 的摘要，避免把完整 mapping 喂给 LLM。"""
    out: Dict[str, str] = {}

    def _walk(props: Dict[str, Any], pfx: str):
        nonlocal out
        if len(out) >= limit:
            return
        for field, meta in props.items():
            if len(out) >= limit:
                return
            fqn = f"{pfx}{field}" if pfx else field
            ftype = meta.get("type", "object")
            out[fqn] = ftype

            # multi-fields: fields.keyword, fields.raw ...
            fields = meta.get("fields") or {}
            for sub, sub_meta in fields.items():
                if len(out) >= limit:
                    return
                out[f"{fqn}.{sub}"] = sub_meta.get("type", "object")

            # nested/object recursion
            child_props = meta.get("properties") or {}
            if isinstance(child_props, dict) and child_props:
                _walk(child_props, f"{fqn}.")
    _walk(properties or {}, prefix)
    return out


def _mapping_summary(es: Elasticsearch, index: str, timeout: float, max_fields: int = 400) -> Dict[str, Any]:
    """获取 mapping 并压缩为字段类型摘要。"""
    resp = es.indices.get_mapping(index=index, request_timeout=timeout)
    # resp: {index_name: {"mappings": {...}}}
    entry = resp.get(index) or next(iter(resp.values()))
    mappings = (entry or {}).get("mappings") or {}
    props = mappings.get("properties") or {}
    return {
        "index": index,
        "field_types": _flatten_mapping_fields(props, limit=max_fields),
    }


def _as_pretty_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)


def _safe_json_loads(s: str) -> Any:
    """尽可能把字符串解析为 JSON（容错：去掉代码块标记）。"""
    s = s.strip()
    s = s.replace("```json", "```").replace("```", "").strip()
    return json.loads(s)


# -----------------------------
# base config typing
# -----------------------------

class _ToolkitConfigProto(BaseModel):
    """为避免在 tools.py 引入 toolkit.py（循环依赖），这里用一个最小协议类做类型提示。"""

    index_allowlist: Optional[Sequence[str]] = None
    request_timeout: float = 10.0
    max_hits_cap: int = 200
    max_sample_docs: int = 20


# -----------------------------
# Tool 1: list indices
# -----------------------------

class ESListIndicesTool(BaseTool):
    name: str = "es_list_indices"
    description: str = "List indices in Elasticsearch cluster (read-only). Input is optional; output is a list of index names."

    def __init__(self, es: Elasticsearch, config: Any):
        super().__init__()
        self.es = es
        self.config = config

    def _run(self, _: str = "") -> List[str]:
        indices = self.es.indices.get_alias(index="*")
        names = sorted(indices.keys())
        if self.config.index_allowlist:
            allow = set(self.config.index_allowlist)
            names = [n for n in names if n in allow]
        return names


# -----------------------------
# Tool 2: get mapping
# -----------------------------

class ESGetMappingInput(BaseModel):
    index: str = Field(..., description="Index name")


class ESGetMappingTool(BaseTool):
    name: str = "es_get_mapping"
    description: str = "Get index mapping. Input: {index}. Output: mapping JSON (read-only)."
    args_schema: Type[BaseModel] = ESGetMappingInput

    def __init__(self, es: Elasticsearch, config: Any):
        super().__init__()
        self.es = es
        self.config = config

    def _run(self, index: str) -> Dict[str, Any]:
        _assert_index_allowed(index, self.config.index_allowlist)
        return self.es.indices.get_mapping(index=index, request_timeout=self.config.request_timeout)


# -----------------------------
# Tool 3: field caps
# -----------------------------

class ESFieldCapsInput(BaseModel):
    index: str = Field(..., description="Index name")
    fields: str = Field("*", description="Field pattern, default '*'")


class ESFieldCapsTool(BaseTool):
    name: str = "es_field_caps"
    description: str = "Get field capabilities for an index. Input: {index, fields}. Output: field_caps JSON (read-only)."
    args_schema: Type[BaseModel] = ESFieldCapsInput

    def __init__(self, es: Elasticsearch, config: Any):
        super().__init__()
        self.es = es
        self.config = config

    def _run(self, index: str, fields: str = "*") -> Dict[str, Any]:
        _assert_index_allowed(index, self.config.index_allowlist)
        return self.es.field_caps(
            index=index,
            fields=fields,
            request_timeout=self.config.request_timeout,
        )


# -----------------------------
# Tool 4: sample docs
# -----------------------------

class ESSampleDocsInput(BaseModel):
    index: str = Field(..., description="Index name")
    size: int = Field(5, description="How many docs to sample (capped by config.max_sample_docs)")


class ESSampleDocsTool(BaseTool):
    name: str = "es_sample_docs"
    description: str = "Sample documents from an index for schema understanding. Input: {index, size}. Output: hits (read-only)."
    args_schema: Type[BaseModel] = ESSampleDocsInput

    def __init__(self, es: Elasticsearch, config: Any):
        super().__init__()
        self.es = es
        self.config = config

    def _run(self, index: str, size: int = 5) -> Dict[str, Any]:
        _assert_index_allowed(index, self.config.index_allowlist)
        size = min(int(size), int(getattr(self.config, "max_sample_docs", 20)))
        body = {"size": size, "query": {"match_all": {}}}
        return self.es.search(index=index, body=body, request_timeout=self.config.request_timeout)


# -----------------------------
# Tool 5: search by DSL
# -----------------------------

class ESSearchDSLInput(BaseModel):
    index: str = Field(..., description="Index name")
    body: Dict[str, Any] = Field(..., description="Elasticsearch _search request body (Query DSL JSON)")
    size: Optional[int] = Field(None, description="Optional size override (capped by config.max_hits_cap)")


class ESSearchDSLTool(BaseTool):
    name: str = "es_search_dsl"
    description: str = "Execute an Elasticsearch search with provided Query DSL JSON. Input: {index, body, size}. Output: search response (read-only)."
    args_schema: Type[BaseModel] = ESSearchDSLInput

    def __init__(self, es: Elasticsearch, config: Any):
        super().__init__()
        self.es = es
        self.config = config

    def _run(self, index: str, body: Dict[str, Any], size: Optional[int] = None) -> Dict[str, Any]:
        _assert_index_allowed(index, self.config.index_allowlist)

        if size is None:
            size = body.get("size", 10)
        size = min(int(size), int(getattr(self.config, "max_hits_cap", 200)))
        body = dict(body)
        body["size"] = size

        return self.es.search(index=index, body=body, request_timeout=self.config.request_timeout)


# -----------------------------
# Tool 6: count
# -----------------------------

class ESCountInput(BaseModel):
    index: str = Field(..., description="Index name")
    query: Optional[Dict[str, Any]] = Field(None, description="Optional Query DSL query part (e.g. {'match': ...}).")


class ESCountTool(BaseTool):
    name: str = "es_count"
    description: str = "Count docs matching a query. Input: {index, query}. Output: count response (read-only)."
    args_schema: Type[BaseModel] = ESCountInput

    def __init__(self, es: Elasticsearch, config: Any):
        super().__init__()
        self.es = es
        self.config = config

    def _run(self, index: str, query: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        _assert_index_allowed(index, self.config.index_allowlist)
        body = {"query": query or {"match_all": {}}}
        return self.es.count(index=index, body=body, request_timeout=self.config.request_timeout)


# -----------------------------
# Tool 7: validate query
# -----------------------------

class ESValidateQueryInput(BaseModel):
    index: str = Field(..., description="Index name")
    query: Dict[str, Any] = Field(..., description="Query DSL query part (not full _search body).")


class ESValidateQueryTool(BaseTool):
    name: str = "es_validate_query"
    description: str = "Validate Query DSL via _validate/query. Input: {index, query}. Output: validation response (read-only)."
    args_schema: Type[BaseModel] = ESValidateQueryInput

    def __init__(self, es: Elasticsearch, config: Any):
        super().__init__()
        self.es = es
        self.config = config

    def _run(self, index: str, query: Dict[str, Any]) -> Dict[str, Any]:
        _assert_index_allowed(index, self.config.index_allowlist)
        return self.es.indices.validate_query(
            index=index,
            body={"query": query},
            explain=True,
            request_timeout=self.config.request_timeout,
        )


# -----------------------------
# Tool 8: explain
# -----------------------------

class ESExplainInput(BaseModel):
    index: str = Field(..., description="Index name")
    doc_id: str = Field(..., description="Document _id")
    query: Dict[str, Any] = Field(..., description="Query DSL query part used for explain.")


class ESExplainTool(BaseTool):
    name: str = "es_explain"
    description: str = "Explain why a document matches (or not). Input: {index, doc_id, query}. Output: explain response (read-only)."
    args_schema: Type[BaseModel] = ESExplainInput

    def __init__(self, es: Elasticsearch, config: Any):
        super().__init__()
        self.es = es
        self.config = config

    def _run(self, index: str, doc_id: str, query: Dict[str, Any]) -> Dict[str, Any]:
        _assert_index_allowed(index, self.config.index_allowlist)
        return self.es.explain(
            index=index,
            id=doc_id,
            body={"query": query},
            request_timeout=self.config.request_timeout,
        )


# -----------------------------
# Tool 9: LLM-based DSL checker (analog to QuerySQLCheckerTool)
# -----------------------------

class QueryESDSLCheckerInput(BaseModel):
    index: str = Field(..., description="Index name")
    body: Union[Dict[str, Any], str] = Field(..., description="Full _search body JSON (dict or JSON string).")


class QueryESDSLCheckerTool(BaseTool):
    name: str = "es_dsl_checker"
    description = (
        "Use an LLM to sanity-check and correct an Elasticsearch _search request body (Query DSL). " 
        "Input: {index, body}. Output: corrected JSON body."
    )
    args_schema: Type[BaseModel] = QueryESDSLCheckerInput

    def __init__(self, es: Elasticsearch, llm: Any, config: Any):
        super().__init__()
        self.es = es
        self.llm = llm
        self.config = config

        system = (
            "You are a senior Elasticsearch engineer. Your task is to review and correct an Elasticsearch _search request body (Query DSL).\n"
            "Rules:\n"
            "- Return ONLY a valid JSON object (no markdown, no commentary).\n"
            "- Keep the query READ-ONLY. Do not propose index creation or updates.\n"
            "- If you change field names, ensure they exist in mapping summary.\n"
            "- Prefer safe, common patterns: term queries should target keyword fields; match queries for text fields.\n"
            "- Ensure the output is a FULL _search body (may include query/aggs/sort/knn/size).\n"
        )
        human = (
            "Index mapping summary (field -> type):\n{mapping_summary}\n\n"
            "Original _search body (JSON):\n{body}\n\n"
            "Return corrected _search body as JSON only."
        )
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", human),
            ]
        )
        self._parser = StrOutputParser()

    def _run(self, index: str, body: Union[Dict[str, Any], str]) -> Dict[str, Any]:
        _assert_index_allowed(index, self.config.index_allowlist)

        # 1) mapping summary (bounded)
        mapping = _mapping_summary(self.es, index=index, timeout=self.config.request_timeout)

        # 2) normalize body to JSON string for prompt
        if isinstance(body, str):
            body_obj = _safe_json_loads(body)
        else:
            body_obj = body
        body_str = _as_pretty_json(body_obj)

        # 3) invoke llm
        chain = self._prompt | self.llm | self._parser
        out = chain.invoke({"mapping_summary": _as_pretty_json(mapping), "body": body_str})
        try:
            corrected = _safe_json_loads(out)
        except Exception:
            # 兜底：返回原 body 并附上错误，避免 agent 崩
            return {"error": "es_dsl_checker returned non-JSON", "raw": out, "original": body_obj}
        return corrected


# -----------------------------
# Tool 10: Query vector generator
# -----------------------------

class ESQueryVectorInput(BaseModel):
    text: str = Field(..., description="Text to embed into a vector")
    return_vector: bool = Field(
        False,
        description="Whether to include the full vector in output. Default false to avoid very large tool outputs.",
    )


class ESQueryVectorTool(BaseTool):
    name: str = "es_query_vector"
    description = (
        "Generate an embedding vector for a query text. Default output does NOT include the full vector to avoid huge outputs; " 
        "set return_vector=true only for debugging/out-of-band usage."
    )
    args_schema: Type[BaseModel] = ESQueryVectorInput

    def __init__(self, embeddings: Any = None, embedding_fn: Optional[Callable[[str], List[float]]] = None):
        super().__init__()
        self.embeddings = embeddings
        self.embedding_fn = embedding_fn

    def _embed(self, text: str) -> List[float]:
        if self.embedding_fn is not None:
            vec = self.embedding_fn(text)
        elif self.embeddings is not None and hasattr(self.embeddings, "embed_query"):
            vec = self.embeddings.embed_query(text)
        else:
            raise ValueError("No embeddings or embedding_fn provided.")
        if not isinstance(vec, list):
            vec = list(vec)
        return vec

    def _run(self, text: str, return_vector: bool = False) -> Dict[str, Any]:
        vec = self._embed(text)
        raw = json.dumps(vec, separators=(",", ":")).encode("utf-8")
        sha = hashlib.sha256(raw).hexdigest()
        return {
            "dims": len(vec),
            "sha256": sha,
            **({"vector": vec} if return_vector else {}),
        }


# -----------------------------
# Tool 11: Vector search (recommended for agent usage)
# -----------------------------

class ESVectorSearchInput(BaseModel):
    index: str = Field(..., description="Index name")
    query_text: str = Field(..., description="Natural language query text")
    vector_field: str = Field(..., description="dense_vector field name")
    k: int = Field(10, description="Top k nearest neighbors to retrieve")
    num_candidates: int = Field(100, description="HNSW candidates per shard (>=k)")
    filter: Optional[Dict[str, Any]] = Field(None, description="Optional filter query DSL (placed under knn.filter)")
    source_includes: Optional[List[str]] = Field(None, description="Optional _source includes")


class ESVectorSearchTool(BaseTool):
    name: str = "es_vector_search"
    description = (
        "Vector search helper: embeds query_text then runs Elasticsearch vector search on a dense_vector field. "
        "It will use approximate kNN (search API 'knn' option) when the vector field is indexed; otherwise it "
        "falls back to an exact script_score approach. This avoids returning the full embedding vector to the model."
    )
    args_schema: Type[BaseModel] = ESVectorSearchInput

    def __init__(
        self,
        es: Elasticsearch,
        config: Any,
        embeddings: Any = None,
        embedding_fn: Optional[Callable[[str], List[float]]] = None,
    ):
        super().__init__()
        self.es = es
        self.config = config
        self.embeddings = embeddings
        self.embedding_fn = embedding_fn

    def _embed(self, text: str) -> List[float]:
        if self.embedding_fn is not None:
            vec = self.embedding_fn(text)
        elif self.embeddings is not None and hasattr(self.embeddings, "embed_query"):
            vec = self.embeddings.embed_query(text)
        else:
            raise ValueError("No embeddings or embedding_fn provided for vector search.")
        if not isinstance(vec, list):
            vec = list(vec)
        return vec

    def _get_vector_field_meta(self, index: str, vector_field: str) -> Dict[str, Any]:
        """Fetch vector field mapping meta (best-effort)."""
        resp = self.es.indices.get_mapping(index=index, request_timeout=self.config.request_timeout)
        entry = resp.get(index) or next(iter(resp.values()))
        mappings = (entry or {}).get("mappings") or {}
        props = mappings.get("properties") or {}

        cur = props
        meta: Dict[str, Any] = {}
        for part in vector_field.split("."):
            meta = cur.get(part) or {}
            cur = meta.get("properties") or {}
        return meta or {}

    def _run(
        self,
        index: str,
        query_text: str,
        vector_field: str,
        k: int = 10,
        num_candidates: int = 100,
        filter: Optional[Dict[str, Any]] = None,
        source_includes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        _assert_index_allowed(index, self.config.index_allowlist)

        k = int(k)
        num_candidates = max(int(num_candidates), k)
        size = min(k, int(getattr(self.config, "max_hits_cap", 200)))

        qv = self._embed(query_text)

        meta = self._get_vector_field_meta(index=index, vector_field=vector_field)
        can_knn = bool(meta.get("index") is True)

        # 1) Prefer approximate kNN via search API 'knn' option when indexed
        if can_knn:
            body: Dict[str, Any] = {
                "knn": {
                    "field": vector_field,
                    "query_vector": qv,
                    "k": k,
                    "num_candidates": num_candidates,
                },
                "size": size,
            }
            if filter:
                body["knn"]["filter"] = filter
            if source_includes is not None:
                body["_source"] = {"includes": source_includes}
            return self.es.search(index=index, body=body, request_timeout=self.config.request_timeout)

        # 2) Fallback: exact vector scoring via script_score (does not require index:true)
        base_query = filter or {"match_all": {}}
        body = {
            "size": size,
            "query": {
                "script_score": {
                    "query": base_query,
                    "script": {
                        "source": f"cosineSimilarity(params.query_vector, doc['{vector_field}']) + 1.0",
                        "params": {"query_vector": qv},
                    },
                }
            },
        }
        if source_includes is not None:
            body["_source"] = {"includes": source_includes}
        return self.es.search(index=index, body=body, request_timeout=self.config.request_timeout)
