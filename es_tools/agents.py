"""ES 只读智能体（LangChain tool-calling）
提供基于 Elasticsearch 的只读查询 Agent 构建函数。核心能力：
- 自动注入一组 ES 工具（索引枚举、字段能力、示例文档、查询/统计、校验）
- 使用系统/人类消息模板构造 prompt（支持 top_k 限制）
- 以 tool-calling 方式运行，返回 `AgentExecutor` 供外部执行

适用场景：当需要在会话中动态探索索引结构、生成并执行 Query DSL 时使用。
注意事项：
- `prompt` 与 `prefix/suffix/format_instructions/input_variables` 互斥，只能二选一
- 未显式传入 `toolkit` 时，必须提供 `es_client`
- `index_allowlist` 用于限制可访问索引集合，提升安全性与可控性
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor
from langchain.agents.tool_calling_agent import create_tool_calling_agent

from .toolkit import ElasticsearchDatabaseToolkit, ElasticsearchToolkitConfig
from .prompts import ES_PREFIX, ES_SUFFIX_TOOLCALLING

def create_es_agent(
    llm,
    toolkit=None,
    agent_type=None,
    callback_manager=None,
    prefix=None,
    suffix=None,
    format_instructions=None,
    input_variables=None,
    top_k=10,
    max_iterations=15,
    max_execution_time=None,
    early_stopping_method="force",
    verbose=False,
    agent_executor_kwargs=None,
    extra_tools=(),
    *,
    es_client=None,
    index_allowlist=None,
    prompt=None,
    **kwargs,
):
    """构建并返回一个基于 Elasticsearch 的只读 AgentExecutor（tool-calling）

    参数：
    - llm：LLM 实例，用于智能工具调用与（可选）DSL 检查
    - toolkit：ES 工具集合（可选）；若未提供需使用 `es_client` 自动构建
    - agent_type：Agent 策略（当前仅支持 `"tool-calling"`）
    - callback_manager：已废弃的回调管理器（不建议使用）
    - prefix/suffix/format_instructions/input_variables：自定义 prompt 片段（与 `prompt` 互斥）
    - top_k：默认结果数量上限，用于模板中提示与工具 size 限制
    - max_iterations：Agent 迭代上限
    - max_execution_time：Agent 最大执行时长（秒）
    - early_stopping_method：提前停止策略，默认 `"force"`
    - verbose：是否打印详细日志
    - agent_executor_kwargs：透传给 `AgentExecutor` 的参数（如 `callbacks`）
    - extra_tools：额外注入的工具列表
    - es_client：Elasticsearch 客户端（在未传入 `toolkit` 时必需）
    - index_allowlist：允许访问的索引名单（序列）
    - prompt：完整的 `ChatPromptTemplate`，与 prefix/suffix 等互斥
    - kwargs：预留扩展参数

    返回：
    - `AgentExecutor`：可执行的 LangChain Agent 执行器
    """
    if toolkit is None:
        if es_client is None:
            raise ValueError("Must provide either toolkit or es_client.")
        toolkit = ElasticsearchDatabaseToolkit(
            es_client=es_client,
            llm=llm,  # 可：用于 QueryESDSLCheckerTool
            config=ElasticsearchToolkitConfig(index_allowlist=index_allowlist),
        )

    tools = toolkit.get_tools() + list(extra_tools)

    if prompt is not None and any([prefix, suffix, format_instructions, input_variables]):
        raise ValueError("prompt is mutually exclusive with prefix/suffix/format_instructions/input_variables.")

    # 默认 prompt（tool-calling）
    if prompt is None:
        sys = (prefix or ES_PREFIX).format(top_k=top_k)
        hum = suffix or ES_SUFFIX_TOOLCALLING
        prompt = ChatPromptTemplate.from_messages([("system", sys), ("human", hum)])

    # agent_type 策略：建议默认 tool-calling
    at = agent_type or "tool-calling"
    if at != "tool-calling":
        # 若你们需要兼容旧类型，可在这里分支到 create_openai_tools_agent 等
        raise NotImplementedError("Only tool-calling is implemented in this design.")

    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)

    executor_kwargs = agent_executor_kwargs or {}
    # callback_manager 已被官方标记为 deprecated，建议走 executor_kwargs["callbacks"]。:contentReference[oaicite:11]{index=11}
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        max_iterations=max_iterations,
        max_execution_time=max_execution_time,
        early_stopping_method=early_stopping_method,
        **executor_kwargs,
    )
