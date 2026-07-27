"""
Agent 编排模块
基于 DeepAgent 编排固井质量评测流程
"""

from loguru import logger

from agent.tools import create_cement_tools
from config import BASE_DIR, settings
from core import DocumentIngester, HistoryRetriever, IndexManager, QualityEvaluator
from core.cascade_evaluator import CascadeEvaluator


class CementAgent:
    """固井质量评测智能体

    编排流程：
    1. 用户输入查询（井名/井段/时间段）
    2. 检索历史资料（LlamaIndex）
    3. 提取关键质量指标
    4. 对比标准/邻井数据
    5. 生成质量评测报告（结论 + 建议）
    """

    def __init__(self):
        # 初始化核心组件
        self.ingester = DocumentIngester(settings.raw_dir)
        self.indexer = IndexManager()
        self.retriever = HistoryRetriever(self.indexer)
        self.evaluator = QualityEvaluator(self.retriever)

        # 初始化分层评估器
        self.cascade_evaluator = CascadeEvaluator(
            llm_evaluator=self.evaluator,
            retriever=self.retriever,
        )

        # 创建工具集（@tool 装饰的函数列表）
        self.tools = create_cement_tools(
            self.retriever,
            self.evaluator,
            self.cascade_evaluator,
        )

        # 初始化 DeepAgent
        self._agent = self._create_agent()

    def build_index(self):
        """构建索引（从 raw 目录读取并索引）"""
        logger.info("开始构建索引...")
        documents = self.ingester.ingest_all()
        if not documents:
            logger.warning("没有找到可索引的文档")
            return False

        self.indexer.build_index(documents)
        logger.info("索引构建完成（已写入 Zilliz Cloud）")
        return True

    def load_index(self) -> bool:
        """连接 Zilliz Cloud 加载已有索引"""
        return self.indexer.connect()

    def _create_agent(self):
        """创建 DeepAgent 实例

        Raises:
            ImportError: deepagents 未安装
            Exception: 创建失败
        """
        from deepagents import create_deep_agent
        from deepagents.backends import FilesystemBackend
        from langchain_openai import ChatOpenAI

        logger.info("[_create_agent] 导入模块成功")

        # 创建 LangChain 模型实例
        llm = ChatOpenAI(
            model=settings.llm.model,
            openai_api_base=settings.llm.base_url,
            openai_api_key=settings.llm.api_key,
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens,
        )
        logger.info(f"[_create_agent] LLM 创建成功: {settings.llm.model}")

        logger.info(
            f"[_create_agent] 注册 {len(self.tools)} 个工具: {[t.name for t in self.tools]}"
        )

        agent = create_deep_agent(
            name="固井质量评测助手",
            model=llm,
            tools=self.tools,
            system_prompt=self._get_system_prompt(),
            backend=FilesystemBackend(root_dir=str(BASE_DIR)),
        )
        logger.info("[_create_agent] DeepAgent 创建成功")
        return agent

    def _get_system_prompt(self) -> str:
        """获取 Agent 系统提示词"""
        return """你是一位固井质量评测专家助手。你的职责是：

1. **检索历史资料**：根据用户查询，从固井历史资料库中检索相关信息
2. **质量评测**：对指定井的固井质量进行全面评测，包括：
   - 水泥浆性能（密度、失水量、稠化时间）
   - 施工参数（泵速、压力、替量）
   - 固井效果（返高、候凝、胶结质量）
   - 异常情况（漏失、窜槽、气侵）
3. **测井数据分析**：使用分层评估系统分析 LAS 测井数据
   - 规则引擎 → XGBoost → LLM 兜底
   - 基于 AC/CAL/GR/RT 等曲线评估水泥胶结质量
4. **生成报告**：输出结构化的评测报告，包含：
   - 综合得分和等级
   - 各维度详细评分
   - 质量结论
   - 改进建议

工作流程：
- 当用户询问某口井的质量情况时，先用 search_history 检索资料
- 当用户询问某口井的胶结情况时,调用evaluate_well_log对./data/raw/威202H16-6_COM_20180525(360-2546).TXT的数据进行分层评估,返回胶结评估和xgboost计算时间
- 当用户提供 LAS 文件或测井数据时，使用 evaluate_well_log 进行分层评估
- 当用户需要对比数据时，使用 compare_data 工具
- 回答要专业、准确、有依据，引用具体数据

请用中文回答。
"""

    def run(self, query: str) -> str:
        """运行 Agent 处理用户查询

        Args:
            query: 用户查询文本

        Returns:
            Agent 回复文本

        Raises:
            RuntimeError: DeepAgent 未初始化或执行失败
        """
        logger.info(f"[CementAgent.run] 收到查询: {query}")

        # 确保索引已加载
        if not self.indexer.is_ready:
            logger.info("[CementAgent.run] 索引未就绪，尝试加载...")
            if not self.load_index():
                return "错误：索引未就绪。请先运行 build_index() 构建索引。"
            logger.info("[CementAgent.run] 索引加载成功")

        if self._agent is None:
            raise RuntimeError(
                "DeepAgent 未初始化，请检查 deepagents 是否安装以及 LLM 配置是否正确"
            )

        logger.info("[CementAgent.run] 使用 DeepAgent 处理查询...")
        result = self._agent.invoke({"messages": [{"role": "user", "content": query}]})
        # 详细日志：打印每条消息的类型和内容摘要
        if isinstance(result, dict) and "messages" in result:
            for i, msg in enumerate(result["messages"]):
                logger.info(
                    f"[CementAgent.run] 消息[{i}] type={type(msg).__name__} content={str(msg.content)[:150]}"
                )
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        logger.info(
                            f"[CementAgent.run]   工具调用: {tc['name']}({tc.get('args', {})})"
                        )
            return result["messages"][-1].content
        logger.warning(f"[CementAgent.run] 非预期的返回格式: {type(result)}")
        return str(result)

    def interactive(self):
        """交互模式"""
        print("=" * 50)
        print("  DeepCement 固井质量评测系统")
        print("  输入井名和查询，系统将自动检索和评测")
        print("=" * 50)

        # 确保索引已连接
        if not self.indexer.is_ready:
            print("\n正在连接 Zilliz Cloud...")
            if not self.load_index():
                print("连接失败")

        print("\n就绪！请输入查询：\n")

        while True:
            try:
                query = input(">>> ").strip()
                if not query:
                    continue
                if query.lower() in ("quit", "exit", "q"):
                    print("再见！")
                    break

                result = self.run(query)
                print(f"\n{result}\n")

            except KeyboardInterrupt:
                print("\n再见！")
                break
            except Exception as e:
                logger.error(f"处理异常: {e}")
                print(f"\n处理出错: {e}\n")
