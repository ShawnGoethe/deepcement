"""
Agent 编排模块
基于 DeepAgent 编排固井质量评测流程
"""

from typing import Optional

from loguru import logger

from config import settings
from core import DocumentIngester, IndexManager, HistoryRetriever, QualityEvaluator
from agent.tools import create_cement_tools


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

        # 创建工具集（@tool 装饰的函数列表）
        self.tools = create_cement_tools(self.retriever, self.evaluator)
        # 为降级模式创建工具名到函数的映射
        self._tool_map = {t.name: t for t in self.tools}

        # DeepAgent 实例
        self._agent = None

    def build_index(self):
        """构建索引（从 raw 目录读取并索引）"""
        logger.info("开始构建索引...")
        documents = self.ingester.ingest_all()
        if not documents:
            logger.warning("没有找到可索引的文档")
            return False

        self.indexer.build_index(documents)
        self.indexer.save_index()
        logger.info("索引构建并保存完成")
        return True

    def load_index(self) -> bool:
        """加载已有索引"""
        return self.indexer.load_index()

    def _init_agent(self):
        """初始化 DeepAgent（延迟加载）"""
        if self._agent is not None:
            return

        try:
            from deepagents import create_deep_agent
            from langchain_openai import ChatOpenAI

            # 创建 LangChain 模型实例
            llm = ChatOpenAI(
                model=settings.llm.model,
                openai_api_base=settings.llm.base_url,
                openai_api_key=settings.llm.api_key,
                temperature=settings.llm.temperature,
                max_tokens=settings.llm.max_tokens,
            )

            self._agent = create_deep_agent(
                name="固井质量评测助手",
                model=llm,
                tools=self.tools,  # 已经是 @tool 装饰的函数列表
                system_prompt=self._get_system_prompt(),
            )
            logger.info("DeepAgent 初始化完成")
        except ImportError:
            logger.warning("DeepAgent 未安装，将使用直接调用模式")
            self._agent = None

    def _get_system_prompt(self) -> str:
        """获取 Agent 系统提示词"""
        return """你是一位固井质量评测专家助手。你的职责是：

1. **检索历史资料**：根据用户查询，从固井历史资料库中检索相关信息
2. **质量评测**：对指定井的固井质量进行全面评测，包括：
   - 水泥浆性能（密度、失水量、稠化时间）
   - 施工参数（泵速、压力、替量）
   - 固井效果（返高、候凝、胶结质量）
   - 异常情况（漏失、窜槽、气侵）
3. **生成报告**：输出结构化的评测报告，包含：
   - 综合得分和等级
   - 各维度详细评分
   - 质量结论
   - 改进建议

工作流程：
- 当用户询问某口井的质量情况时，先用 search_history 检索资料，再用 evaluate_quality 生成评测
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
        """
        # 确保索引已加载
        if not self.indexer.is_ready:
            if not self.load_index():
                return "错误：索引未就绪。请先运行 build_index() 构建索引。"

        self._init_agent()

        if self._agent is not None:
            # 使用 DeepAgent
            try:
                result = self._agent.invoke({"messages": [{"role": "user", "content": query}]})
                # 提取最后一条消息作为回复
                if isinstance(result, dict) and "messages" in result:
                    return result["messages"][-1].content
                return str(result)
            except Exception as e:
                logger.error(f"Agent 执行失败: {e}")
                return self._fallback_run(query)
        else:
            # 降级：直接调用工具
            return self._fallback_run(query)

    def _fallback_run(self, query: str) -> str:
        """降级模式：不依赖 DeepAgent，直接调用工具链

        简单规则：
        - 查询中包含"评测"/"质量"/"报告" → 调用 evaluate_quality
        - 查询中包含"对比"/"比较" → 调用 compare_data
        - 其他 → 调用 search_history
        """
        logger.info("使用降级模式处理查询")

        # 尝试从查询中提取井名
        well_name = self._extract_well_name(query)

        if any(kw in query for kw in ["评测", "质量", "报告", "评估"]):
            if well_name:
                return self._tool_map["evaluate_quality"].invoke({"well_name": well_name, "query": query})
            else:
                return "请提供井名以进行质量评测。例如：评测 XX-1 井的固井质量"
        elif any(kw in query for kw in ["对比", "比较", "vs"]):
            # 简单处理：需要两口井名
            return "请提供两口井名以进行对比。例如：对比 XX-1 和 XX-2 的固井数据"
        else:
            return self._tool_map["search_history"].invoke({"query": query, "well_name": well_name})

    def _extract_well_name(self, query: str) -> Optional[str]:
        """从查询中提取井名（简单规则匹配）"""
        import re

        # 常见井名格式：XX-1、XX1-H1、XX-1-2
        patterns = [
            r'([A-Za-z]+-\d+(?:-\d+)?(?:-H\d+)?)',  # XX-1, XX-1-2, XX-1-H1
            r'(\d+-\d+(?:-\d+)?)',                      # 1-1, 1-1-2
            r'([一-龥]+-\d+)',                   # 中文-1
        ]

        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                return match.group(1)

        return None

    def interactive(self):
        """交互模式"""
        print("=" * 50)
        print("  DeepCement 固井质量评测系统")
        print("  输入井名和查询，系统将自动检索和评测")
        print("  输入 'quit' 退出")
        print("=" * 50)

        # 确保索引已加载
        if not self.indexer.is_ready:
            print("\n正在加载索引...")
            if not self.load_index():
                print("索引未找到，是否构建？(y/n)")
                if input().strip().lower() == 'y':
                    self.build_index()
                else:
                    print("无法继续，退出。")
                    return

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
