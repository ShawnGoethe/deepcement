"""
Agent 编排模块
基于 DeepAgent 编排固井质量评测流程
"""

from pathlib import Path

from loguru import logger

from agent.tools import create_cement_tools
from config import BASE_DIR, settings
from core import DocumentIngester, HistoryRetriever, IndexManager, QualityEvaluator
from core.cascade_evaluator import CascadeEvaluator

# Workspace 路径（AGENTS.md + skills）
WORKSPACE_DIR = Path(__file__).parent / "workspace"


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

        # 会话状态
        self._session_initialized = False

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

        Workspace 结构：
            agent/workspace/
            ├── memory/
            │   ├── long-term.md       # 长期记忆（跨会话持久知识）
            │   └── short-term.md      # 短期记忆（当前会话上下文）
            ├── prompts/
            │   └── system-prompt.md   # 系统提示词
            └── skills/                # 技能目录
                ├── build-index/
                ├── evaluate-well/
                ├── analyze-las/
                ├── train-model/
                └── code-quality/

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

        # 从 workspace 加载系统提示词
        system_prompt = self._load_prompt("system-prompt.md")

        # FilesystemBackend 以 workspace 为根目录
        # 虚拟路径 /skills/ → agent/workspace/skills/
        # 虚拟路径 /AGENTS.md → agent/workspace/AGENTS.md
        backend = FilesystemBackend(root_dir=str(WORKSPACE_DIR), virtual_mode=True)

        agent = create_deep_agent(
            name="固井质量评测助手",
            model=llm,
            tools=self.tools,
            system_prompt=system_prompt,
            skills=["./skills/"],
            memory=[
                "./memory/long-term.md",  # 长期记忆：跨会话持久知识
                "./memory/short-term.md",  # 短期记忆：当前会话上下文
            ],
            interrupt_on={
                "remove_file": {
                    "allowed_decisions": ["approve", "reject"],
                    "message": "⚠️ 文件删除操作需要确认：此操作不可逆，请审核待删除文件是否正确。",
                },
                "updateXGBoost": {
                    "allowed_decisions": ["approve", "reject"],
                    "message": "⚠️ XGBoost 模型重新训练需要确认：训练将覆盖现有模型，可能需要较长时间。",
                },
            },
            backend=backend,
        )
        logger.info(f"[_create_agent] DeepAgent 创建成功，workspace: {WORKSPACE_DIR}")
        return agent

    @staticmethod
    def _load_prompt(name: str) -> str:
        """从 workspace/prompts/ 加载提示词文件

        Args:
            name: 文件名（如 "system-prompt.md"）

        Returns:
            文件内容文本

        Raises:
            FileNotFoundError: 文件不存在
        """
        path = WORKSPACE_DIR / "prompts" / name
        if not path.exists():
            raise FileNotFoundError(f"提示词文件不存在: {path}")
        logger.info(f"[_load_prompt] 加载提示词: {path}")
        return path.read_text(encoding="utf-8")

    @staticmethod
    def reset_short_term_memory():
        """重置短期记忆（每个会话开始时调用）

        清空当前会话的临时上下文，保留长期记忆不变。
        """
        import datetime

        short_term_path = WORKSPACE_DIR / "memory" / "short-term.md"
        if not short_term_path.exists():
            return

        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        content = f"""# Short-Term Memory

<!-- 本文件是 Agent 的短期记忆，记录当前会话的上下文。
     每次会话开始时应重置本文件。
     Agent 可通过 edit_file 更新本文件，跟踪当前任务状态。 -->

## Current Session

- **Session Start**: {now}
- **Status**: idle

## Active Context

<!-- 当前正在处理的井/文件/任务 -->

- **Active Well**: 无
- **Active File**: 无
- **Active Task**: 无

## Recent Queries

<!-- 本次会话中用户最近的查询，最多记录 10 条 -->

暂无查询记录。

## Temporary Findings

<!-- 本次会话中的临时发现和中间结果 -->

暂无临时发现。

## Conversation Summary

<!-- Agent 在对话变长时更新此节，压缩关键信息 -->

暂无对话摘要。
"""
        short_term_path.write_text(content, encoding="utf-8")
        logger.info(f"[reset_short_term_memory] 短期记忆已重置: {now}")

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

        # 会话首次调用：重置短期记忆
        if not self._session_initialized:
            self.reset_short_term_memory()
            self._session_initialized = True

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
