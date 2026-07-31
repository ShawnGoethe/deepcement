"""
固井数据结构化抽取模块
使用 LLM 从文档文本中抽取核心固井关系三元组
"""

import json
import re
from typing import List, Optional

from loguru import logger
from pydantic import BaseModel, Field


class ExtractedTriple(BaseModel):
    """抽取的关系三元组"""

    subject: str = Field(description="主语（井名/地层名）")
    predicate: str = Field(description="谓语（关系类型）")
    object: str = Field(description="宾语（值/实体）")
    properties: dict = Field(default_factory=dict, description="附加属性")
    source: str = Field(default="", description="来源文件")


# 抽取 Prompt
EXTRACT_PROMPT = """你是固井工程数据抽取专家。请从以下固井文档中抽取结构化关系三元组。

抽取类型（只抽取文档中明确提到的数据，不要推测）：

1. 地层信息：井名 + 有地层 + 地层类型（生物灰岩、纯砂岩、盐膏层、高压盐水层等），附带顶深、底深、厚度
2. 水泥浆密度：井名/地层 + 水泥浆密度 + 密度值（g/cm³）
3. 纯砂岩厚度：井名 + 纯砂岩厚度 + 厚度值（m）
4. 固井质量：井名 + 固井质量 + 评价等级（优秀/良好/合格/不合格）
5. 异常情况：井名 + 遇到异常 + 异常类型（井漏/井涌/憋泵/卡钻等）
6. 水泥浆配方：井名 + 水泥浆配方 + 配方描述（水泥类型、添加剂等）
7. 井身结构：井名 + 井身结构 + 套管程序（表层/技术/生产套管等）
8. 施工参数：井名 + 施工参数 + 参数描述（泵压、排量、替浆量等）

请严格按以下 JSON 数组格式输出，不要输出其他内容：
```json
[
  {{
    "subject": "XX-1井",
    "predicate": "有地层",
    "object": "生物灰岩段",
    "properties": {{"顶深": "2500m", "底深": "2550m", "厚度": "50m"}}
  }},
  {{
    "subject": "XX-1井",
    "predicate": "水泥浆密度",
    "object": "1.85g/cm³",
    "properties": {{"用途": "高压盐水层封固"}}
  }}
]
```

文档内容：
{text}

井名/文件名提示：{source}
"""


class CementDataExtractor:
    """固井数据抽取器

    使用 LLM 从 CementDocument 中抽取结构化三元组
    """

    def __init__(self):
        from config import settings

        self._settings = settings

    def _get_llm(self):
        """延迟初始化 LLM"""
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=self._settings.llm.model,
            openai_api_base=self._settings.llm.base_url,
            openai_api_key=self._settings.llm.api_key,
            temperature=0.1,  # 低温度保证抽取稳定性
            max_tokens=4096,
        )

    def extract_from_text(self, text: str, source: str = "") -> List[ExtractedTriple]:
        """从文本中抽取三元组

        Args:
            text: 文档文本内容
            source: 来源文件名

        Returns:
            抽取出的三元组列表
        """
        if not text.strip():
            return []

        llm = self._get_llm()

        # 截断过长文本（避免超出 token 限制）
        max_chars = 12000
        if len(text) > max_chars:
            text = text[:max_chars] + "\n...(文本已截断)"

        prompt = EXTRACT_PROMPT.format(text=text, source=source)

        try:
            response = llm.invoke(prompt)
            return self._parse_response(response.content, source)
        except Exception as e:
            logger.warning(f"LLM 抽取失败 [{source}]: {e}")
            return []

    def extract_all(self, documents: list) -> List[ExtractedTriple]:
        """从所有文档中抽取三元组

        Args:
            documents: CementDocument 列表

        Returns:
            所有三元组的汇总列表
        """
        all_triples = []
        total = len(documents)

        for i, doc in enumerate(documents):
            logger.info(f"抽取进度: {i + 1}/{total} — {doc.source}")
            triples = self.extract_from_text(doc.content, source=doc.source)
            all_triples.extend(triples)
            logger.info(f"  → 抽取到 {len(triples)} 个三元组")

        logger.info(f"抽取完成: 共 {len(all_triples)} 个三元组")
        return all_triples

    @staticmethod
    def _parse_response(response: str, source: str = "") -> List[ExtractedTriple]:
        """解析 LLM 返回的 JSON 三元组

        支持从 markdown 代码块或纯 JSON 中提取
        """
        # 尝试从 markdown 代码块中提取 JSON
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            # 尝试直接解析整个响应
            json_str = response.strip()

        # 尝试找到 JSON 数组
        array_match = re.search(r"\[.*\]", json_str, re.DOTALL)
        if array_match:
            json_str = array_match.group(0)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON 解析失败: {e}\n原始响应: {response[:200]}")
            return []

        if not isinstance(data, list):
            logger.warning(f"期望 JSON 数组，实际得到: {type(data)}")
            return []

        triples = []
        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                triple = ExtractedTriple(
                    subject=item.get("subject", ""),
                    predicate=item.get("predicate", ""),
                    object=item.get("object", ""),
                    properties=item.get("properties", {}),
                    source=source,
                )
                if triple.subject and triple.predicate and triple.object:
                    triples.append(triple)
            except Exception as e:
                logger.debug(f"跳过无效三元组: {item} ({e})")

        return triples
