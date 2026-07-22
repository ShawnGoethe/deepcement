"""
数据摄入模块
负责解析 PDF/Excel/CSV 固井资料，输出标准化 Document 对象
"""

from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field

from loguru import logger


@dataclass
class CementDocument:
    """固井文档标准化结构"""
    content: str                          # 文本内容
    source: str                           # 来源文件名
    doc_type: str = "unknown"             # 文档类型：report / data / log
    metadata: dict = field(default_factory=dict)  # 元数据（井名、日期、井段等）


class DocumentIngester:
    """固井文档摄入器

    支持格式：
    - PDF（固井施工报告、完井报告）—— 重点支持
    - Excel（固井数据表、施工参数记录）
    - CSV（结构化数据）
    """

    def __init__(self, raw_dir: Path):
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def ingest_all(self) -> List[CementDocument]:
        """扫描 raw_dir 下所有支持的文件并解析"""
        documents = []
        for file_path in self.raw_dir.rglob("*"):
            if file_path.is_dir():
                continue
            suffix = file_path.suffix.lower()
            if suffix == ".pdf":
                docs = self._parse_pdf(file_path)
            elif suffix in (".xlsx", ".xls"):
                docs = self._parse_excel(file_path)
            elif suffix == ".csv":
                docs = self._parse_csv(file_path)
            else:
                logger.debug(f"跳过不支持的文件格式: {file_path}")
                continue
            documents.extend(docs)
            logger.info(f"解析完成: {file_path.name} → {len(docs)} 个文档片段")

        logger.info(f"共解析 {len(documents)} 个文档片段")
        return documents

    def ingest_file(self, file_path: Path) -> List[CementDocument]:
        """解析单个文件"""
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            return self._parse_pdf(file_path)
        elif suffix in (".xlsx", ".xls"):
            return self._parse_excel(file_path)
        elif suffix == ".csv":
            return self._parse_csv(file_path)
        else:
            logger.warning(f"不支持的文件格式: {suffix}")
            return []

    def _parse_pdf(self, file_path: Path) -> List[CementDocument]:
        """解析 PDF 文件（固井施工报告）

        使用 PyMuPDF 提取文本，按页拆分并提取元数据
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.error("请安装 PyMuPDF: pip install pymupdf")
            return []

        documents = []
        try:
            doc = fitz.open(str(file_path))
            full_text = ""
            for page in doc:
                full_text += page.get_text()

            # 提取元数据（从文本中匹配常见固井字段）
            metadata = self._extract_metadata(full_text, file_path)
            metadata["file_path"] = str(file_path)
            metadata["total_pages"] = len(doc)

            # 按段落拆分（空行分隔）
            paragraphs = [p.strip() for p in full_text.split("\n\n") if p.strip()]

            if paragraphs:
                # 整体作为一个文档（保留上下文完整性）
                documents.append(CementDocument(
                    content=full_text,
                    source=file_path.name,
                    doc_type="report",
                    metadata=metadata,
                ))

            doc.close()
        except Exception as e:
            logger.error(f"PDF 解析失败 {file_path}: {e}")

        return documents

    def _parse_excel(self, file_path: Path) -> List[CementDocument]:
        """解析 Excel 文件（固井数据表）"""
        try:
            import pandas as pd
        except ImportError:
            logger.error("请安装 pandas + openpyxl")
            return []

        documents = []
        try:
            xls = pd.ExcelFile(str(file_path))
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet_name)
                if df.empty:
                    continue

                # 转为文本描述
                content = f"【工作表: {sheet_name}】\n"
                content += df.to_string(index=False)

                metadata = self._extract_metadata(content, file_path)
                metadata["sheet_name"] = sheet_name
                metadata["row_count"] = len(df)
                metadata["columns"] = list(df.columns)

                documents.append(CementDocument(
                    content=content,
                    source=file_path.name,
                    doc_type="data",
                    metadata=metadata,
                ))
        except Exception as e:
            logger.error(f"Excel 解析失败 {file_path}: {e}")

        return documents

    def _parse_csv(self, file_path: Path) -> List[CementDocument]:
        """解析 CSV 文件"""
        try:
            import pandas as pd
        except ImportError:
            logger.error("请安装 pandas")
            return []

        documents = []
        try:
            df = pd.read_csv(str(file_path))
            content = df.to_string(index=False)

            metadata = self._extract_metadata(content, file_path)
            metadata["row_count"] = len(df)
            metadata["columns"] = list(df.columns)

            documents.append(CementDocument(
                content=content,
                source=file_path.name,
                doc_type="data",
                metadata=metadata,
            ))
        except Exception as e:
            logger.error(f"CSV 解析失败 {file_path}: {e}")

        return documents

    def _extract_metadata(self, text: str, file_path: Path) -> dict:
        """从文本中提取固井相关元数据（井名、日期、井段等）

        使用简单的正则匹配，可根据实际数据格式扩展
        """
        import re

        metadata = {"filename": file_path.name}

        # 匹配井名（常见格式：XX-1、XX1-H1、XX-1-2）
        well_pattern = re.search(r'井\s*名[：:]\s*(\S+)', text)
        if well_pattern:
            metadata["well_name"] = well_pattern.group(1)
        else:
            # 尝试从文件名提取
            metadata["well_name"] = file_path.stem

        # 匹配日期
        date_pattern = re.search(r'(\d{4})[年/-](\d{1,2})[月/-](\d{1,2})', text)
        if date_pattern:
            metadata["date"] = f"{date_pattern.group(1)}-{date_pattern.group(2).zfill(2)}-{date_pattern.group(3).zfill(2)}"

        # 匹配井深
        depth_pattern = re.search(r'(?:井深|完钻井深)[：:]\s*([\d.]+)\s*[mM米]', text)
        if depth_pattern:
            metadata["well_depth"] = float(depth_pattern.group(1))

        # 匹配井段
        section_pattern = re.search(r'(?:固井井段|封固井段)[：:]\s*([\d.]+)\s*[-~—]\s*([\d.]+)', text)
        if section_pattern:
            metadata["cement_section"] = f"{section_pattern.group(1)}-{section_pattern.group(2)}m"

        return metadata
