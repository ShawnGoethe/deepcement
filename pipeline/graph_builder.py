"""
固井知识图谱 + 结构化数据存储模块
使用 SQLite 存储结构化数据，LlamaIndex PropertyGraphIndex 存储关系三元组
"""

import json
import sqlite3
from pathlib import Path
from typing import List

from loguru import logger

from pipeline.extractor import ExtractedTriple


class GraphBuilder:
    """固井知识图谱构建器

    双写模式：
    - SQLite: 结构化数据（井信息、地层、水泥数据），支持 SQL 查询
    - PropertyGraphIndex: 关系三元组（实体-关系-实体），持久化到本地 JSON
    """

    def __init__(self, sqlite_path: str, graph_dir: str):
        self.sqlite_path = Path(sqlite_path)
        self.graph_dir = Path(graph_dir)

        # 确保目录存在
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.graph_dir.mkdir(parents=True, exist_ok=True)

        # 初始化 SQLite
        self._init_sqlite()

    def _init_sqlite(self):
        """初始化 SQLite 数据库表结构"""
        conn = sqlite3.connect(str(self.sqlite_path))
        cursor = conn.cursor()

        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS wells (
                well_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL UNIQUE,
                depth       REAL,
                date        TEXT,
                cement_section TEXT,
                quality_grade   TEXT,
                source_file TEXT
            );

            CREATE TABLE IF NOT EXISTS geological_layers (
                layer_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                well_id     INTEGER NOT NULL,
                layer_type  TEXT NOT NULL,
                top_depth   REAL,
                bottom_depth REAL,
                thickness   REAL,
                properties  TEXT DEFAULT '{}',
                FOREIGN KEY (well_id) REFERENCES wells(well_id)
            );

            CREATE TABLE IF NOT EXISTS cement_data (
                data_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                well_id     INTEGER NOT NULL,
                data_type   TEXT NOT NULL,
                value       TEXT NOT NULL,
                unit        TEXT,
                properties  TEXT DEFAULT '{}',
                source_file TEXT,
                FOREIGN KEY (well_id) REFERENCES wells(well_id)
            );

            CREATE INDEX IF NOT EXISTS idx_wells_name ON wells(name);
            CREATE INDEX IF NOT EXISTS idx_layers_well ON geological_layers(well_id);
            CREATE INDEX IF NOT EXISTS idx_layers_type ON geological_layers(layer_type);
            CREATE INDEX IF NOT EXISTS idx_cement_well ON cement_data(well_id);
            CREATE INDEX IF NOT EXISTS idx_cement_type ON cement_data(data_type);
        """)

        conn.commit()
        conn.close()
        logger.debug(f"SQLite 初始化完成: {self.sqlite_path}")

    def build(self, triples: List[ExtractedTriple]):
        """从抽取的三元组构建 SQLite + 知识图谱

        Args:
            triples: LLM 抽取的关系三元组列表
        """
        if not triples:
            logger.warning("没有三元组可构建")
            return

        # 1. 写入 SQLite
        self._build_sqlite(triples)

        # 2. 构建 PropertyGraphIndex
        self._build_graph(triples)

    def _get_or_create_well(self, cursor: sqlite3.Cursor, well_name: str,
                            source: str = "") -> int:
        """获取或创建井记录，返回 well_id"""
        cursor.execute("SELECT well_id FROM wells WHERE name = ?", (well_name,))
        row = cursor.fetchone()
        if row:
            return row[0]

        cursor.execute(
            "INSERT INTO wells (name, source_file) VALUES (?, ?)",
            (well_name, source)
        )
        return cursor.lastrowid

    def _build_sqlite(self, triples: List[ExtractedTriple]):
        """将三元组写入 SQLite 结构化表"""
        conn = sqlite3.connect(str(self.sqlite_path))
        cursor = conn.cursor()

        well_cache = {}  # well_name → well_id

        for t in triples:
            subject = t.subject.strip()
            predicate = t.predicate.strip()
            obj = t.object.strip()
            props = t.properties or {}

            # 获取井 ID（主语通常是井名）
            if subject not in well_cache:
                well_id = self._get_or_create_well(cursor, subject, t.source)
                well_cache[subject] = well_id
            well_id = well_cache[subject]

            # 按谓语类型分发存储
            if predicate == "有地层":
                self._insert_layer(cursor, well_id, obj, props)

            elif predicate == "水泥浆密度":
                self._insert_cement_data(cursor, well_id, "水泥浆密度",
                                         obj, "g/cm³", props, t.source)

            elif predicate == "纯砂岩厚度":
                self._insert_cement_data(cursor, well_id, "纯砂岩厚度",
                                         obj, "m", props, t.source)

            elif predicate == "固井质量":
                # 更新井的质量等级
                cursor.execute(
                    "UPDATE wells SET quality_grade = ? WHERE well_id = ?",
                    (obj, well_id)
                )

            elif predicate == "遇到异常":
                self._insert_cement_data(cursor, well_id, "异常情况",
                                         obj, None, props, t.source)

            elif predicate == "水泥浆配方":
                self._insert_cement_data(cursor, well_id, "水泥浆配方",
                                         obj, None, props, t.source)

            elif predicate == "井身结构":
                self._insert_cement_data(cursor, well_id, "井身结构",
                                         obj, None, props, t.source)

            elif predicate == "施工参数":
                self._insert_cement_data(cursor, well_id, "施工参数",
                                         obj, None, props, t.source)

            else:
                # 其他类型通用存储
                self._insert_cement_data(cursor, well_id, predicate,
                                         obj, None, props, t.source)

        conn.commit()
        conn.close()
        logger.info(f"SQLite 写入完成: {len(well_cache)} 口井, {len(triples)} 条记录")

    @staticmethod
    def _insert_layer(cursor: sqlite3.Cursor, well_id: int,
                      layer_type: str, props: dict):
        """插入地层记录"""
        top_depth = _parse_depth(props.get("顶深", ""))
        bottom_depth = _parse_depth(props.get("底深", ""))
        thickness = _parse_depth(props.get("厚度", ""))

        # 检查是否已存在（避免重复）
        cursor.execute(
            "SELECT layer_id FROM geological_layers WHERE well_id=? AND layer_type=? AND top_depth=?",
            (well_id, layer_type, top_depth)
        )
        if cursor.fetchone():
            return

        cursor.execute(
            """INSERT INTO geological_layers
               (well_id, layer_type, top_depth, bottom_depth, thickness, properties)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (well_id, layer_type, top_depth, bottom_depth, thickness,
             json.dumps(props, ensure_ascii=False))
        )

    @staticmethod
    def _insert_cement_data(cursor: sqlite3.Cursor, well_id: int,
                            data_type: str, value: str, unit: str,
                            props: dict, source: str):
        """插入水泥相关数据"""
        # 检查是否已存在
        cursor.execute(
            "SELECT data_id FROM cement_data WHERE well_id=? AND data_type=? AND value=?",
            (well_id, data_type, value)
        )
        if cursor.fetchone():
            return

        cursor.execute(
            """INSERT INTO cement_data
               (well_id, data_type, value, unit, properties, source_file)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (well_id, data_type, value, unit,
             json.dumps(props, ensure_ascii=False), source)
        )

    def _build_graph(self, triples: List[ExtractedTriple]):
        """构建 LlamaIndex PropertyGraphIndex 并持久化"""
        try:
            from llama_index.core.graph_stores.simple_labelled import (
                SimplePropertyGraphStore, EntityNode, Relation
            )
        except ImportError:
            logger.error("llama-index-core 版本不支持 PropertyGraphIndex，请升级")
            return

        store = SimplePropertyGraphStore()

        nodes = []
        relations = []

        for t in triples:
            subject_name = t.subject.strip()
            predicate = t.predicate.strip()
            object_name = t.object.strip()

            # 创建实体节点
            subject_node = EntityNode(
                name=subject_name,
                label="Well" if "井" in subject_name else "Entity",
                properties={"source": t.source, **t.properties},
            )

            # 根据谓语决定实体类型
            object_label = _get_entity_label(predicate, object_name)
            object_node = EntityNode(
                name=object_name,
                label=object_label,
                properties=t.properties or {},
            )

            # 创建关系
            relation = Relation(
                label=predicate,
                source_id=subject_node.id,
                target_id=object_node.id,
                properties=t.properties or {},
            )

            nodes.extend([subject_node, object_node])
            relations.append(relation)

        # 批量写入图谱
        store.upsert_nodes(nodes)
        store.upsert_relations(relations)

        # 持久化到 JSON 文件
        persist_file = str(self.graph_dir / "property_graph_store.json")
        store.persist(persist_file)
        logger.info(f"知识图谱已保存: {persist_file} ({len(triples)} 个三元组)")

    def save(self):
        """显式保存（图谱在 build 时已自动持久化）"""
        logger.info(f"数据已保存 — SQLite: {self.sqlite_path}, 图谱: {self.graph_dir}")


def _parse_depth(value: str) -> float:
    """从深度字符串中提取数值，如 '2500m' → 2500.0"""
    if not value:
        return None
    import re
    match = re.search(r'([\d.]+)', str(value))
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def _get_entity_label(predicate: str, obj: str) -> str:
    """根据谓语和宾语推断实体类型标签"""
    label_map = {
        "有地层": "GeologicalLayer",
        "水泥浆密度": "CementDensity",
        "纯砂岩厚度": "SandstoneThickness",
        "固井质量": "QualityGrade",
        "遇到异常": "Anomaly",
        "水泥浆配方": "CementFormula",
        "井身结构": "WellStructure",
        "施工参数": "OperationParam",
    }
    return label_map.get(predicate, "Property")
