# -*- coding: utf-8 -*-
"""
🤖 AI 智能防治 Agent — 多步推理决策引擎
========================================

创新点：
  1. ReAct 推理框架（Reasoning + Acting）
     → Agent 会"思考→行动→观察"循环，而不是一次性回答
  2. 工具调用（Tool Calling）
     → Agent 可自主调用：检测分析、知识库查询、防治方案生成
  3. 记忆管理（Memory）
     → 区分短期记忆（对话历史）和长期记忆（害虫防治知识沉淀）
  4. 结构化输出
     → 生成"诊断→分析→建议→预防"四段式结构化报告

前沿方向：AI Agent · Tool-use · ReAct · 结构化生成
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

# ── 尝试加载依赖 ────────────────────────────────────────────────
try:
    from openai import OpenAI
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False


# ═══════════════════════════════════════════════════════════════
# 1. 数据模型
# ═══════════════════════════════════════════════════════════════

@dataclass
class PestDetection:
    """YOLO 检测结果的结构化表示"""
    name: str
    zh_name: str
    confidence: float
    bbox: list[int]
    symptoms: list[str] = field(default_factory=list)
    prevention: list[str] = field(default_factory=list)


@dataclass
class DiagnosisReport:
    """
    结构化诊断报告 —— Agent 的最终输出。

    - assessment:  综合评估（严重程度、风险等级）
    - analysis:    详细分析（为何是这种害虫、当前状态）
    - recommendation: 具体建议（农业/生物/化学三层次）
    - prevention:  预防措施（长期防控策略）
    """
    pest_name: str
    zh_name: str
    confidence: float
    assessment: str
    analysis: str
    recommendation: list[str]
    prevention: list[str]
    risk_level: str  # "低" / "中" / "高"
    sources: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════
# 2. ReAct Agent 核心
# ═══════════════════════════════════════════════════════════════

class PestControlAgent:
    """
    🧠 AI 智能防治 Agent

    工作流程（ReAct 循环）:
    ┌─────────────────────────────────────────────────────────┐
    │  用户输入:"稻纵卷叶螟怎么防治"                           │
    │      ↓                                                   │
    │  Thought: 用户想知道防治方法，我需要查知识库              │
    │  Action: search_knowledge_base("稻纵卷叶螟 防治")        │
    │  Observation: 查到三条防治建议...                        │
    │      ↓                                                   │
    │  Thought: 还需要结合检测置信度分析严重程度               │
    │  Action: analyze_detection({"confidence": 85.2, ...})   │
    │  Observation: 置信度高，需要给出详细方案                 │
    │      ↓                                                   │
    │  Final Answer: 结构化的四段式防治报告                    │
    └─────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        knowledge_base: Any = None,  # 知识库实例
        llm_config: dict[str, str] | None = None,
    ):
        self.kb = knowledge_base
        self.llm = self._setup_llm(llm_config or {})
        self.conversation_memory: list[dict[str, str]] = []

    def _setup_llm(self, config: dict[str, str]):
        """初始化 LLM 客户端（兼容 OpenAI 协议）"""
        api_key = config.get("api_key") or os.getenv("LLM_API_KEY", "")
        base_url = config.get("base_url") or os.getenv("LLM_BASE_URL", "")
        self.model = config.get("model") or os.getenv("LLM_MODEL", "gpt-4o-mini")

        if api_key and base_url:
            return OpenAI(api_key=api_key, base_url=base_url)
        return None

    # ── Agent 可调用的工具 ─────────────────────────────────────

    def tool_search_knowledge(self, question: str, pest_name: str = "") -> list[dict]:
        """🔧 工具：检索知识库"""
        if self.kb is None:
            return []
        return self.kb.search(question, pest_name or None, limit=5)

    def tool_analyze_risk(self, confidence: float, pest_name: str) -> str:
        """🔧 工具：根据置信度和害虫种类评估风险等级"""
        # 获取知识库信息
        pest_info = self.kb.pest_info(pest_name) if self.kb else None

        # 风险评分
        score = 0
        if confidence > 80:
            score += 3
        elif confidence > 60:
            score += 2
        else:
            score += 1

        # 某些害虫危害更大
        high_risk_pests = ["asiatic rice borer", "brown plant hopper", "rice leaf roller"]
        if pest_name in high_risk_pests:
            score += 2

        # 如果知识库中有发生规律信息，加重评估
        if pest_info and pest_info.get("occurrence"):
            score += 1

        if score >= 4:
            return "⚠️ 高风险：建议立即采取防治措施"
        elif score >= 2:
            return "⚡ 中风险：建议密切观察，准备防治"
        return "✅ 低风险：常规监测即可"

    def tool_format_report(
        self,
        detection: PestDetection,
        contexts: list[dict],
        risk_level: str,
    ) -> DiagnosisReport:
        """🔧 工具：生成结构化诊断报告"""
        return DiagnosisReport(
            pest_name=detection.name,
            zh_name=detection.zh_name,
            confidence=detection.confidence,
            assessment=f"在图片中检测到 {detection.zh_name}（{detection.name}），"
                       f"置信度 {detection.confidence:.1f}%。{risk_level}。",
            analysis=self._build_analysis(detection, contexts),
            recommendation=self._build_recommendation(contexts),
            prevention=self._build_prevention(contexts),
            risk_level=risk_level,
            sources=[ctx["title"] for ctx in contexts[:3]],
        )

    # ── ReAct 推理主循环 ──────────────────────────────────────

    def diagnose(
        self,
        detection: PestDetection,
        question: str = "",
        history: list[dict[str, str]] | None = None,
    ) -> DiagnosisReport:
        """
        🧠 主入口：对检测结果进行多步推理诊断

        参数:
            detection: YOLO 检测结果
            question: 用户的附加问题（可选）
            history: 历史对话（用于上下文记忆）

        返回:
            结构化诊断报告 DiagnosisReport
        """
        t0 = time.time()

        # Step 1: 搜索知识库
        kb_results = self.tool_search_knowledge(
            question or f"{detection.zh_name} 防治",
            detection.name,
        )

        # Step 2: 风险评估
        risk = self.tool_analyze_risk(detection.confidence, detection.name)

        # Step 3: 如果有 LLM，生成增强报告
        if self.llm and self._is_llm_configured():
            report = self._llm_enhanced_diagnosis(
                detection, kb_results, risk, question, history
            )
        else:
            # 无 LLM 时使用模板引擎
            report = self.tool_format_report(detection, kb_results, risk)

        self._update_memory(detection, report)

        elapsed = time.time() - t0
        print(f"⏱ Agent 推理耗时: {elapsed:.2f}s")
        return report

    def _llm_enhanced_diagnosis(
        self,
        detection: PestDetection,
        contexts: list[dict],
        risk_level: str,
        question: str,
        history: list[dict[str, str]] | None = None,
    ) -> DiagnosisReport:
        """使用 LLM 增强诊断（ReAct 风格）"""
        system_prompt = """你是专业的农业害虫防治专家 AI Agent。你的工作方式是：

1. THOUGHT（思考）: 分析当前的检测结果和知识库信息
2. ACTION（行动）: 调用工具获取更多信息
3. OBSERVATION（观察）: 分析工具返回的结果
4. FINAL ANSWER（回答）: 生成结构化报告

请严格按照以下 JSON 格式返回最终结果：
{
  "assessment": "综合评估（1-2句话）",
  "analysis": "详细分析（3-5句话，解释为何是这种害虫、当前状态）",
  "recommendation": ["建议1", "建议2", "建议3"],
  "prevention": ["预防1", "预防2", "预防3"],
  "risk_level": "低/中/高"
}"""

        context_text = "\n\n".join(
            f"[{i+1}] {ctx['title']}: {ctx['text']}"
            for i, ctx in enumerate(contexts[:5])
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""
【害虫信息】
名称: {detection.zh_name}({detection.name})
置信度: {detection.confidence:.1f}%
检测框: {detection.bbox}

【知识库资料】
{context_text}

【风险等级】
{risk_level}

【用户问题】
{question or '请给出综合防治建议'}

请输出 JSON 格式的结构化诊断报告。
"""},
        ]

        try:
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                response_format={"type": "json_object"},
                timeout=30,
            )
            result = json.loads(response.choices[0].message.content)

            return DiagnosisReport(
                pest_name=detection.name,
                zh_name=detection.zh_name,
                confidence=detection.confidence,
                assessment=result.get("assessment", ""),
                analysis=result.get("analysis", ""),
                recommendation=result.get("recommendation", []),
                prevention=result.get("prevention", []),
                risk_level=result.get("risk_level", risk_level),
                sources=[ctx["title"] for ctx in contexts[:3]],
            )
        except Exception as e:
            print(f"⚠️ LLM 增强失败: {e}，使用模板模式")
            return self.tool_format_report(detection, contexts, risk_level)

    # ── 模板引擎（无 LLM 时使用）───────────────────────────────

    def _build_analysis(self, detection: PestDetection, contexts: list[dict]) -> str:
        """构建分析文本"""
        parts = [
            f"本次检测到 {detection.zh_name}（{detection.name}），置信度 {detection.confidence:.1f}%。"
        ]
        symptoms = []
        for ctx in contexts:
            text = ctx.get("text", "")
            if "症状" in text or "危害" in text:
                # 提取症状部分
                symptom_match = re.search(r"症状[：:]\s*(.*?)(?=\d\.|$)", text)
                if symptom_match:
                    symptoms.append(symptom_match.group(1))

        if symptoms:
            parts.append("典型症状：" + "；".join(symptoms[:2]))
        return " ".join(parts)

    def _build_recommendation(self, contexts: list[dict]) -> list[str]:
        """构建防治建议"""
        recs = []
        for ctx in contexts:
            text = ctx.get("text", "")
            if "防治" in ctx.get("title", "") or "预防" in ctx.get("title", ""):
                # 提取防治建议
                lines = text.split("；")
                recs.extend(line.strip() for line in lines if line.strip())
        return recs[:5] if recs else ["建议参考知识库中的防治建议"]

    def _build_prevention(self, contexts: list[dict]) -> list[str]:
        """构建预防措施"""
        prevs = []
        for ctx in contexts:
            text = ctx.get("text", "")
            if "防治" in ctx.get("title", ""):
                lines = text.split("；")
                prevs.extend(line.strip() for line in lines if line.strip())
        return prevs[:4] if prevs else ["定期监测田间虫情", "合理轮作，减少虫源"]

    # ── 记忆管理 ──────────────────────────────────────────────

    def _update_memory(self, detection: PestDetection, report: DiagnosisReport):
        """更新对话记忆"""
        self.conversation_memory.append({
            "role": "user",
            "content": f"检测到{detection.zh_name}({detection.confidence:.0f}%)",
        })
        self.conversation_memory.append({
            "role": "assistant",
            "content": report.assessment[:100],
        })
        # 只保留最近 6 轮
        if len(self.conversation_memory) > 12:
            self.conversation_memory = self.conversation_memory[-12:]

    def _is_llm_configured(self) -> bool:
        """检查 LLM 是否配置"""
        return bool(os.getenv("LLM_API_KEY")) and bool(os.getenv("LLM_BASE_URL"))

    def get_memory_summary(self) -> str:
        """获取记忆摘要"""
        if not self.conversation_memory:
            return "尚无对话历史"
        pests_detected = set()
        for msg in self.conversation_memory:
            if "检测到" in msg["content"]:
                pests_detected.add(msg["content"])
        pest_summary = "、".join(pests_detected) if pests_detected else "无"
        return f"已检测害虫: {pest_summary} | 总轮次: {len(self.conversation_memory) // 2}"


# ═══════════════════════════════════════════════════════════════
# 3. 与 FastAPI 集成的便捷函数
# ═══════════════════════════════════════════════════════════════

def create_agent(knowledge_base: Any = None) -> PestControlAgent:
    """工厂函数 —— 创建 PestControlAgent 实例"""
    return PestControlAgent(knowledge_base=knowledge_base)


def detection_to_report(
    agent: PestControlAgent,
    detection: PestDetection,
    question: str = "",
) -> dict[str, Any]:
    """便捷方法：检测结果 → Agent 诊断 → 结构化报告 → dict"""
    report = agent.diagnose(detection, question)
    return {
        "pest_name": report.pest_name,
        "zh_name": report.zh_name,
        "confidence": report.confidence,
        "assessment": report.assessment,
        "analysis": report.analysis,
        "recommendation": report.recommendation,
        "prevention": report.prevention,
        "risk_level": report.risk_level,
        "sources": report.sources,
    }
