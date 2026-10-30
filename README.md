\# 📊 智能数据分析 Agent



基于 LangGraph + FastAPI + DeepSeek 的自动化统计与计量分析工具。用户只需上传 CSV 文件并输入自然语言需求，系统将自动规划分析步骤、生成 Python 代码、执行代码并输出包含图表的 Markdown 分析报告。



\## ✨ 功能特性

\- \*\*自然语言驱动\*\*：只需说“做 OLS 回归”，自动完成全套统计流程。

\- \*\*自我反思与纠错\*\*：利用 LangGraph 构建状态机，包含“执行”与“评审”节点，代码出错自动重试。

\- \*\*现代化前端\*\*：原生 HTML + JS，支持 Markdown 自动渲染与图表网格展示。

\- \*\*本地化支持\*\*：自动处理 CSV 中文编码问题（GBK/UTF-8）。



\## 🛠️ 技术栈

\- \*\*后端\*\*：Python 3.11, FastAPI, LangGraph, langchain-openai

\- \*\*数据分析\*\*：Pandas, NumPy, SciPy, Statsmodels, Seaborn, Matplotlib

\- \*\*前端\*\*：HTML, CSS, Vanilla JS, Marked.js



\## 🚀 快速开始



\### 1. 安装依赖

```bash

pip install -r requirements.txt

