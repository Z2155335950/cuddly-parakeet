import os
import re
import io
import time
import math
import shutil
import uuid
import contextlib
import traceback
from typing import TypedDict, Optional
import matplotlib
matplotlib.use("Agg")
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# ============ 环境与LLM初始化 ============
load_dotenv()

base_url = os.getenv("LLM_BASE_URL")
api_key = os.getenv("LLM_API_KEY")
model_name = os.getenv("LLM_MODEL")

print("===== LLM配置信息 =====")
print("base_url:", base_url)
print("api_key:", api_key[:10] + "******" if api_key else api_key)
print("model_name:", model_name)

if not all([base_url, api_key, model_name]):
    raise ValueError("❌ .env读取失败！检查：文件名是不是.env、和backend.py同级、不要有多余空格")

llm = ChatOpenAI(
    base_url=base_url,
    api_key=api_key,
    model=model_name,
    temperature=0.2
)

plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============ State ============
class AgentState(TypedDict):
    query: str
    columns: list
    data_summary: str
    plan: str
    code: str
    exec_result: str
    critic_result: str
    report: str
    retry_count: int

# ============ 工具函数 ============
def get_data_summary(df: pd.DataFrame) -> str:
    buf = io.StringIO()
    df.info(buf=buf)
    info_str = buf.getvalue()
    desc = df.describe().to_string()
    na_info = df.isnull().sum().to_string()
    return f"【数据信息】\n{info_str}\n【描述统计】\n{desc}\n【缺失值】\n{na_info}"

def strip_code_fences(code: str) -> str:
    code = (code or "").strip()
    m = re.match(r"^```[A-Za-z0-9_+.-]*[ \t]*\r?\n(.*?)\r?\n?```$", code, re.S)
    if m:
        return m.group(1).strip()
    return code

def read_csv_robust(path: str) -> pd.DataFrame:
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as e:
            last_err = e
    raise last_err

# ============ LangGraph 节点 ============
def planner_node(state: AgentState) -> AgentState:
    prompt = f"""
你是统计分析规划师。
数据集字段：{state['columns']}
数据概览：{state['data_summary']}
用户需求：{state['query']}
输出有序分析步骤，每一步写明：目标 + 要用的统计方法 + 是否绘图。
规则：只用本科统计、计量经济学方法；区分定性/定量变量；不写代码。
输出格式：1.xxx 2.xxx
"""
    resp = llm.invoke(prompt)
    state["plan"] = resp.content
    return state

def coder_node(state: AgentState) -> AgentState:
    prompt = f"""
根据下面的分析计划，生成可直接运行的Python代码。
计划：{state['plan']}
数据集变量信息：{state['data_summary']}
可用包：pandas, numpy, scipy, statsmodels, seaborn, matplotlib, os, time, math
（已注入命名空间，直接用即可：df 是数据，OUTPUT_DIR 是图片目录）
要求：
1. 数据已经存入变量 df，不要写读取文件代码
2. 图片保存到 OUTPUT_DIR 目录，文件名唯一，建议使用时间戳命名
3. 打印关键统计结果（p值、R²、系数、VIF等）
4. 代码简洁，增加简单异常处理
5. f-string 的花括号里不能出现反斜杠或嵌套同种引号
6. 代码必须语法正确、可直接执行
⚠️【硬性要求】只输出纯代码文本！严禁输出 ```python、解释文字！
"""
    resp = llm.invoke(prompt)
    state["code"] = strip_code_fences(resp.content)
    return state

def safe_exec(code: str, df: pd.DataFrame) -> str:
    ns = {
        "df": df.copy(),
        "pd": pd, "np": np, "sns": sns, "plt": plt,
        "stats": stats, "sm": sm, "variance_inflation_factor": variance_inflation_factor,
        "OUTPUT_DIR": OUTPUT_DIR,
        "os": os, "re": re, "time": time, "math": math,
    }
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, ns, ns)
        return buf.getvalue()
    except Exception:
        return f"代码执行报错：{traceback.format_exc()}"
    finally:
        plt.close("all")

def executor_node(state: AgentState, df: pd.DataFrame) -> AgentState:
    res = safe_exec(state["code"], df)
    state["exec_result"] = res
    return state

def exec_failed(state: AgentState) -> bool:
    return "代码执行报错" in (state.get("exec_result") or "")

def critic_node(state: AgentState) -> AgentState:
    prompt = f"""
你是统计学评审，检查本次分析。
计划：{state['plan']}
代码输出结果：{state['exec_result']}
代码：{state['code']}
检查清单：
1. 统计方法适配变量类型？（分类变量不能直接皮尔逊相关）
2. 检验前提是否满足（样本量、正态性）
3. 是否有明显逻辑错误、数值异常
4. 计量回归：多重共线性、样本缺失是否忽略
5. 代码是否真的跑通（出现“代码执行报错”一律判 FAIL）

输出格式严格：
状态：PASS / FAIL
理由：xxx
修正建议：xxx
"""
    resp = llm.invoke(prompt)
    state["critic_result"] = resp.content
    if "FAIL" in (resp.content or "").upper() or exec_failed(state):
        state["retry_count"] = state.get("retry_count", 0) + 1
    return state

def reporter_node(state: AgentState) -> AgentState:
    prompt = f"""
整理本次全部步骤、统计数值、图表路径，生成一份易懂的数据分析Markdown报告。
包含：分析目标、描述统计、检验结果、图表说明、结论、研究局限性。
专业但不晦涩，标注p值、显著性、R²等关键指标。
计划：{state['plan']}
运行结果：{state['exec_result']}
评审意见：{state['critic_result']}
"""
    resp = llm.invoke(prompt)
    state["report"] = resp.content
    return state

def should_retry(state: AgentState) -> str:
    failed = exec_failed(state) or "FAIL" in (state["critic_result"] or "").upper()
    if not failed:
        return "report"
    if state.get("retry_count", 0) >= 3:
        return "report"
    return "code"

# ============ 核心执行入口 ============
def run_agent_from_path(csv_path: str, query: str):
    df = read_csv_robust(csv_path)
    cols = df.columns.tolist()
    summary = get_data_summary(df)
    init_state: AgentState = {
        "query": query,
        "columns": cols,
        "data_summary": summary,
        "plan": "", "code": "", "exec_result": "",
        "critic_result": "", "report": "", "retry_count": 0,
    }

    builder = StateGraph(AgentState)
    builder.add_node("plan", planner_node)
    builder.add_node("code", coder_node)
    builder.add_node("exec", lambda s: executor_node(s, df))
    builder.add_node("critic", critic_node)
    builder.add_node("report", reporter_node)
    builder.set_entry_point("plan")
    builder.add_edge("plan", "code")
    builder.add_edge("code", "exec")
    builder.add_edge("exec", "critic")
    builder.add_conditional_edges("critic", should_retry, {"code": "code", "report": "report"})
    builder.add_edge("report", END)
    graph = builder.compile()

    before = set(os.listdir(OUTPUT_DIR))
    final = graph.invoke(init_state)

    img_names = [
        name
        for name in sorted(set(os.listdir(OUTPUT_DIR)) - before)
        if name.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    return final["report"], img_names

# ============ FastAPI 应用 ============
app = FastAPI(title="数据分析Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载图片目录，前端可访问 /images/xxx.png
app.mount("/images", StaticFiles(directory=OUTPUT_DIR), name="images")

# 挂载前端静态页，注意这里改成了 "/ui"
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/ui", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), query: str = Form(...)):
    suffix = os.path.splitext(file.filename or "")[1] or ".csv"
    tmp_path = os.path.join(OUTPUT_DIR, f"upload_{uuid.uuid4().hex}{suffix}")
    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        report, images = run_agent_from_path(tmp_path, query)
        image_urls = [f"/images/{name}" for name in images]
        return {"report": report, "images": image_urls}
    except Exception:
        return JSONResponse(status_code=500, content={"error": traceback.format_exc()})
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50)
    print("🚀 服务已启动！请在浏览器打开： http://127.0.0.1:8000/ui")
    print("="*50 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)