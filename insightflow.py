"""
InsightFlow - 智能数据决策助手
作者：Tuotuo09
功能：上传任意 Excel，自然语言提问，AI 智能分析 + 智能图表 + 决策建议 + 分页显示
支持：人事、销售、考勤、财务等任何表格数据
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
import json
import requests
import random

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="InsightFlow · 智能数据决策助手",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== 科技蓝配色 ====================
PRIMARY_BLUE = "#1E88E5"
DARK_BLUE = "#0A66C2"
LIGHT_BLUE = "#E3F2FD"
BG_GRAY = "#F8F9FA"

# ==================== 俏皮话库 ====================
LOADING_MESSAGES = [
    "🤔 让我想想...",
    "✨ 马上揭晓...",
    "🔍 数据侦探工作中...",
    "💭 正在为你整理答案...",
    "🎯 分析中，稍等片刻...",
    "📊 数据加工中...",
    "🧠 大脑飞速运转中...",
    "⚡ 即将揭晓答案...",
    "🎲 让我算一算...",
    "💡 有灵感了..."
]

# ==================== 会话状态初始化 ====================
if 'api_available' not in st.session_state:
    st.session_state.api_available = None
if 'api_checked' not in st.session_state:
    st.session_state.api_checked = False
if 'filtered_df' not in st.session_state:
    st.session_state.filtered_df = None
if 'filter_name' not in st.session_state:
    st.session_state.filter_name = None
if 'current_result_df' not in st.session_state:
    st.session_state.current_result_df = None
if 'current_chart_type' not in st.session_state:
    st.session_state.current_chart_type = None
if 'current_chart_x' not in st.session_state:
    st.session_state.current_chart_x = None
if 'current_chart_y' not in st.session_state:
    st.session_state.current_chart_y = None
if 'current_insight' not in st.session_state:
    st.session_state.current_insight = None
if 'current_recommendation' not in st.session_state:
    st.session_state.current_recommendation = None
if 'current_fun_fact' not in st.session_state:
    st.session_state.current_fun_fact = None
if 'current_summary' not in st.session_state:
    st.session_state.current_summary = None
if 'has_result' not in st.session_state:
    st.session_state.has_result = False
if 'current_df' not in st.session_state:
    st.session_state.current_df = None

# ==================== DeepSeek API 配置 ====================
DEEPSEEK_API_KEY = "sk-52bcbd3d232945828250c3a1408598ff"

def check_api_availability():
    """检查 API 是否可用"""
    if st.session_state.api_checked:
        return st.session_state.api_available
    
    if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "sk-你的密钥":
        st.session_state.api_available = False
        st.session_state.api_checked = True
        return False
    
    try:
        headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
        data = {"model": "deepseek-chat", "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1}
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=5
        )
        st.session_state.api_available = (response.status_code == 200)
        st.session_state.api_checked = True
        return st.session_state.api_available
    except:
        st.session_state.api_available = False
        st.session_state.api_checked = True
        return False

def call_deepseek(prompt):
    """调用 DeepSeek API"""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是一个专业的数据分析专家。你需要理解用户的数据分析需求，并给出深刻的洞察和可执行的建议。回答要简洁、专业、有决策价值。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 2000
    }
    
    try:
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        return None
    except:
        return None

def ai_analyze(query, filtered_df, original_df):
    """AI 模式：智能分析 + 智能图表（基于筛选后的数据）"""
    df = filtered_df if filtered_df is not None else original_df
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # 排除 ID 类字段
    id_keywords = ['id', '编号', '工号', '序号', '员工id', '用户id']
    real_numeric_cols = [c for c in numeric_cols if not any(kw in c.lower() for kw in id_keywords)]
    if not real_numeric_cols:
        real_numeric_cols = numeric_cols
    
    stats = f"数据共有{len(df)}行，{len(df.columns)}列。"
    stats += f"数值列：{', '.join(real_numeric_cols[:5])}。"
    stats += f"文本列：{', '.join(text_cols[:5])}。"
    
    # 统计摘要
    summary_stats = {}
    for col in real_numeric_cols[:3]:
        summary_stats[col] = {
            "平均值": float(round(df[col].mean(), 2)),
            "最大值": float(df[col].max()),
            "最小值": float(df[col].min())
        }
    
    for col in text_cols[:3]:
        top_values = df[col].value_counts().head(5).to_dict()
        top_values = {str(k): int(v) for k, v in top_values.items()}
        summary_stats[col] = {"前5个值": top_values}
    
    # 示例数据
    sample_data = []
    for _, row in df.head(10).iterrows():
        row_dict = {}
        for col in df.columns:
            val = row[col]
            if pd.isna(val):
                row_dict[col] = None
            elif isinstance(val, (np.integer, np.floating)):
                row_dict[col] = float(val)
            elif isinstance(val, pd.Timestamp):
                row_dict[col] = val.strftime('%Y-%m-%d')
            else:
                row_dict[col] = str(val)
        sample_data.append(row_dict)
    
    prompt = f"""用户问题：{query}

当前分析的数据共有 {len(df)} 条记录。
数据概况：
- {stats}
- 关键统计：{json.dumps(summary_stats, ensure_ascii=False)}
- 示例数据（前10行）：{json.dumps(sample_data, ensure_ascii=False)}

请基于当前数据进行分析，并返回一个**纯 JSON 对象**。

【分析要求】
1. summary: 多维度分析，包含：规模、数值特征、分布情况、关键发现，用简洁的分点形式
2. insight: 深度洞察，包含问题诊断和风险识别
3. recommendation: 可执行的行动建议，分优先级（高/中/低）
4. fun_fact: 一个有趣的数据发现

JSON 格式：
{{
    "chart_type": "pie或bar或line或none",
    "chart_x": "X轴字段名",
    "chart_y": "Y轴字段名",
    "summary": "多维度分析结果",
    "insight": "深度洞察",
    "recommendation": "行动建议（分优先级）",
    "fun_fact": "趣味事实"
}}

只返回 JSON 对象，不要有任何其他文字。"""
    
    response = call_deepseek(prompt)
    
    if response:
        try:
            result = json.loads(response)
            return result
        except:
            pass
        
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group())
                return result
            except:
                pass
        
        json_match_large = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
        if json_match_large:
            try:
                result = json.loads(json_match_large.group())
                return result
            except:
                pass
        
        return {
            "chart_type": "none",
            "summary": f"📊 数据共{len(df)}条记录，{len(df.columns)}个字段",
            "insight": "数据整体正常，建议进一步探索",
            "recommendation": "1. 高：尝试问具体问题如「部门分布」「薪资前10」",
            "fun_fact": f"数值列有{len(real_numeric_cols)}个"
        }
    
    return None

def create_chart(df, chart_type, x_col, y_col):
    """创建时尚科技感图表"""
    if chart_type == "none" or df is None or len(df) == 0:
        return None
    
    try:
        if chart_type == "pie" and x_col and y_col:
            fig = px.pie(df, names=x_col, values=y_col, title=f"{x_col}分布")
            fig.update_traces(
                marker=dict(colors=px.colors.sequential.Blues_r),
                textinfo='percent+label',
                hole=0.3
            )
            fig.update_layout(
                title_font_size=14,
                title_font_color="#1E88E5",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family="PingFang SC, Microsoft YaHei", size=12)
            )
            return fig
            
        elif chart_type == "bar" and x_col and y_col:
            fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col}排行", color=y_col)
            fig.update_traces(
                marker_color=[PRIMARY_BLUE, DARK_BLUE, "#3A6EA5", "#5A8EC5", "#7AAEE5"],
                textposition='outside'
            )
            fig.update_layout(
                title_font_size=14,
                title_font_color=PRIMARY_BLUE,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family="PingFang SC, Microsoft YaHei", size=12)
            )
            return fig
            
        elif chart_type == "line" and x_col and y_col:
            fig = px.line(df, x=x_col, y=y_col, title=f"{y_col}趋势", markers=True)
            fig.update_traces(line_color=PRIMARY_BLUE, marker_color=DARK_BLUE, marker_size=8)
            fig.update_layout(
                title_font_size=14,
                title_font_color=PRIMARY_BLUE,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family="PingFang SC, Microsoft YaHei", size=12)
            )
            return fig
            
        elif len(df.columns) >= 2:
            fig = px.bar(df, x=df.columns[0], y=df.columns[1], title="数据分布")
            fig.update_traces(marker_color=PRIMARY_BLUE)
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            return fig
    except:
        return None
    return None

def generate_dynamic_examples(df):
    """根据数据动态生成示例问题"""
    examples = []
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    id_keywords = ['id', '编号', '工号', '序号', '员工id', '用户id']
    real_numeric_cols = [c for c in numeric_cols if not any(kw in c.lower() for kw in id_keywords)]
    
    # 优先选择有意义的分类字段
    priority_cols = ['部门', '岗位', '产品', '地区', '城市', '状态', '等级', '类型', '渠道']
    selected_text = None
    for pc in priority_cols:
        if pc in text_cols:
            selected_text = pc
            break
    if not selected_text and text_cols:
        selected_text = text_cols[0]
    
    if selected_text:
        examples.append(f"「{selected_text}分布」")
        # 获取该字段的前2个常见值作为筛选示例
        top_values = df[selected_text].value_counts().head(2).index.tolist()
        for val in top_values:
            examples.append(f"「{val}」")
    
    if real_numeric_cols:
        examples.append(f"「{real_numeric_cols[0]}前10」")
        examples.append(f"「平均{real_numeric_cols[0]}」")
    
    # 去重并限制数量
    seen = set()
    unique_examples = []
    for ex in examples:
        if ex not in seen:
            seen.add(ex)
            unique_examples.append(ex)
    
    return " · ".join(unique_examples[:5])

def extract_filter_from_query(query, df):
    """从用户问题中提取筛选条件"""
    query_lower = query.lower()
    
    # 常见分类字段的筛选值
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    for col in text_cols:
        unique_vals = df[col].unique().tolist()
        for val in unique_vals:
            if str(val) in query:
                return col, str(val)
    return None, None

def display_paginated_table(df, title="数据列表", rows_per_page=10, key_prefix="table"):
    """分页显示表格"""
    if df is None or len(df) == 0:
        st.markdown(f"**{title}**（暂无数据）")
        return
    
    total_rows = len(df)
    total_pages = (total_rows + rows_per_page - 1) // rows_per_page
    
    page_key = f"page_{key_prefix}"
    if page_key not in st.session_state:
        st.session_state[page_key] = 1
    
    current_page = st.session_state[page_key]
    if current_page < 1:
        current_page = 1
        st.session_state[page_key] = 1
    if current_page > total_pages:
        current_page = total_pages
        st.session_state[page_key] = total_pages
    
    start_idx = (current_page - 1) * rows_per_page
    end_idx = min(start_idx + rows_per_page, total_rows)
    
    st.markdown(f"**{title}**（共 {total_rows} 条记录，第 {current_page} / {total_pages} 页）")
    
    page_df = df.iloc[start_idx:end_idx]
    st.dataframe(page_df, use_container_width=True)
    
    if total_pages > 1:
        col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
        with col1:
            if st.button("⏮️ 首页", key=f"first_{key_prefix}"):
                st.session_state[page_key] = 1
                st.rerun()
        with col2:
            if st.button("◀ 上一页", key=f"prev_{key_prefix}"):
                if st.session_state[page_key] > 1:
                    st.session_state[page_key] -= 1
                    st.rerun()
        with col3:
            st.markdown(f"<div style='text-align: center; padding-top: 8px; color: {PRIMARY_BLUE};'>第 {current_page} / {total_pages} 页</div>", unsafe_allow_html=True)
        with col4:
            if st.button("下一页 ▶", key=f"next_{key_prefix}"):
                if st.session_state[page_key] < total_pages:
                    st.session_state[page_key] += 1
                    st.rerun()
        with col5:
            if st.button("末页 ⏭️", key=f"last_{key_prefix}"):
                st.session_state[page_key] = total_pages
                st.rerun()

# ==================== 自定义 CSS ====================
st.markdown(f"""
<style>
    .stApp {{ background-color: {BG_GRAY}; }}
    
    .card {{
        background-color: white;
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.04);
        border: 1px solid #E5E7EB;
    }}
    
    .title {{
        font-size: 48px;
        font-weight: 700;
        background: linear-gradient(135deg, {PRIMARY_BLUE}, {DARK_BLUE});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }}
    
    .stButton > button {{
        background: linear-gradient(135deg, {PRIMARY_BLUE}, {DARK_BLUE});
        color: white;
        border: none;
        border-radius: 40px;
        padding: 10px 24px;
        font-weight: 600;
        font-size: 14px;
        transition: all 0.3s ease;
        width: auto;
    }}
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(30,136,229,0.3);
    }}
    
    .stTextInput > div > div > input {{
        border-radius: 40px;
        border: 2px solid #E5E7EB;
        padding: 12px 20px;
        font-size: 16px;
    }}
    .stTextInput > div > div > input:focus {{
        border-color: {PRIMARY_BLUE};
        box-shadow: 0 0 0 2px rgba(30,136,229,0.1);
    }}
    
    .ai-card {{
        background: linear-gradient(135deg, {LIGHT_BLUE}, white);
        border-radius: 16px;
        padding: 20px;
        border-left: 4px solid {PRIMARY_BLUE};
        margin-top: 20px;
        margin-bottom: 20px;
    }}
    
    .stat-item {{
        background: {LIGHT_BLUE};
        border-radius: 12px;
        padding: 8px 16px;
        font-size: 14px;
        display: inline-block;
        margin-right: 12px;
        margin-bottom: 8px;
    }}
    .stat-value {{
        font-weight: 700;
        color: {PRIMARY_BLUE};
        margin-left: 6px;
    }}
    
    hr {{ margin: 20px 0; border-color: #E5E7EB; }}
    
    .mode-badge-ai {{
        background-color: #4CAF50;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        display: inline-block;
        margin-bottom: 16px;
    }}
    
    [data-testid="stFileUploader"] > div:first-child {{
        background: linear-gradient(135deg, {PRIMARY_BLUE}, {DARK_BLUE});
        border-radius: 50px;
        padding: 16px 40px;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(30,136,229,0.3);
        border: none;
    }}
    [data-testid="stFileUploader"] > div:first-child:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(30,136,229,0.4);
    }}
    [data-testid="stFileUploader"] button {{
        background: transparent !important;
        color: white !important;
        font-size: 18px !important;
        font-weight: 600 !important;
        border: none !important;
        padding: 0 !important;
    }}
    [data-testid="stFileUploader"] button svg {{
        display: none;
    }}
    [data-testid="stFileUploader"] button:before {{
        content: "🚀 点击上传 Excel 或 CSV";
        font-size: 18px;
        font-weight: 600;
    }}
    [data-testid="stFileUploader"] > div:last-child {{
        display: none;
    }}
</style>
""", unsafe_allow_html=True)

# ==================== 标题区域 ====================
st.markdown("""
<div style="text-align: center; margin-bottom: 32px;">
    <h1 class="title">✨ InsightFlow · 智能数据决策助手</h1>
    <p style="font-size: 18px; color: #666; margin-top: -8px;">
        🤖 Built by Tuotuo09 · AI 数据智能平台
    </p>
    <p style="font-size: 14px; color: #888;">
        💡 随便问 · AI 帮你分析 · 智能图表 · 决策建议
    </p>
</div>
""", unsafe_allow_html=True)

# ==================== 上传区域 ====================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    uploaded_file = st.file_uploader(
        "",
        type=['xlsx', 'xls', 'csv'],
        label_visibility="collapsed"
    )

if uploaded_file:
    # 显示文件信息
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"📄 **{uploaded_file.name}** ({(uploaded_file.size / 1024):.1f} KB)")
    with col2:
        if st.button("🗑️ 删除", key="delete_btn"):
            st.session_state.filtered_df = None
            st.session_state.has_result = False
            st.rerun()
    
    # 加载数据
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    st.session_state.current_df = df
    
    # 数据详情（可展开）
    with st.expander("📋 查看数据详情"):
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        text_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        st.markdown(f"""
        <div>
            <span class="stat-item">📊 总行数：<span class="stat-value">{len(df)}</span></span>
            <span class="stat-item">📋 总列数：<span class="stat-value">{len(df.columns)}</span></span>
            <span class="stat-item">🔢 数值列：<span class="stat-value">{len(numeric_cols)}</span></span>
            <span class="stat-item">📝 文本列：<span class="stat-value">{len(text_cols)}</span></span>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("**字段列表**")
        st.dataframe(df.dtypes.reset_index().rename(columns={'index': '字段', 0: '类型'}), use_container_width=True)
        st.write("**数据预览（前100行）**")
        st.dataframe(df.head(100), use_container_width=True)
    
    # 问答区域
    st.markdown("---")
    
    query = st.text_input("", placeholder="例如：哪个部门人最多？｜薪资合理吗？｜给我一些建议", label_visibility="collapsed")
    analyze_btn = st.button("🚀 开始分析", type="primary")
    
    # 动态生成示例问题
    examples = generate_dynamic_examples(df)
    st.caption(f"💡 试试这些：{examples}")
    
    # ==================== 分析逻辑 ====================
    if analyze_btn and query:
        api_ok = check_api_availability()
        
        if api_ok:
            st.markdown('<div><span class="mode-badge-ai">🤖 AI 智能模式</span></div>', unsafe_allow_html=True)
            
            # 提取筛选条件
            filter_col, filter_val = extract_filter_from_query(query, df)
            
            if filter_col and filter_val:
                display_df = df[df[filter_col] == filter_val]
                filter_name = filter_val
            else:
                display_df = df
                filter_name = None
            
            loading_msg = random.choice(LOADING_MESSAGES)
            with st.spinner(loading_msg):
                ai_result = ai_analyze(query, display_df, df)
                
                if ai_result:
                    chart_type = ai_result.get("chart_type", "none")
                    chart_x = ai_result.get("chart_x", None)
                    chart_y = ai_result.get("chart_y", None)
                    summary_text = ai_result.get("summary", "")
                    insight = ai_result.get("insight", "")
                    recommendation = ai_result.get("recommendation", "")
                    fun_fact = ai_result.get("fun_fact", "")
                    
                    # 准备图表数据
                    result_df = None
                    if chart_type != "none" and chart_x and chart_y and chart_x in display_df.columns and chart_y in display_df.columns:
                        if chart_type == "pie":
                            result_df = display_df.groupby(chart_x)[chart_y].sum().reset_index()
                        elif chart_type in ["bar", "line"]:
                            result_df = display_df.groupby(chart_x)[chart_y].sum().reset_index()
                            result_df = result_df.sort_values(chart_y, ascending=False).head(10)
                    
                    # 保存所有结果到 session_state
                    st.session_state.filtered_df = display_df
                    st.session_state.filter_name = filter_name
                    st.session_state.current_result_df = result_df
                    st.session_state.current_chart_type = chart_type
                    st.session_state.current_chart_x = chart_x
                    st.session_state.current_chart_y = chart_y
                    st.session_state.current_summary = summary_text
                    st.session_state.current_insight = insight
                    st.session_state.current_recommendation = recommendation
                    st.session_state.current_fun_fact = fun_fact
                    st.session_state.has_result = True
                    
                    # 重置分页
                    page_key = "page_table"
                    if page_key in st.session_state:
                        st.session_state[page_key] = 1
                    
                    # 显示结果
                    st.markdown("---")
                    st.markdown("### 📊 分析结果")
                    
                    if summary_text:
                        st.markdown(summary_text)
                    
                    # AI 智能洞察区域
                    st.markdown('<div class="ai-card">', unsafe_allow_html=True)
                    st.markdown("### 🧠 Tuotuo's AI 智能洞察")
                    
                    if insight:
                        st.markdown(f"**🔍 洞察**\n{insight}")
                    if recommendation:
                        st.markdown(f"\n**🎯 建议**\n{recommendation}")
                    if fun_fact:
                        st.markdown(f"\n**📌 趣味发现**\n{fun_fact}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # 表格 + 图表（左右布局）
                    col_left, col_right = st.columns([3, 2])
                    
                    with col_left:
                        if filter_name:
                            title = f"{filter_name}数据明细"
                        else:
                            title = "数据明细"
                        display_paginated_table(display_df, title, rows_per_page=10, key_prefix="table")
                    
                    with col_right:
                        if result_df is not None:
                            fig = create_chart(result_df, chart_type, chart_x, chart_y)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                        elif text_cols and chart_type != "none":
                            priority_cols = ['部门', '岗位', '产品', '地区', '类型']
                            chart_col = text_cols[0]
                            for pc in priority_cols:
                                if pc in text_cols:
                                    chart_col = pc
                                    break
                            default_data = display_df[chart_col].value_counts().head(5)
                            if len(default_data) > 0:
                                fig = px.pie(values=default_data.values, names=default_data.index, title=f"{chart_col}分布")
                                fig.update_traces(marker=dict(colors=px.colors.sequential.Blues_r), hole=0.3)
                                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                else:
                    st.warning("🤖 AI 服务繁忙，请稍后再试")
        
        else:
            st.warning("🤖 AI 服务繁忙，请稍后再试")
    
    elif analyze_btn and not query:
        st.warning("💡 请输入一个问题～")
    
    # ==================== 显示上次结果（用于翻页后保持）====================
    elif st.session_state.has_result and not analyze_btn:
        st.markdown("---")
        st.markdown("### 📊 分析结果")
        
        if st.session_state.current_summary:
            st.markdown(st.session_state.current_summary)
        
        st.markdown('<div class="ai-card">', unsafe_allow_html=True)
        st.markdown("### 🧠 Tuotuo's AI 智能洞察")
        
        if st.session_state.current_insight:
            st.markdown(f"**🔍 洞察**\n{st.session_state.current_insight}")
        if st.session_state.current_recommendation:
            st.markdown(f"\n**🎯 建议**\n{st.session_state.current_recommendation}")
        if st.session_state.current_fun_fact:
            st.markdown(f"\n**📌 趣味发现**\n{st.session_state.current_fun_fact}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        col_left, col_right = st.columns([3, 2])
        
        with col_left:
            if st.session_state.filter_name:
                title = f"{st.session_state.filter_name}数据明细"
            else:
                title = "数据明细"
            if st.session_state.filtered_df is not None:
                display_paginated_table(st.session_state.filtered_df, title, rows_per_page=10, key_prefix="table")
        
        with col_right:
            if st.session_state.current_result_df is not None:
                fig = create_chart(
                    st.session_state.current_result_df,
                    st.session_state.current_chart_type,
                    st.session_state.current_chart_x,
                    st.session_state.current_chart_y
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

else:
    # 引导页面
    st.markdown("""
    <div style="margin-top: 32px;">
        <div class="card" style="text-align: center;">
            <div style="font-size: 20px; margin-bottom: 16px;">🎯 能做什么</div>
            <div style="display: flex; justify-content: center; gap: 32px; flex-wrap: wrap;">
                <div>📊 数据分布</div>
                <div>📈 趋势分析</div>
                <div>🏆 排名对比</div>
                <div>🎯 决策建议</div>
                <div>📌 趣味发现</div>
                <div>📉 智能图表</div>
            </div>
        </div>
    </div>
    
    <div style="margin-top: 24px; text-align: center; font-size: 14px; color: #888;">
        <p>💡 上传你的数据（人事、销售、考勤、财务...），随便问</p>
        <p>🤖 例如：「哪个部门人最多？」「给我一些建议」</p>
    </div>
    """, unsafe_allow_html=True)

# ==================== 底部 ====================
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 16px; font-size: 12px; color: #888;">
    ⚡ Made with ☕ by Tuotuo09 · Powered by DeepSeek AI<br>
    🔒 数据本地处理 · 不上传服务器 · 隐私安全
</div>
""", unsafe_allow_html=True)
