"""
InsightFlow - 智能数据决策助手
作者：Tuotuo09
功能：上传任意 Excel，自然语言提问，AI 智能分析 + 智能图表 + 决策建议 + 分页显示
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
            {"role": "system", "content": "你是一个专业的数据分析专家。你需要理解用户的数据分析需求，并给出清晰的洞察和建议。回答要简洁、专业、有决策价值。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 1500
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

def ai_analyze(query, df):
    """AI 模式：智能分析 + 智能图表"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # 排除 ID 类字段
    id_keywords = ['id', '编号', '工号', '序号', '员工id']
    real_numeric_cols = [c for c in numeric_cols if not any(kw in c.lower() for kw in id_keywords)]
    if not real_numeric_cols:
        real_numeric_cols = numeric_cols
    
    # 获取所有可能的部门名称
    dept_names = []
    if '部门' in df.columns:
        dept_names = df['部门'].unique().tolist()
    
    stats = f"数据共有{len(df)}行，{len(df.columns)}列。"
    stats += f"数值列：{', '.join(real_numeric_cols[:5])}。"
    stats += f"文本列：{', '.join(text_cols[:5])}。"
    if dept_names:
        stats += f"部门列表：{', '.join(dept_names[:10])}。"
    
    # 统计摘要
    summary = {}
    for col in real_numeric_cols[:3]:
        summary[col] = {
            "平均值": float(round(df[col].mean(), 2)),
            "最大值": float(df[col].max()),
            "最小值": float(df[col].min())
        }
    
    for col in text_cols[:3]:
        top_values = df[col].value_counts().head(5).to_dict()
        top_values = {str(k): int(v) for k, v in top_values.items()}
        summary[col] = {"前5个值": top_values}
    
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

数据概况：
- {stats}
- 关键统计：{json.dumps(summary, ensure_ascii=False)}
- 示例数据（前10行）：{json.dumps(sample_data, ensure_ascii=False)}

【重要规则】
1. 如果用户问的是具体部门（如技术部、销售部、人力资源部、产品部、市场部、财务部、运营部），必须在 filter 字段填写部门名称
2. 例如用户问「人力资源部人员」，filter 应该是 "部门=人力资源部"
3. 例如用户问「技术部平均年龄」，filter 应该是 "部门=技术部"
4. 如果用户没有指定部门，filter 填 null
5. 如果用户问「分布」「占比」，用 pie 图
6. 如果用户问「前几名」「排名」「最高」，用 bar 图
7. 如果用户问「趋势」「变化」，用 line 图

请分析用户问题，并返回一个**纯 JSON 对象**。

JSON 格式：
{{
    "chart_type": "pie或bar或line或none",
    "chart_x": "X轴字段名",
    "chart_y": "Y轴字段名",
    "filter": "筛选条件，如：部门=技术部，如果没有筛选填 null",
    "insight": "数据洞察（一句话）",
    "recommendation": "决策建议（具体可操作）",
    "fun_fact": "趣味事实",
    "summary": "分析结果总结"
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
            "filter": None,
            "insight": f"数据共{len(df)}条记录",
            "recommendation": "试试问：「部门分布」「薪资前10」",
            "fun_fact": f"数值列有{len(real_numeric_cols)}个",
            "summary": "分析完成"
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
    
    id_keywords = ['id', '编号', '工号', '序号', '员工id']
    real_numeric_cols = [c for c in numeric_cols if not any(kw in c.lower() for kw in id_keywords)]
    
    priority_cols = ['部门', '岗位', '地区', '城市', '状态', '等级']
    selected_text = None
    for pc in priority_cols:
        if pc in text_cols:
            selected_text = pc
            break
    if not selected_text and text_cols:
        selected_text = text_cols[0]
    
    if selected_text:
        examples.append(f"「{selected_text}分布」")
    
    if real_numeric_cols:
        examples.append(f"「{real_numeric_cols[0]}前10」")
        examples.append(f"「平均{real_numeric_cols[0]}」")
    
    if len(text_cols) > 1:
        examples.append(f"「{text_cols[1]}」")
    
    return " · ".join(examples[:4])

def display_paginated_table(df, title="数据列表", rows_per_page=10, key_prefix="table"):
    """分页显示表格"""
    if df is None or len(df) == 0:
        st.info("暂无数据")
        return
    
    total_rows = len(df)
    total_pages = (total_rows + rows_per_page - 1) // rows_per_page
    
    # 使用独立的 session_state key
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
    
    # 分页按钮
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
            # 清空所有保存的结果
            st.session_state.filtered_df = None
            st.session_state.has_result = False
            st.rerun()
    
    # 加载数据
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
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
    
    examples = generate_dynamic_examples(df)
    st.caption(f"💡 试试这些：{examples}")
    
    # ==================== 分析逻辑 ====================
    if analyze_btn and query:
        api_ok = check_api_availability()
        
        if api_ok:
            st.markdown('<div><span class="mode-badge-ai">🤖 AI 智能模式</span></div>', unsafe_allow_html=True)
            
            loading_msg = random.choice(LOADING_MESSAGES)
            with st.spinner(loading_msg):
                ai_result = ai_analyze(query, df)
                
                if ai_result:
                    chart_type = ai_result.get("chart_type", "none")
                    chart_x = ai_result.get("chart_x", None)
                    chart_y = ai_result.get("chart_y", None)
                    filter_condition = ai_result.get("filter", None)
                    insight = ai_result.get("insight", "")
                    recommendation = ai_result.get("recommendation", "")
                    fun_fact = ai_result.get("fun_fact", "")
                    summary = ai_result.get("summary", "")
                    
                    # 处理筛选
                    display_df = df
                    filter_name = None
                    if filter_condition and "=" in filter_condition:
                        filter_field, filter_value = filter_condition.split("=")
                        filter_field = filter_field.strip()
                        filter_value = filter_value.strip()
                        if filter_field in df.columns:
                            display_df = df[df[filter_field] == filter_value]
                            filter_name = filter_value
                    
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
                    st.session_state.current_insight = insight
                    st.session_state.current_recommendation = recommendation
                    st.session_state.current_fun_fact = fun_fact
                    st.session_state.current_summary = summary
                    st.session_state.has_result = True
                    
                    # 重置分页页码
                    page_key = "page_table"
                    if page_key in st.session_state:
                        st.session_state[page_key] = 1
                    
                    # 显示结果
                    st.markdown("---")
                    st.markdown("### 📊 分析结果")
                    
                    if summary:
                        st.markdown(f"**{summary}**")
                    
                    # AI 洞察区域（放在表格前面）
                    st.markdown('<div class="ai-card">', unsafe_allow_html=True)
                    st.markdown("### 🧠 Tuotuo's AI 智能洞察")
                    
                    if insight:
                        st.markdown(f"🔍 **洞察**：{insight}")
                    if recommendation:
                        st.markdown(f"🎯 **决策建议**：{recommendation}")
                    if fun_fact:
                        st.markdown(f"📌 **有趣的数据**：{fun_fact}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # 表格 + 图表（左右布局）
                    col_left, col_right = st.columns([3, 2])
                    
                    with col_left:
                        if filter_name:
                            title = f"{filter_name}员工名单"
                        else:
                            title = "数据列表"
                        display_paginated_table(display_df, title, rows_per_page=10, key_prefix="table")
                    
                    with col_right:
                        if result_df is not None:
                            fig = create_chart(result_df, chart_type, chart_x, chart_y)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                        elif text_cols and chart_type != "none":
                            priority_cols = ['部门', '岗位', '地区', '状态']
                            chart_col = text_cols[0]
                            for pc in priority_cols:
                                if pc in text_cols:
                                    chart_col = pc
                                    break
                            default_data = display_df[chart_col].value_counts().head(5)
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
            st.markdown(f"**{st.session_state.current_summary}**")
        
        # AI 洞察区域（放在表格前面）
        st.markdown('<div class="ai-card">', unsafe_allow_html=True)
        st.markdown("### 🧠 Tuotuo's AI 智能洞察")
        
        if st.session_state.current_insight:
            st.markdown(f"🔍 **洞察**：{st.session_state.current_insight}")
        if st.session_state.current_recommendation:
            st.markdown(f"🎯 **决策建议**：{st.session_state.current_recommendation}")
        if st.session_state.current_fun_fact:
            st.markdown(f"📌 **有趣的数据**：{st.session_state.current_fun_fact}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 表格 + 图表（左右布局）
        col_left, col_right = st.columns([3, 2])
        
        with col_left:
            if st.session_state.filter_name:
                title = f"{st.session_state.filter_name}员工名单"
            else:
                title = "数据列表"
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
            elif '部门' in df.columns:
                default_data = df['部门'].value_counts().head(5)
                fig = px.pie(values=default_data.values, names=default_data.index, title="部门分布")
                fig.update_traces(marker=dict(colors=px.colors.sequential.Blues_r), hole=0.3)
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
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
                <div>📌 有趣的数据</div>
                <div>📉 智能图表</div>
            </div>
        </div>
    </div>
    
    <div style="margin-top: 24px; text-align: center; font-size: 14px; color: #888;">
        <p>💡 上传你的数据，随便问</p>
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
