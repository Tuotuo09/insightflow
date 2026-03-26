"""
InsightFlow - 智能数据决策助手（通用版 + 字段选择器）
作者：Tuotuo09
功能：用户选择分析维度，工具准确计算，AI 智能解读
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
]

# ==================== 会话状态初始化 ====================
if 'api_available' not in st.session_state:
    st.session_state.api_available = None
if 'api_checked' not in st.session_state:
    st.session_state.api_checked = False
if 'has_result' not in st.session_state:
    st.session_state.has_result = False
if 'current_df' not in st.session_state:
    st.session_state.current_df = None
if 'filtered_df' not in st.session_state:
    st.session_state.filtered_df = None
if 'filter_name' not in st.session_state:
    st.session_state.filter_name = None
if 'filter_col' not in st.session_state:
    st.session_state.filter_col = None
if 'group_stats' not in st.session_state:
    st.session_state.group_stats = None
if 'numeric_stats' not in st.session_state:
    st.session_state.numeric_stats = None
if 'ai_response' not in st.session_state:
    st.session_state.ai_response = None
if 'total_rows' not in st.session_state:
    st.session_state.total_rows = 0
if 'total_columns' not in st.session_state:
    st.session_state.total_columns = 0
if 'group_col' not in st.session_state:
    st.session_state.group_col = None
if 'value_col' not in st.session_state:
    st.session_state.value_col = None

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
            {"role": "system", "content": "你是一个专业的数据分析专家。基于提供的准确数据，给出深刻的洞察和可执行的建议。"},
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

def clean_dataframe(df):
    """清洗数据框"""
    valid_columns = [c for c in df.columns if not str(c).startswith('Unnamed')]
    df = df[valid_columns]
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna('')
    return df

def detect_filter_from_query(query, df):
    """从用户问题中提取筛选条件（通用版）"""
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    for col in text_cols:
        unique_vals = df[col].dropna().unique().tolist()
        for val in unique_vals:
            if val and str(val) in query:
                return col, str(val)
    return None, None

def generate_filtered_chart(df, filter_col, filter_val, group_col, value_col):
    """根据筛选条件生成对应图表"""
    try:
        if group_col in df.columns and value_col in df.columns:
            grouped = df.groupby(group_col)[value_col].sum().reset_index()
            if len(grouped) > 0:
                fig = px.bar(grouped, x=group_col, y=value_col, title=f"📊 {filter_val} - {group_col}分布")
                fig.update_traces(marker_color=PRIMARY_BLUE)
                return fig
    except:
        pass
    return None

def precompute_stats(df, group_col, value_col):
    """工具准确计算各项统计（使用用户选择的字段）"""
    
    # 整体统计
    total_rows = len(df)
    total_columns = len(df.columns)
    
    # 自动识别数值列和文本列
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # 排除 ID 类字段
    id_keywords = ['id', '编号', '工号', '序号', '用户id', '员工id']
    real_numeric_cols = [c for c in numeric_cols if not any(kw in c.lower() for kw in id_keywords)]
    
    # 分组统计（使用用户选择的字段）
    group_stats = None
    if group_col and group_col in df.columns and value_col and value_col in df.columns:
        grouped = df.groupby(group_col)[value_col].sum().reset_index()
        grouped.columns = [group_col, value_col]
        group_stats = grouped.sort_values(value_col, ascending=False)
    
    # 数值列统计（所有数值列，不限制数量）
    numeric_stats = None
    if real_numeric_cols:
        stats_data = []
        for col in real_numeric_cols:
            stats_data.append({
                "字段": col,
                "总和": round(df[col].sum(), 2),
                "平均值": round(df[col].mean(), 2),
                "最大值": df[col].max(),
                "最小值": df[col].min()
            })
        numeric_stats = pd.DataFrame(stats_data)
    
    return {
        "total_rows": total_rows,
        "total_columns": total_columns,
        "text_cols": text_cols,
        "numeric_cols": real_numeric_cols,
        "group_stats": group_stats,
        "numeric_stats": numeric_stats,
        "group_col": group_col,
        "value_col": value_col
    }

def generate_ai_insight(query, stats):
    """生成 AI 洞察（发送完整数据）"""
    summary = f"用户问题：{query}\n\n"
    summary += f"【数据概况】共 {stats['total_rows']} 行，{stats['total_columns']} 列\n"
    summary += f"【文本字段】{', '.join(stats['text_cols'][:10])}\n"
    summary += f"【数值字段】{', '.join(stats['numeric_cols'][:10])}\n\n"
    
    # 发送用户选择的分组统计（全部发送）
    if stats['group_stats'] is not None and len(stats['group_stats']) > 0:
        group_col = stats['group_col']
        value_col = stats['value_col']
        summary += f"【{group_col} 分组统计（按 {value_col} 求和）】\n"
        for _, row in stats['group_stats'].iterrows():
            summary += f"- {row[group_col]}: {row[value_col]:.0f}\n"
    
    # 发送所有数值统计（不限制数量）
    if stats['numeric_stats'] is not None and len(stats['numeric_stats']) > 0:
        summary += "\n【所有数值字段统计】\n"
        for _, row in stats['numeric_stats'].iterrows():
            summary += f"- {row['字段']}: 总和{row['总和']:.0f}, 平均{row['平均值']:.1f}, 最大{row['最大值']}, 最小{row['最小值']}\n"
    
    # 增加关键摘要
    total_rows = stats['total_rows']
    summary += f"\n【关键摘要】\n"
    summary += f"- 总记录数: {total_rows}\n"
    
    # 列出所有数值字段的平均值
    if stats['numeric_stats'] is not None and len(stats['numeric_stats']) > 0:
        for _, row in stats['numeric_stats'].iterrows():
            summary += f"- 平均{row['字段']}: {row['平均值']:.1f}\n"
    
    prompt = f"""基于以下准确数据，给出洞察和建议：

{summary}

请按格式输出：
【洞察】：（2-3点，基于数据发现的核心问题）
【建议】：（分优先级：高/中/低，具体可执行）
【趣味发现】：（一个有趣的数据洞察）"""
    
    return call_deepseek(prompt)

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
    }}
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(30,136,229,0.3);
    }}
    .stSelectbox label {{
        font-weight: 600;
        color: {PRIMARY_BLUE};
    }}
    .stTextInput > div > div > input {{
        border-radius: 40px;
        border: 2px solid #E5E7EB;
        padding: 12px 20px;
        font-size: 16px;
    }}
    .ai-card {{
        background: linear-gradient(135deg, {LIGHT_BLUE}, white);
        border-radius: 16px;
        padding: 20px;
        border-left: 4px solid {PRIMARY_BLUE};
        margin-top: 20px;
    }}
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
    }}
    [data-testid="stFileUploader"] button {{
        background: transparent !important;
        color: white !important;
        font-size: 18px !important;
        font-weight: 600 !important;
    }}
    [data-testid="stFileUploader"] button svg {{ display: none; }}
    [data-testid="stFileUploader"] button:before {{
        content: "🚀 点击上传 Excel 或 CSV";
    }}
</style>
""", unsafe_allow_html=True)

# ==================== 标题 ====================
st.markdown("""
<div style="text-align: center; margin-bottom: 32px;">
    <h1 class="title">✨ InsightFlow · 智能数据决策助手</h1>
    <p style="font-size: 18px; color: #666; margin-top: -8px;">
        🤖 Built by Tuotuo09 · 用户选择维度 · 工具准确计算 · AI 智能解读
    </p>
</div>
""", unsafe_allow_html=True)

# ==================== 上传区域 ====================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    uploaded_file = st.file_uploader("", type=['xlsx', 'xls', 'csv'], label_visibility="collapsed")

if uploaded_file:
    # 加载数据
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    df = clean_dataframe(df)
    st.session_state.current_df = df
    
    # 数据详情
    with st.expander("📋 查看数据详情"):
        st.dataframe(df.head(100), use_container_width=True)
        st.caption(f"共 {len(df)} 行，{len(df.columns)} 列")
    
    # ==================== 字段选择器 ====================
    st.markdown("---")
    st.markdown("### 🎯 选择分析维度")
    
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 排除 ID 类字段
    id_keywords = ['id', '编号', '工号', '序号', '用户id', '员工id']
    real_numeric_cols = [c for c in numeric_cols if not any(kw in c.lower() for kw in id_keywords)]
    
    col1, col2 = st.columns(2)
    with col1:
        group_col = st.selectbox("📊 分组字段（X轴）", text_cols, key="group_col")
    with col2:
        if real_numeric_cols:
            value_col = st.selectbox("📈 数值字段（Y轴）", real_numeric_cols, key="value_col")
        else:
            value_col = None
            st.warning("⚠️ 没有找到数值字段，请确保数据包含数字列")
    
    # 问答区域
    st.markdown("---")
    query = st.text_input("", placeholder="例如：给我一些建议 | 分析一下数据", label_visibility="collapsed")
    analyze_btn = st.button("🚀 开始分析", type="primary")
    
    # 动态示例
    if text_cols and real_numeric_cols:
        st.caption(f"💡 试试：给我一些建议 · {text_cols[0]}分布 · 筛选条件会自动识别")
    else:
        st.caption("💡 试试：给我一些建议")
    
    # ==================== 分析逻辑 ====================
    if analyze_btn and query and value_col:
        # 检测筛选条件
        filter_col, filter_val = detect_filter_from_query(query, df)
        
        if filter_col and filter_val and filter_col in df.columns:
            display_df = df[df[filter_col] == filter_val]
            st.info(f"🔍 已筛选：{filter_col} = {filter_val}，共 {len(display_df)} 条记录")
        else:
            display_df = df
            filter_col = None
            filter_val = None
        
        # 计算统计（使用用户选择的字段）
        stats = precompute_stats(display_df, group_col, value_col)
        
        # 保存到 session_state
        st.session_state.has_result = True
        st.session_state.filtered_df = display_df
        st.session_state.filter_name = filter_val
        st.session_state.filter_col = filter_col
        st.session_state.group_stats = stats['group_stats']
        st.session_state.numeric_stats = stats['numeric_stats']
        st.session_state.total_rows = stats['total_rows']
        st.session_state.total_columns = stats['total_columns']
        st.session_state.group_col = group_col
        st.session_state.value_col = value_col
        
        # 显示整体指标
        st.markdown("---")
        st.markdown("### 📈 整体指标")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("总行数", stats['total_rows'])
        with col2:
            st.metric("总列数", stats['total_columns'])
        with col3:
            st.metric("数值字段", len(stats['numeric_cols']))
        
        # 显示分组统计表格（用户选择的维度）
        if stats['group_stats'] is not None and len(stats['group_stats']) > 0:
            st.markdown(f"### 📊 {group_col} 分布（按 {value_col} 求和）")
            st.dataframe(stats['group_stats'], use_container_width=True)
            
            # 显示图表
            fig = px.bar(stats['group_stats'], x=group_col, y=value_col, title=f"{group_col} 分布")
            fig.update_traces(marker_color=PRIMARY_BLUE)
            st.plotly_chart(fig, use_container_width=True)
        
        # 显示所有数值字段统计
        if stats['numeric_stats'] is not None and len(stats['numeric_stats']) > 0:
            st.markdown("### 🔢 所有数值字段统计")
            st.dataframe(stats['numeric_stats'], use_container_width=True)
        
        # AI 智能洞察
        api_ok = check_api_availability()
        if api_ok:
            with st.spinner(random.choice(LOADING_MESSAGES)):
                ai_response = generate_ai_insight(query, stats)
                if ai_response:
                    st.session_state.ai_response = ai_response
                    st.markdown('<div class="ai-card">', unsafe_allow_html=True)
                    st.markdown("### 🧠 Tuotuo's AI 智能洞察")
                    st.markdown(ai_response)
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("🤖 AI 服务暂不可用，请检查 API Key")
        
        # 数据明细（分页）
        st.markdown("---")
        st.markdown(f"### 📋 数据明细（共 {len(display_df)} 条）")
        
        rows_per_page = 10
        total_pages = (len(display_df) + rows_per_page - 1) // rows_per_page
        page_key = "data_page"
        if page_key not in st.session_state:
            st.session_state[page_key] = 1
        
        current_page = st.session_state[page_key]
        start_idx = (current_page - 1) * rows_per_page
        end_idx = min(start_idx + rows_per_page, len(display_df))
        
        st.dataframe(display_df.iloc[start_idx:end_idx], use_container_width=True)
        
        if total_pages > 1:
            col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
            with col1:
                if st.button("⏮️ 首页"):
                    st.session_state[page_key] = 1
                    st.rerun()
            with col2:
                if st.button("◀ 上一页"):
                    if st.session_state[page_key] > 1:
                        st.session_state[page_key] -= 1
                        st.rerun()
            with col3:
                st.markdown(f"<div style='text-align:center;'>第 {current_page} / {total_pages} 页</div>", unsafe_allow_html=True)
            with col4:
                if st.button("下一页 ▶"):
                    if st.session_state[page_key] < total_pages:
                        st.session_state[page_key] += 1
                        st.rerun()
            with col5:
                if st.button("⏭️ 末页"):
                    st.session_state[page_key] = total_pages
                    st.rerun()
    
    elif analyze_btn and not query:
        st.warning("💡 请输入一个问题～")
    
    elif analyze_btn and not value_col:
        st.warning("⚠️ 请先选择数值字段")
    
    # 显示上次结果
    elif st.session_state.has_result and not analyze_btn:
        st.markdown("---")
        st.markdown("### 📈 整体指标")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("总行数", st.session_state.total_rows)
        with col2:
            st.metric("总列数", st.session_state.total_columns)
        with col3:
            numeric_count = len(st.session_state.current_df.select_dtypes(include=[np.number]).columns) if st.session_state.current_df is not None else 0
            st.metric("数值列", numeric_count)
        
        if st.session_state.group_stats is not None and len(st.session_state.group_stats) > 0:
            group_col = st.session_state.group_col
            value_col = st.session_state.value_col
            st.markdown(f"### 📊 {group_col} 分布（按 {value_col} 求和）")
            st.dataframe(st.session_state.group_stats, use_container_width=True)
            
            fig = px.bar(st.session_state.group_stats, x=group_col, y=value_col, title=f"{group_col} 分布")
            fig.update_traces(marker_color=PRIMARY_BLUE)
            st.plotly_chart(fig, use_container_width=True)
        
        if st.session_state.numeric_stats is not None and len(st.session_state.numeric_stats) > 0:
            st.markdown("### 🔢 所有数值字段统计")
            st.dataframe(st.session_state.numeric_stats, use_container_width=True)
        
        if st.session_state.ai_response:
            st.markdown('<div class="ai-card">', unsafe_allow_html=True)
            st.markdown("### 🧠 Tuotuo's AI 智能洞察")
            st.markdown(st.session_state.ai_response)
            st.markdown('</div>', unsafe_allow_html=True)
        
        if st.session_state.filtered_df is not None:
            st.markdown("---")
            st.markdown(f"### 📋 数据明细（共 {len(st.session_state.filtered_df)} 条）")
            rows_per_page = 10
            total_pages = (len(st.session_state.filtered_df) + rows_per_page - 1) // rows_per_page
            page_key = "data_page"
            if page_key not in st.session_state:
                st.session_state[page_key] = 1
            current_page = st.session_state[page_key]
            start_idx = (current_page - 1) * rows_per_page
            end_idx = min(start_idx + rows_per_page, len(st.session_state.filtered_df))
            st.dataframe(st.session_state.filtered_df.iloc[start_idx:end_idx], use_container_width=True)
            
            if total_pages > 1:
                col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
                with col1:
                    if st.button("⏮️ 首页"):
                        st.session_state[page_key] = 1
                        st.rerun()
                with col2:
                    if st.button("◀ 上一页"):
                        if st.session_state[page_key] > 1:
                            st.session_state[page_key] -= 1
                            st.rerun()
                with col3:
                    st.markdown(f"<div style='text-align:center;'>第 {current_page} / {total_pages} 页</div>", unsafe_allow_html=True)
                with col4:
                    if st.button("下一页 ▶"):
                        if st.session_state[page_key] < total_pages:
                            st.session_state[page_key] += 1
                            st.rerun()
                with col5:
                    if st.button("⏭️ 末页"):
                        st.session_state[page_key] = total_pages
                        st.rerun()

else:
    # 引导页面
    st.markdown("""
    <div class="card" style="text-align: center;">
        <div style="font-size: 20px; margin-bottom: 16px;">🎯 上传你的数据，开始智能分析</div>
        <div style="display: flex; justify-content: center; gap: 24px; flex-wrap: wrap;">
            <div>📊 选择分组字段</div>
            <div>📈 选择数值字段</div>
            <div>🔍 自动筛选</div>
            <div>📉 智能图表</div>
            <div>🎯 AI 决策建议</div>
        </div>
        <div style="margin-top: 20px; font-size: 14px; color: #888;">
            💡 支持任何 Excel/CSV 数据（人事、销售、财务、用户分析...）
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==================== 底部 ====================
st.markdown("---")
st.markdown("""
<div style="text-align: center; font-size: 12px; color: #888;">
    ⚡ Made with ☕ by Tuotuo09 · 用户选择维度 · 工具准确计算 · AI 智能解读<br>
    🔒 数据本地处理 · 只发送统计结果
</div>
""", unsafe_allow_html=True)
