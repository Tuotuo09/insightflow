"""
InsightFlow - 智能数据决策助手（完整版）
作者：Tuotuo09
功能：准确计算 + 智能筛选 + 图表展示 + AI 解读
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
if 'channel_stats' not in st.session_state:
    st.session_state.channel_stats = None
if 'city_stats' not in st.session_state:
    st.session_state.city_stats = None
if 'type_stats' not in st.session_state:
    st.session_state.type_stats = None
if 'retention_stats' not in st.session_state:
    st.session_state.retention_stats = None
if 'ai_response' not in st.session_state:
    st.session_state.ai_response = None
if 'total_users' not in st.session_state:
    st.session_state.total_users = 0
if 'pay_users' not in st.session_state:
    st.session_state.pay_users = 0
if 'pay_rate' not in st.session_state:
    st.session_state.pay_rate = 0
if 'total_amount' not in st.session_state:
    st.session_state.total_amount = 0

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

def detect_filter_from_query(query):
    """从用户问题中提取筛选条件"""
    # 渠道筛选
    channels = ['抖音', '小红书', '微信', '应用商店', '官网']
    for channel in channels:
        if channel in query:
            return "渠道来源", channel
    
    # 城市筛选
    cities = ['一线', '新一线', '二线', '三线']
    for city in cities:
        if city in query:
            return "城市等级", city
    
    # 用户类型筛选
    user_types = ['活跃用户', '新用户', '沉睡用户', '流失用户']
    for utype in user_types:
        if utype in query:
            return "用户类型", utype
    
    # 留存状态筛选
    retention = ['高留存', '中留存', '低留存']
    for r in retention:
        if r in query:
            return "留存状态", r
    
    return None, None

def generate_filtered_chart(df, filter_col, filter_val):
    """根据筛选条件生成对应图表"""
    try:
        if filter_col == "渠道来源" and "累计付费金额" in df.columns:
            channel_pay = df.groupby("渠道来源")["累计付费金额"].sum().reset_index()
            if len(channel_pay) > 0:
                fig = px.bar(channel_pay, x="渠道来源", y="累计付费金额", title=f"📊 {filter_val}渠道付费金额")
                fig.update_traces(marker_color=PRIMARY_BLUE)
                return fig
        
        elif filter_col == "城市等级" and "累计付费金额" in df.columns:
            city_pay = df.groupby("城市等级")["累计付费金额"].sum().reset_index()
            if len(city_pay) > 0:
                fig = px.bar(city_pay, x="城市等级", y="累计付费金额", title=f"🏙️ {filter_val}城市付费金额")
                fig.update_traces(marker_color=PRIMARY_BLUE)
                return fig
        
        elif filter_col == "用户类型" and "累计付费金额" in df.columns:
            type_pay = df.groupby("用户类型")["累计付费金额"].sum().reset_index()
            if len(type_pay) > 0:
                fig = px.pie(type_pay, names="用户类型", values="累计付费金额", title=f"👥 {filter_val}用户付费分布")
                fig.update_traces(marker=dict(colors=px.colors.sequential.Blues_r))
                return fig
        
        elif filter_col == "留存状态" and "留存状态" in df.columns:
            retention_count = df["留存状态"].value_counts().reset_index()
            retention_count.columns = ["留存状态", "用户数"]
            if len(retention_count) > 0:
                fig = px.bar(retention_count, x="留存状态", y="用户数", title=f"💾 {filter_val}留存状态分布")
                fig.update_traces(marker_color=PRIMARY_BLUE)
                return fig
    except:
        pass
    return None

def precompute_stats(df):
    """工具准确计算各项统计"""
    # 整体统计
    total_users = len(df)
    pay_users = len(df[df["是否付费"] == "是"]) if "是否付费" in df.columns else 0
    total_amount = df["累计付费金额"].sum() if "累计付费金额" in df.columns else 0
    pay_rate = round(pay_users / total_users * 100, 1) if total_users > 0 else 0
    
    # 渠道统计
    channel_stats = None
    if "渠道来源" in df.columns and "累计付费金额" in df.columns:
        channel_data = []
        for channel in df["渠道来源"].unique():
            channel_df = df[df["渠道来源"] == channel]
            pay_users_c = len(channel_df[channel_df["是否付费"] == "是"]) if "是否付费" in df.columns else 0
            channel_data.append({
                "渠道": channel,
                "用户数": len(channel_df),
                "付费用户数": pay_users_c,
                "付费率": f"{round(pay_users_c/len(channel_df)*100,1)}%",
                "总付费金额": round(channel_df["累计付费金额"].sum(), 0),
                "人均付费": round(channel_df["累计付费金额"].mean(), 1)
            })
        channel_stats = pd.DataFrame(channel_data).sort_values("总付费金额", ascending=False)
    
    # 城市统计
    city_stats = None
    if "城市等级" in df.columns and "累计付费金额" in df.columns:
        city_data = []
        for city in df["城市等级"].unique():
            city_df = df[df["城市等级"] == city]
            pay_users_c = len(city_df[city_df["是否付费"] == "是"]) if "是否付费" in df.columns else 0
            city_data.append({
                "城市等级": city,
                "用户数": len(city_df),
                "付费用户数": pay_users_c,
                "人均付费": round(city_df["累计付费金额"].mean(), 1)
            })
        city_stats = pd.DataFrame(city_data).sort_values("用户数", ascending=False)
    
    # 用户类型统计
    type_stats = None
    if "用户类型" in df.columns and "累计付费金额" in df.columns:
        type_data = []
        for utype in df["用户类型"].unique():
            type_df = df[df["用户类型"] == utype]
            pay_users_c = len(type_df[type_df["是否付费"] == "是"]) if "是否付费" in df.columns else 0
            type_data.append({
                "用户类型": utype,
                "用户数": len(type_df),
                "付费用户数": pay_users_c,
                "人均付费": round(type_df["累计付费金额"].mean(), 1)
            })
        type_stats = pd.DataFrame(type_data).sort_values("用户数", ascending=False)
    
    # 留存统计
    retention_stats = None
    if "留存状态" in df.columns:
        retention_data = []
        for status in df["留存状态"].unique():
            status_df = df[df["留存状态"] == status]
            pay_users_c = len(status_df[status_df["是否付费"] == "是"]) if "是否付费" in df.columns else 0
            retention_data.append({
                "留存状态": status,
                "用户数": len(status_df),
                "付费用户数": pay_users_c,
                "付费率": round(pay_users_c/len(status_df)*100, 1)
            })
        retention_stats = pd.DataFrame(retention_data).sort_values("用户数", ascending=False)
    
    return {
        "total_users": total_users,
        "pay_users": pay_users,
        "pay_rate": pay_rate,
        "total_amount": total_amount,
        "channel_stats": channel_stats,
        "city_stats": city_stats,
        "type_stats": type_stats,
        "retention_stats": retention_stats
    }

def generate_ai_insight(query, stats):
    """生成 AI 洞察"""
    summary = f"用户问题：{query}\n\n"
    summary += f"【整体情况】总用户{stats['total_users']}人，付费{stats['pay_users']}人，付费率{stats['pay_rate']}%，总金额{stats['total_amount']:.0f}元\n\n"
    
    if stats['channel_stats'] is not None:
        summary += "【渠道分析】\n"
        for _, row in stats['channel_stats'].iterrows():
            summary += f"- {row['渠道']}: {row['用户数']}人, 付费{row['付费用户数']}人, 总金额{row['总付费金额']:.0f}元\n"
    
    prompt = f"""基于以下准确数据，给出洞察和建议：

{summary}

请按格式输出：
【洞察】：（2-3点）
【建议】：（分优先级）
【趣味发现】：（一个有趣的发现）"""
    
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
        🤖 Built by Tuotuo09 · 准确计算 · 智能筛选 · AI 解读
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
    
    # 问答区域
    st.markdown("---")
    query = st.text_input("", placeholder="例如：各渠道付费情况｜抖音｜一线城市｜给我一些建议", label_visibility="collapsed")
    analyze_btn = st.button("🚀 开始分析", type="primary")
    st.caption("💡 试试：各渠道付费情况 · 抖音 · 一线城市 · 活跃用户")
    
    # 分析逻辑
    if analyze_btn and query:
        # 检测筛选条件
        filter_col, filter_val = detect_filter_from_query(query)
        
        if filter_col and filter_val and filter_col in df.columns:
            display_df = df[df[filter_col] == filter_val]
            st.info(f"🔍 已筛选：{filter_col} = {filter_val}，共 {len(display_df)} 条记录")
        else:
            display_df = df
            filter_col = None
            filter_val = None
        
        # 计算统计
        stats = precompute_stats(display_df)
        
        # 保存到 session_state
        st.session_state.has_result = True
        st.session_state.filtered_df = display_df
        st.session_state.filter_name = filter_val
        st.session_state.filter_col = filter_col
        st.session_state.channel_stats = stats['channel_stats']
        st.session_state.city_stats = stats['city_stats']
        st.session_state.type_stats = stats['type_stats']
        st.session_state.retention_stats = stats['retention_stats']
        st.session_state.total_users = stats['total_users']
        st.session_state.pay_users = stats['pay_users']
        st.session_state.pay_rate = stats['pay_rate']
        st.session_state.total_amount = stats['total_amount']
        
        # 显示整体指标
        st.markdown("---")
        st.markdown("### 📈 整体指标")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("总用户数", stats['total_users'])
        with col2:
            st.metric("付费用户数", stats['pay_users'])
        with col3:
            st.metric("付费率", f"{stats['pay_rate']}%")
        with col4:
            st.metric("总付费金额", f"{stats['total_amount']:.0f}元")
        
        # 显示筛选图表
        if filter_col and filter_val:
            filter_fig = generate_filtered_chart(display_df, filter_col, filter_val)
            if filter_fig:
                st.plotly_chart(filter_fig, use_container_width=True)
        
        # 显示统计表格
        if stats['channel_stats'] is not None and len(stats['channel_stats']) > 0:
            st.markdown("### 📊 渠道付费分析")
            st.dataframe(stats['channel_stats'], use_container_width=True)
        
        col_left, col_right = st.columns(2)
        with col_left:
            if stats['city_stats'] is not None and len(stats['city_stats']) > 0:
                st.markdown("### 🏙️ 城市等级分析")
                st.dataframe(stats['city_stats'], use_container_width=True)
        with col_right:
            if stats['type_stats'] is not None and len(stats['type_stats']) > 0:
                st.markdown("### 👥 用户类型分析")
                st.dataframe(stats['type_stats'], use_container_width=True)
        
        if stats['retention_stats'] is not None and len(stats['retention_stats']) > 0:
            st.markdown("### 💾 留存状态分析")
            st.dataframe(stats['retention_stats'], use_container_width=True)
        
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
    
    # 显示上次结果
    elif st.session_state.has_result and not analyze_btn:
        st.markdown("---")
        st.markdown("### 📈 整体指标")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("总用户数", st.session_state.total_users)
        with col2:
            st.metric("付费用户数", st.session_state.pay_users)
        with col3:
            st.metric("付费率", f"{st.session_state.pay_rate}%")
        with col4:
            st.metric("总付费金额", f"{st.session_state.total_amount:.0f}元")
        
        if st.session_state.filter_col and st.session_state.filtered_df is not None:
            filter_fig = generate_filtered_chart(st.session_state.filtered_df, st.session_state.filter_col, st.session_state.filter_name)
            if filter_fig:
                st.plotly_chart(filter_fig, use_container_width=True)
        
        if st.session_state.channel_stats is not None and len(st.session_state.channel_stats) > 0:
            st.markdown("### 📊 渠道付费分析")
            st.dataframe(st.session_state.channel_stats, use_container_width=True)
        
        col_left, col_right = st.columns(2)
        with col_left:
            if st.session_state.city_stats is not None and len(st.session_state.city_stats) > 0:
                st.markdown("### 🏙️ 城市等级分析")
                st.dataframe(st.session_state.city_stats, use_container_width=True)
        with col_right:
            if st.session_state.type_stats is not None and len(st.session_state.type_stats) > 0:
                st.markdown("### 👥 用户类型分析")
                st.dataframe(st.session_state.type_stats, use_container_width=True)
        
        if st.session_state.retention_stats is not None and len(st.session_state.retention_stats) > 0:
            st.markdown("### 💾 留存状态分析")
            st.dataframe(st.session_state.retention_stats, use_container_width=True)
        
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
            <div>📊 渠道付费分析</div>
            <div>🏙️ 城市等级分析</div>
            <div>👥 用户类型分析</div>
            <div>💾 留存状态分析</div>
            <div>🎯 AI 决策建议</div>
        </div>
        <div style="margin-top: 20px; font-size: 14px; color: #888;">
            💡 试试：「各渠道付费情况」「抖音」「一线城市」「给我一些建议」
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==================== 底部 ====================
st.markdown("---")
st.markdown("""
<div style="text-align: center; font-size: 12px; color: #888;">
    ⚡ Made with ☕ by Tuotuo09 · 准确计算 · 智能筛选 · AI 解读<br>
    🔒 数据本地处理 · 只发送统计结果
</div>
""", unsafe_allow_html=True)
