"""
InsightFlow - 智能数据决策助手（修复版）
作者：Tuotuo09
核心改进：工具先准确计算，AI 只负责解读，确保数据准确性
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
if 'channel_stats' not in st.session_state:
    st.session_state.channel_stats = None
if 'city_stats' not in st.session_state:
    st.session_state.city_stats = None
if 'type_stats' not in st.session_state:
    st.session_state.type_stats = None
if 'ai_response' not in st.session_state:
    st.session_state.ai_response = None

# ==================== DeepSeek API 配置 ====================
# 方式1：直接填入
DEEPSEEK_API_KEY = "sk-52bcbd3d232945828250c3a1408598ff"
# 方式2：从环境变量读取（部署时用）
# import os
# DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

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
            {"role": "system", "content": "你是一个专业的数据分析专家。基于提供的准确数据，给出深刻的洞察和可执行的建议。回答要简洁、专业、有决策价值。"},
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
    """清洗数据框：过滤 Unnamed 列，处理空值"""
    valid_columns = [c for c in df.columns if not str(c).startswith('Unnamed')]
    df = df[valid_columns]
    
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna('')
    
    return df

def precompute_channel_stats(df):
    """【工具准确计算】渠道付费统计"""
    
    # 自动识别列名
    channel_col = None
    pay_col = None
    amount_col = None
    
    for col in df.columns:
        if "渠道" in col or "来源" in col:
            channel_col = col
        if "付费" in col and "金额" in col:
            amount_col = col
        if "付费" in col and "是否" in col:
            pay_col = col
    
    if not channel_col:
        return None
    
    stats = []
    for channel in df[channel_col].unique():
        channel_df = df[df[channel_col] == channel]
        total_users = len(channel_df)
        
        # 付费用户数
        if pay_col:
            pay_users = len(channel_df[channel_df[pay_col] == "是"])
        else:
            pay_users = 0
        
        pay_rate = round(pay_users / total_users * 100, 1) if total_users > 0 else 0
        
        # 付费金额
        if amount_col:
            total_amount = channel_df[amount_col].sum()
            avg_amount = round(total_amount / pay_users, 1) if pay_users > 0 else 0
        else:
            total_amount = 0
            avg_amount = 0
        
        stats.append({
            "渠道": channel,
            "用户数": total_users,
            "付费用户数": pay_users,
            "付费率": f"{pay_rate}%",
            "总付费金额": round(total_amount, 0),
            "人均付费": avg_amount
        })
    
    result_df = pd.DataFrame(stats)
    result_df = result_df.sort_values("总付费金额", ascending=False)
    
    return result_df

def precompute_demographic_stats(df):
    """【工具准确计算】用户画像统计"""
    
    # 城市等级统计
    city_col = None
    for col in df.columns:
        if "城市" in col:
            city_col = col
            break
    
    city_stats = None
    if city_col:
        city_data = []
        for city in df[city_col].unique():
            city_df = df[df[city_col] == city]
            pay_users = len(city_df[city_df["是否付费"] == "是"]) if "是否付费" in df.columns else 0
            avg_amount = round(city_df["累计付费金额"].mean(), 1) if "累计付费金额" in df.columns else 0
            city_data.append({
                "城市等级": city,
                "用户数": len(city_df),
                "付费用户数": pay_users,
                "人均付费": avg_amount
            })
        city_stats = pd.DataFrame(city_data).sort_values("用户数", ascending=False)
    
    # 用户类型统计
    type_col = None
    for col in df.columns:
        if "用户类型" in col:
            type_col = col
            break
    
    type_stats = None
    if type_col:
        type_data = []
        for utype in df[type_col].unique():
            type_df = df[df[type_col] == utype]
            pay_users = len(type_df[type_df["是否付费"] == "是"]) if "是否付费" in df.columns else 0
            avg_amount = round(type_df["累计付费金额"].mean(), 1) if "累计付费金额" in df.columns else 0
            type_data.append({
                "用户类型": utype,
                "用户数": len(type_df),
                "付费用户数": pay_users,
                "人均付费": avg_amount
            })
        type_stats = pd.DataFrame(type_data).sort_values("用户数", ascending=False)
    
    return city_stats, type_stats

def precompute_retention_stats(df):
    """【工具准确计算】留存状态统计"""
    
    retention_col = None
    for col in df.columns:
        if "留存" in col:
            retention_col = col
            break
    
    if not retention_col:
        return None
    
    retention_data = []
    for status in df[retention_col].unique():
        status_df = df[df[retention_col] == status]
        pay_users = len(status_df[status_df["是否付费"] == "是"]) if "是否付费" in df.columns else 0
        retention_data.append({
            "留存状态": status,
            "用户数": len(status_df),
            "付费用户数": pay_users,
            "付费率": round(pay_users / len(status_df) * 100, 1)
        })
    
    return pd.DataFrame(retention_data).sort_values("用户数", ascending=False)

def ai_analyze_with_precomputed(query, df):
    """AI 解读模式：工具先计算，AI 只解读"""
    
    # 1. 工具准确计算所有统计
    channel_stats = precompute_channel_stats(df)
    city_stats, type_stats = precompute_demographic_stats(df)
    retention_stats = precompute_retention_stats(df)
    
    # 2. 整体统计
    total_users = len(df)
    pay_users = len(df[df["是否付费"] == "是"]) if "是否付费" in df.columns else 0
    total_amount = df["累计付费金额"].sum() if "累计付费金额" in df.columns else 0
    
    # 3. 构建统计摘要（只发送计算结果给 AI）
    summary_text = "📊 【渠道付费分析】\n"
    if channel_stats is not None:
        for _, row in channel_stats.iterrows():
            summary_text += f"  • {row['渠道']}: {row['用户数']}人, 付费{row['付费用户数']}人, 付费率{row['付费率']}, 总金额{row['总付费金额']:.0f}元, 人均{row['人均付费']}元\n"
    else:
        summary_text += "  • 无渠道数据\n"
    
    summary_text += "\n🏙️ 【城市等级分析】\n"
    if city_stats is not None:
        for _, row in city_stats.iterrows():
            summary_text += f"  • {row['城市等级']}: {row['用户数']}人, 付费{row['付费用户数']}人, 人均{row['人均付费']}元\n"
    else:
        summary_text += "  • 无城市数据\n"
    
    summary_text += "\n👥 【用户类型分析】\n"
    if type_stats is not None:
        for _, row in type_stats.iterrows():
            summary_text += f"  • {row['用户类型']}: {row['用户数']}人, 付费{row['付费用户数']}人, 人均{row['人均付费']}元\n"
    else:
        summary_text += "  • 无用户类型数据\n"
    
    summary_text += "\n💾 【留存状态分析】\n"
    if retention_stats is not None:
        for _, row in retention_stats.iterrows():
            summary_text += f"  • {row['留存状态']}: {row['用户数']}人, 付费{row['付费用户数']}人, 付费率{row['付费率']}%\n"
    else:
        summary_text += "  • 无留存数据\n"
    
    summary_text += f"\n📈 【整体情况】\n"
    summary_text += f"  • 总用户数: {total_users}人\n"
    summary_text += f"  • 付费用户数: {pay_users}人\n"
    summary_text += f"  • 付费率: {pay_users/total_users*100:.1f}%\n"
    summary_text += f"  • 总付费金额: {total_amount:.0f}元\n"
    summary_text += f"  • 人均付费: {total_amount/total_users:.1f}元\n"
    if pay_users > 0:
        summary_text += f"  • 付费用户人均: {total_amount/pay_users:.1f}元\n"
    
    # 4. 发送给 AI 解读
    prompt = f"""你是数据分析专家。基于以下**准确计算**的数据，回答用户问题：{query}

{summary_text}

请按以下格式输出：
【洞察】：（2-3点，基于数据发现的核心问题）
【决策建议】：（分优先级：高/中/低，具体可执行）
【趣味发现】：（一个有趣的数据洞察）

注意：数据已经计算准确，请基于数据解读。"""
    
    response = call_deepseek(prompt)
    
    return {
        "summary": summary_text,
        "response": response,
        "channel_stats": channel_stats,
        "city_stats": city_stats,
        "type_stats": type_stats,
        "retention_stats": retention_stats,
        "total_users": total_users,
        "pay_users": pay_users,
        "pay_rate": pay_users/total_users*100 if total_users > 0 else 0,
        "total_amount": total_amount
    }

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
        🤖 Built by Tuotuo09 · 准确计算 · AI 智能解读
    </p>
    <p style="font-size: 14px; color: #888;">
        💡 随便问 · 工具准确计算 · AI 给出洞察和建议
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
    # 加载数据并清洗
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    df = clean_dataframe(df)
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
    
    query = st.text_input("", placeholder="例如：各渠道付费情况｜哪个渠道用户质量最高｜给我一些建议", label_visibility="collapsed")
    analyze_btn = st.button("🚀 开始分析", type="primary")
    
    # 动态示例问题
    st.caption("💡 试试这些：各渠道付费情况 · 城市等级分析 · 用户类型对比 · 留存状态分析")
    
    # ==================== 分析逻辑 ====================
    if analyze_btn and query:
        api_ok = check_api_availability()
        
        if api_ok:
            st.markdown('<div><span class="mode-badge-ai">🤖 AI 智能模式 · 工具准确计算 + AI 智能解读</span></div>', unsafe_allow_html=True)
            
            loading_msg = random.choice(LOADING_MESSAGES)
            with st.spinner(loading_msg):
                result = ai_analyze_with_precomputed(query, df)
                
                if result:
                    # 保存到 session_state
                    st.session_state.has_result = True
                    st.session_state.channel_stats = result["channel_stats"]
                    st.session_state.city_stats = result["city_stats"]
                    st.session_state.type_stats = result["type_stats"]
                    st.session_state.retention_stats = result["retention_stats"]
                    st.session_state.ai_response = result["response"]
                    st.session_state.total_users = result["total_users"]
                    st.session_state.pay_users = result["pay_users"]
                    st.session_state.pay_rate = result["pay_rate"]
                    st.session_state.total_amount = result["total_amount"]
                    
                    # 显示整体指标卡片
                    st.markdown("---")
                    st.markdown("### 📈 整体指标")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("总用户数", f"{result['total_users']}")
                    with col2:
                        st.metric("付费用户数", f"{result['pay_users']}")
                    with col3:
                        st.metric("付费率", f"{result['pay_rate']:.1f}%")
                    with col4:
                        st.metric("总付费金额", f"{result['total_amount']:.0f}元")
                    
                    # 显示渠道统计表格
                    if result["channel_stats"] is not None:
                        st.markdown("### 📊 渠道付费分析")
                        st.dataframe(result["channel_stats"], use_container_width=True)
                    
                    # 城市和用户类型并排
                    col_left, col_right = st.columns(2)
                    with col_left:
                        if result["city_stats"] is not None:
                            st.markdown("### 🏙️ 城市等级分析")
                            st.dataframe(result["city_stats"], use_container_width=True)
                    with col_right:
                        if result["type_stats"] is not None:
                            st.markdown("### 👥 用户类型分析")
                            st.dataframe(result["type_stats"], use_container_width=True)
                    
                    # 留存状态
                    if result["retention_stats"] is not None:
                        st.markdown("### 💾 留存状态分析")
                        st.dataframe(result["retention_stats"], use_container_width=True)
                    
                    # AI 智能洞察
                    if result["response"]:
                        st.markdown('<div class="ai-card">', unsafe_allow_html=True)
                        st.markdown("### 🧠 Tuotuo's AI 智能洞察")
                        
                        response = result["response"]
                        
                        if "【洞察】" in response:
                            insight_part = response.split("【洞察】")[1].split("【决策建议】")[0] if "【决策建议】" in response else response.split("【洞察】")[1]
                            st.markdown(f"**🔍 洞察**\n{insight_part.strip()}")
                        
                        if "【决策建议】" in response:
                            rec_part = response.split("【决策建议】")[1].split("【趣味发现】")[0] if "【趣味发现】" in response else response.split("【决策建议】")[1]
                            st.markdown(f"\n**🎯 建议**\n{rec_part.strip()}")
                        
                        if "【趣味发现】" in response:
                            fun_part = response.split("【趣味发现】")[1]
                            st.markdown(f"\n**📌 趣味发现**\n{fun_part.strip()}")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # 显示原始数据表格（分页）
                    st.markdown("---")
                    st.markdown(f"### 📋 数据明细（共 {len(df)} 条记录）")
                    
                    # 分页显示
                    rows_per_page = 10
                    total_pages = (len(df) + rows_per_page - 1) // rows_per_page
                    
                    page_key = "data_page"
                    if page_key not in st.session_state:
                        st.session_state[page_key] = 1
                    
                    current_page = st.session_state[page_key]
                    start_idx = (current_page - 1) * rows_per_page
                    end_idx = min(start_idx + rows_per_page, len(df))
                    
                    page_df = df.iloc[start_idx:end_idx]
                    st.dataframe(page_df, use_container_width=True)
                    
                    # 分页控件
                    if total_pages > 1:
                        col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
                        with col1:
                            if st.button("⏮️ 首页", key="first"):
                                st.session_state[page_key] = 1
                                st.rerun()
                        with col2:
                            if st.button("◀ 上一页", key="prev"):
                                if st.session_state[page_key] > 1:
                                    st.session_state[page_key] -= 1
                                    st.rerun()
                        with col3:
                            st.markdown(f"<div style='text-align: center;'>第 {current_page} / {total_pages} 页</div>", unsafe_allow_html=True)
                        with col4:
                            if st.button("下一页 ▶", key="next"):
                                if st.session_state[page_key] < total_pages:
                                    st.session_state[page_key] += 1
                                    st.rerun()
                        with col5:
                            if st.button("末页 ⏭️", key="last"):
                                st.session_state[page_key] = total_pages
                                st.rerun()
                    
                else:
                    st.warning("🤖 AI 服务繁忙，请稍后再试")
        
        else:
            st.warning("🤖 AI 服务暂不可用，请检查 API Key 配置")
    
    elif analyze_btn and not query:
        st.warning("💡 请输入一个问题～")
    
    # ==================== 显示上次结果（保持状态）====================
    elif st.session_state.has_result and not analyze_btn:
        st.markdown("---")
        st.markdown("### 📈 整体指标")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("总用户数", f"{st.session_state.total_users}")
        with col2:
            st.metric("付费用户数", f"{st.session_state.pay_users}")
        with col3:
            st.metric("付费率", f"{st.session_state.pay_rate:.1f}%")
        with col4:
            st.metric("总付费金额", f"{st.session_state.total_amount:.0f}元")
        
        if st.session_state.channel_stats is not None:
            st.markdown("### 📊 渠道付费分析")
            st.dataframe(st.session_state.channel_stats, use_container_width=True)
        
        col_left, col_right = st.columns(2)
        with col_left:
            if st.session_state.city_stats is not None:
                st.markdown("### 🏙️ 城市等级分析")
                st.dataframe(st.session_state.city_stats, use_container_width=True)
        with col_right:
            if st.session_state.type_stats is not None:
                st.markdown("### 👥 用户类型分析")
                st.dataframe(st.session_state.type_stats, use_container_width=True)
        
        if st.session_state.retention_stats is not None:
            st.markdown("### 💾 留存状态分析")
            st.dataframe(st.session_state.retention_stats, use_container_width=True)
        
        if st.session_state.ai_response:
            st.markdown('<div class="ai-card">', unsafe_allow_html=True)
            st.markdown("### 🧠 Tuotuo's AI 智能洞察")
            
            response = st.session_state.ai_response
            
            if "【洞察】" in response:
                insight_part = response.split("【洞察】")[1].split("【决策建议】")[0] if "【决策建议】" in response else response.split("【洞察】")[1]
                st.markdown(f"**🔍 洞察**\n{insight_part.strip()}")
            
            if "【决策建议】" in response:
                rec_part = response.split("【决策建议】")[1].split("【趣味发现】")[0] if "【趣味发现】" in response else response.split("【决策建议】")[1]
                st.markdown(f"\n**🎯 建议**\n{rec_part.strip()}")
            
            if "【趣味发现】" in response:
                fun_part = response.split("【趣味发现】")[1]
                st.markdown(f"\n**📌 趣味发现**\n{fun_part.strip()}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # 显示原始数据
        st.markdown("---")
        st.markdown(f"### 📋 数据明细（共 {len(st.session_state.current_df)} 条记录）")
        
        df = st.session_state.current_df
        rows_per_page = 10
        total_pages = (len(df) + rows_per_page - 1) // rows_per_page
        
        page_key = "data_page"
        if page_key not in st.session_state:
            st.session_state[page_key] = 1
        
        current_page = st.session_state[page_key]
        start_idx = (current_page - 1) * rows_per_page
        end_idx = min(start_idx + rows_per_page, len(df))
        
        page_df = df.iloc[start_idx:end_idx]
        st.dataframe(page_df, use_container_width=True)
        
        if total_pages > 1:
            col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
            with col1:
                if st.button("⏮️ 首页", key="first_result"):
                    st.session_state[page_key] = 1
                    st.rerun()
            with col2:
                if st.button("◀ 上一页", key="prev_result"):
                    if st.session_state[page_key] > 1:
                        st.session_state[page_key] -= 1
                        st.rerun()
            with col3:
                st.markdown(f"<div style='text-align: center;'>第 {current_page} / {total_pages} 页</div>", unsafe_allow_html=True)
            with col4:
                if st.button("下一页 ▶", key="next_result"):
                    if st.session_state[page_key] < total_pages:
                        st.session_state[page_key] += 1
                        st.rerun()
            with col5:
                if st.button("末页 ⏭️", key="last_result"):
                    st.session_state[page_key] = total_pages
                    st.rerun()

else:
    # 引导页面
    st.markdown("""
    <div style="margin-top: 32px;">
        <div class="card" style="text-align: center;">
            <div style="font-size: 20px; margin-bottom: 16px;">🎯 能做什么</div>
            <div style="display: flex; justify-content: center; gap: 32px; flex-wrap: wrap;">
                <div>📊 渠道付费分析</div>
                <div>🏙️ 城市等级分析</div>
                <div>👥 用户类型分析</div>
                <div>💾 留存状态分析</div>
                <div>🎯 AI 决策建议</div>
            </div>
        </div>
    </div>
    
    <div style="margin-top: 24px; text-align: center; font-size: 14px; color: #888;">
        <p>💡 上传你的用户数据，问：「各渠道付费情况」「给我一些建议」</p>
        <p>🤖 工具准确计算，AI 基于正确数据给出洞察和建议</p>
    </div>
    """, unsafe_allow_html=True)

# ==================== 底部 ====================
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 16px; font-size: 12px; color: #888;">
    ⚡ Made with ☕ by Tuotuo09 · 工具准确计算 · AI 智能解读<br>
    🔒 数据本地处理 · 只发送统计结果给 AI · 隐私安全
</div>
""", unsafe_allow_html=True)
