"""
InsightFlow - AI 智能数据决策助手（筛选修复版）
作者：Tuotuo09
修复：筛选功能、销售额汇总、占比计算
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import re
import json
import requests

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="InsightFlow - AI Data Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== 科技蓝配色 ====================
PRIMARY_BLUE = "#1E88E5"
DARK_BLUE = "#0A66C2"
LIGHT_BLUE = "#E3F2FD"
BG_GRAY = "#F8F9FA"
WARNING_ORANGE = "#FF9800"
SUCCESS_GREEN = "#4CAF50"

# ==================== 自定义 CSS ====================
st.markdown(f"""
<style>
    .stApp {{ background-color: {BG_GRAY}; }}
    .card {{
        background-color: white;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
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
    .upload-area {{
        border: 2px dashed {PRIMARY_BLUE};
        border-radius: 16px;
        padding: 48px;
        text-align: center;
        background-color: white;
        transition: all 0.3s ease;
    }}
    .upload-area:hover {{
        border-color: {DARK_BLUE};
        background-color: {LIGHT_BLUE};
    }}
    .stButton > button {{
        background: linear-gradient(135deg, {PRIMARY_BLUE}, {DARK_BLUE});
        color: white;
        border: none;
        border-radius: 40px;
        padding: 12px 32px;
        font-weight: 600;
        font-size: 16px;
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
    .metric-card {{
        background: linear-gradient(135deg, white, {LIGHT_BLUE});
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        border: 1px solid #E5E7EB;
    }}
    .metric-value {{
        font-size: 28px;
        font-weight: 700;
        color: {PRIMARY_BLUE};
    }}
    .ai-card {{
        background: linear-gradient(135deg, {LIGHT_BLUE}, white);
        border-radius: 16px;
        padding: 24px;
        border-left: 4px solid {PRIMARY_BLUE};
    }}
    .filter-badge {{
        background-color: {PRIMARY_BLUE};
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        display: inline-block;
        margin-right: 8px;
    }}
    hr {{ margin: 24px 0; border-color: #E5E7EB; }}
    .mode-badge-ai {{
        background-color: {SUCCESS_GREEN};
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        display: inline-block;
        margin-bottom: 16px;
    }}
    .mode-badge-fallback {{
        background-color: {WARNING_ORANGE};
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        display: inline-block;
        margin-bottom: 16px;
    }}
</style>
""", unsafe_allow_html=True)

# ==================== 标题区域 ====================
st.markdown("""
<div style="text-align: center; margin-bottom: 32px;">
    <h1 class="title">✨ InsightFlow</h1>
    <p style="font-size: 18px; color: #666; margin-top: -8px;">
        🤖 Built by Tuotuo09 · AI Data Intelligence Platform
    </p>
    <p style="font-size: 14px; color: #888;">
        💡 随便问 · 支持筛选 · 精确计算 · 决策建议
    </p>
</div>
""", unsafe_allow_html=True)

# ==================== 会话状态初始化 ====================
if 'api_available' not in st.session_state:
    st.session_state.api_available = None
if 'api_checked' not in st.session_state:
    st.session_state.api_checked = False

# ==================== DeepSeek API 配置 ====================
DEEPSEEK_API_KEY = st.secrets.get("DEEPSEEK_API_KEY", "")

def check_api_availability():
    """检查 API 是否可用"""
    if st.session_state.api_checked:
        return st.session_state.api_available
    
    if not DEEPSEEK_API_KEY:
        st.session_state.api_available = False
        st.session_state.api_checked = True
        return False
    
    try:
        headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
        data = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1
        }
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=5
        )
        if response.status_code == 200:
            st.session_state.api_available = True
        else:
            st.session_state.api_available = False
        st.session_state.api_checked = True
        return st.session_state.api_available
    except:
        st.session_state.api_available = False
        st.session_state.api_checked = True
        return False

def call_deepseek(prompt):
    """调用 DeepSeek API"""
    if not DEEPSEEK_API_KEY:
        return None
    
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是一个专业的数据分析专家。你需要理解用户的数据分析意图，返回JSON格式的分析需求。不要自己计算数据，只告诉我需要分析什么。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 800
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
        else:
            return None
    except:
        return None

def detect_filter_rule_based(query, df):
    """规则匹配：检测筛选条件"""
    # 品类关键词（电商数据）
    categories = ['连衣裙', '上衣', '裤装', 'T恤', '卫衣', '牛仔裤', '西装裤', '阔腿裤']
    for cat in categories:
        if cat in query:
            # 找到品类字段
            for col in df.columns:
                if '品类' in col or '类别' in col or 'category' in col.lower():
                    return col, cat
    
    # 部门关键词（人事数据）
    departments = ['技术部', '产品部', '市场部', '销售部', '人力资源部', '财务部', '运营部']
    for dept in departments:
        if dept in query:
            for col in df.columns:
                if '部门' in col or 'dept' in col.lower():
                    return col, dept
    
    # 城市关键词
    cities = ['北京', '上海', '广州', '深圳', '杭州', '成都', '武汉', '南京', '苏州', '重庆']
    for city in cities:
        if city in query:
            for col in df.columns:
                if '城市' in col or 'city' in col.lower() or '地区' in col:
                    return col, city
    
    return None, None

def detect_value_field(df):
    """智能检测应该使用哪个数值字段"""
    sales_keywords = ['销售额', '销售金额', '成交额', '金额', 'sales', 'revenue']
    for col in df.columns:
        col_lower = col.lower()
        for kw in sales_keywords:
            if kw in col_lower:
                return col
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    id_keywords = ['id', '编号', '工号', '序号']
    for col in numeric_cols:
        if not any(kw in col.lower() for kw in id_keywords):
            return col
    
    return numeric_cols[0] if numeric_cols else None

def detect_group_field(df):
    """智能检测应该使用哪个分组字段"""
    group_keywords = ['品类', '类别', '分类', '部门', '地区', '城市', '类型', 'category', 'type']
    for col in df.columns:
        col_lower = col.lower()
        for kw in group_keywords:
            if kw in col_lower:
                return col
    
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    return text_cols[0] if text_cols else None

def execute_analysis_by_intent(df, intent_result, query):
    """根据 AI 返回的意图执行精确计算（支持筛选）"""
    
    # 1. 获取筛选条件
    filter_field = intent_result.get("filter_field")
    filter_value = intent_result.get("filter_value")
    
    # 如果 AI 没识别到筛选，用规则匹配兜底
    if not filter_field:
        filter_field, filter_value = detect_filter_rule_based(query, df)
    
    # 2. 应用筛选
    if filter_field and filter_value and filter_field in df.columns:
        df_filtered = df[df[filter_field] == filter_value]
        filter_msg = f"🔍 已筛选：{filter_field} = {filter_value}，共 {len(df_filtered)} 条记录"
        st.info(filter_msg)
    else:
        df_filtered = df
        filter_msg = None
    
    # 3. 获取分析参数
    intent_type = intent_result.get("intent_type", "distribution")
    group_by = intent_result.get("group_by_field")
    value_field = intent_result.get("value_field")
    top_n = intent_result.get("top_n", 5)
    
    # 自动检测缺失字段
    if not value_field:
        value_field = detect_value_field(df_filtered)
    if not group_by and intent_type in ["distribution", "comparison"]:
        group_by = detect_group_field(df_filtered)
    
    result_df = None
    chart_type = "bar"
    summary = ""
    
    try:
        if intent_type == "distribution" and group_by and value_field:
            # 分组汇总
            result_df = df_filtered.groupby(group_by)[value_field].sum().reset_index()
            result_df.columns = [group_by, value_field]
            result_df = result_df.sort_values(value_field, ascending=False)
            chart_type = "pie" if len(result_df) <= 8 else "bar"
            total = result_df[value_field].sum()
            if total > 0:
                top_item = result_df.iloc[0][group_by]
                top_value = result_df.iloc[0][value_field]
                summary = f"{group_by}分布：{top_item}最高，{top_value:,.0f}元，占比{top_value/total*100:.1f}%"
            else:
                summary = f"{group_by}分布（无销售额数据）"
        
        elif intent_type == "ranking" and value_field:
            # 排名
            result_df = df_filtered.nlargest(top_n, value_field)
            chart_type = "bar"
            summary = f"{value_field}最高的{top_n}名"
        
        elif intent_type == "trend":
            # 趋势
            date_col = None
            for col in df_filtered.columns:
                if '日期' in col or 'date' in col.lower():
                    date_col = col
                    break
            if date_col and value_field:
                df_temp = df_filtered.copy()
                df_temp[date_col] = pd.to_datetime(df_temp[date_col])
                df_temp['日期'] = df_temp[date_col].dt.date
                result_df = df_temp.groupby('日期')[value_field].sum().reset_index()
                chart_type = "line"
                summary = f"{value_field}趋势"
        
        elif intent_type == "summary":
            # 汇总统计
            total = df_filtered[value_field].sum() if value_field else len(df_filtered)
            avg = df_filtered[value_field].mean() if value_field else None
            result_df = pd.DataFrame({
                "指标": ["记录数", f"总{value_field}" if value_field else "总数", f"平均{value_field}" if value_field else "平均值"],
                "数值": [len(df_filtered), f"{total:,.0f}" if value_field else total, f"{avg:,.0f}" if avg else "-"]
            })
            chart_type = "none"
            summary = f"共{len(df_filtered)}条，总{value_field}{total:,.0f}元"
        
        else:
            # 默认：显示筛选后的数据
            result_df = df_filtered.head(20)
            chart_type = "none"
            summary = f"筛选结果（共{len(df_filtered)}条）"
        
    except Exception as e:
        st.warning(f"计算时出现小问题: {str(e)[:50]}")
        return None, "none", summary, filter_msg
    
    return result_df, chart_type, summary, filter_msg

def generate_ai_insights(query, df, result_df, intent_result, summary, filter_msg):
    """生成 AI 洞察和决策建议"""
    if result_df is None or len(result_df) == 0:
        return None
    
    # 准备数据摘要
    data_summary = f"用户问题：{query}\n"
    if filter_msg:
        data_summary += f"筛选条件：{filter_msg}\n"
    data_summary += f"分析结果摘要：{summary}\n"
    
    if len(result_df) > 0 and len(result_df.columns) >= 2:
        data_summary += f"数据详情：\n{result_df.head(10).to_string()}\n"
    
    prompt = f"""
你是数据分析专家。基于以下数据，给出洞察和决策建议。

{data_summary}

请按以下格式输出（不要用JSON，用自然语言）：
【洞察】：（数据意味着什么，2-3点）
【决策建议】：（应该怎么做，2-3点，具体可操作）
【趣味事实】：（一个有趣的数据发现）
"""
    
    return call_deepseek(prompt)

# ==================== 降级模式 ====================
def fallback_analyze(query, df):
    """降级模式：规则匹配分析（支持筛选）"""
    query_lower = query.lower()
    value_field = detect_value_field(df)
    group_field = detect_group_field(df)
    
    # 检测筛选条件
    filter_field, filter_value = detect_filter_rule_based(query, df)
    
    if filter_field and filter_value and filter_field in df.columns:
        df_filtered = df[df[filter_field] == filter_value]
        filter_msg = f"🔍 已筛选：{filter_field} = {filter_value}，共 {len(df_filtered)} 条"
    else:
        df_filtered = df
        filter_msg = None
    
    result_df = None
    chart_type = "bar"
    summary = ""
    
    # 分布
    if "分布" in query_lower and group_field and value_field:
        result_df = df_filtered.groupby(group_field)[value_field].sum().reset_index()
        result_df.columns = [group_field, value_field]
        result_df = result_df.sort_values(value_field, ascending=False)
        chart_type = "pie" if len(result_df) <= 8 else "bar"
        total = result_df[value_field].sum()
        if total > 0:
            top = result_df.iloc[0][group_field]
            top_val = result_df.iloc[0][value_field]
            summary = f"{group_field}分布：{top}最高，{top_val:,.0f}元，占比{top_val/total*100:.1f}%"
        else:
            summary = f"{group_field}分布"
    
    # 排名
    elif "前" in query_lower and value_field:
        nums = re.findall(r'\d+', query_lower)
        n = int(nums[0]) if nums else 5
        result_df = df_filtered.nlargest(n, value_field)
        chart_type = "bar"
        summary = f"{value_field}最高的{n}名"
    
    # 平均
    elif "平均" in query_lower and value_field:
        avg_val = df_filtered[value_field].mean()
        summary = f"平均{value_field}：{avg_val:.0f}元"
        result_df = pd.DataFrame([{"指标": f"平均{value_field}", "数值": f"{avg_val:.0f}元"}])
        chart_type = "none"
    
    # 筛选查询（没有其他意图时，显示筛选后的数据）
    elif filter_field and filter_value:
        result_df = df_filtered.head(20)
        chart_type = "none"
        summary = f"{filter_field} = {filter_value} 的数据（共{len(df_filtered)}条）"
    
    # 默认推荐
    else:
        summary = f"试试：{group_field}分布、{value_field}前10、平均{value_field}"
        result_df = pd.DataFrame([{"提示": "支持的问题", "示例": f"{group_field}分布"}])
        chart_type = "none"
    
    return result_df, chart_type, summary, filter_msg

# ==================== 上传区域 ====================
uploaded_file = st.file_uploader(
    "📁 上传数据文件",
    type=['xlsx', 'xls', 'csv'],
    help="支持 Excel 和 CSV 格式"
)

if uploaded_file:
    # 加载数据
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    # 数据预览
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        text_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="metric-card"><div style="font-size:14px;color:#666;">📊 总行数</div><div class="metric-value">{len(df)}</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div style="font-size:14px;color:#666;">📋 总列数</div><div class="metric-value">{len(df.columns)}</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><div style="font-size:14px;color:#666;">🔢 数值列</div><div class="metric-value">{len(numeric_cols)}</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card"><div style="font-size:14px;color:#666;">📝 文本列</div><div class="metric-value">{len(text_cols)}</div></div>', unsafe_allow_html=True)
        
        with st.expander("🔍 查看数据详情"):
            st.dataframe(df.head(100), use_container_width=True)
            st.caption(f"共 {len(df)} 行，显示前 100 行")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 问答区域
    st.markdown("---")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 💬 Ask me anything... (╯°□°）╯︵ ┻━┻")
    st.caption("随便问，AI 理解意图，工具精确计算，支持筛选")
    
    query = st.text_input("", placeholder="例如：各品类销售额分布｜连衣裙｜技术部｜销售额前10", label_visibility="collapsed")
    
    col_btn, col_space = st.columns([1, 5])
    with col_btn:
        analyze_btn = st.button("🚀 Analyze", type="primary", use_container_width=True)
    
    st.markdown("💡 **试试这些**：各品类销售额分布 · 连衣裙 · 销售额前5 · 平均单价")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 分析逻辑
    if analyze_btn and query:
        api_ok = check_api_availability()
        
        if api_ok:
            st.markdown('<div><span class="mode-badge-ai">🤖 AI 智能模式</span></div>', unsafe_allow_html=True)
            
            with st.spinner("AI 正在理解你的问题..."):
                # 准备字段信息
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                text_cols = df.select_dtypes(include=['object']).columns.tolist()
                
                prompt = f"""
用户问题：{query}

数据字段信息：
- 文本列：{text_cols[:10]}
- 数值列：{numeric_cols[:10]}

请理解用户意图，返回JSON格式（只返回JSON）：
{{
    "intent_type": "distribution/ranking/trend/comparison/summary/filter",
    "filter_field": "筛选字段名（如果用户想筛选某个值，如'技术部'、'连衣裙'、'北京'）",
    "filter_value": "筛选值（具体要筛选的值）",
    "group_by_field": "分组字段名（如果是分布/对比类问题）",
    "value_field": "数值字段名（需要分析的字段）",
    "top_n": 5
}}

注意：
- 如果用户问的是「技术部」「连衣裙」这种单一值，intent_type 设为 "filter"
- 如果有销售额相关字段，优先用销售额字段
"""
                
                intent_response = call_deepseek(prompt)
                
                if intent_response:
                    try:
                        intent_result = json.loads(intent_response)
                        st.success("✅ AI 已理解你的问题")
                        
                        # 执行精确计算（带筛选）
                        result_df, chart_type, summary, filter_msg = execute_analysis_by_intent(df, intent_result, query)
                        
                        if result_df is not None:
                            # 显示结果
                            st.markdown("---")
                            st.markdown("### 📊 Analysis Results")
                            st.markdown(f"### {summary}")
                            
                            col_left, col_right = st.columns([3, 2])
                            with col_left:
                                st.dataframe(result_df, use_container_width=True)
                            with col_right:
                                if chart_type != "none" and len(result_df) >= 2:
                                    if chart_type == "pie":
                                        fig = px.pie(result_df, names=result_df.columns[0], values=result_df.columns[1])
                                        fig.update_traces(marker=dict(colors=px.colors.sequential.Blues_r))
                                    elif chart_type == "bar":
                                        fig = px.bar(result_df, x=result_df.columns[0], y=result_df.columns[1])
                                        fig.update_traces(marker_color=PRIMARY_BLUE)
                                    elif chart_type == "line":
                                        fig = px.line(result_df, x=result_df.columns[0], y=result_df.columns[1])
                                        fig.update_traces(line_color=PRIMARY_BLUE)
                                    else:
                                        fig = px.bar(result_df, x=result_df.columns[0], y=result_df.columns[1])
                                    st.plotly_chart(fig, use_container_width=True)
                            
                            # 生成 AI 洞察
                            with st.spinner("AI 正在生成洞察和建议..."):
                                insights = generate_ai_insights(query, df, result_df, intent_result, summary, filter_msg)
                                
                                if insights:
                                    st.markdown("---")
                                    st.markdown('<div class="ai-card">', unsafe_allow_html=True)
                                    st.markdown("### 🧠 Tuotuo's AI Intelligence")
                                    
                                    if "【洞察】" in insights:
                                        parts = insights.split("【洞察】")
                                        if len(parts) > 1:
                                            insight_part = parts[1].split("【决策建议】")[0] if "【决策建议】" in parts[1] else parts[1]
                                            st.markdown(f"🔍 **洞察**：{insight_part.strip()}")
                                        
                                        if "【决策建议】" in insights:
                                            rec_part = insights.split("【决策建议】")[1].split("【趣味事实】")[0] if "【趣味事实】" in insights else insights.split("【决策建议】")[1]
                                            st.markdown(f"🎯 **决策建议**：{rec_part.strip()}")
                                        
                                        if "【趣味事实】" in insights:
                                            fun_part = insights.split("【趣味事实】")[1]
                                            st.markdown(f"✨ **趣味事实**：{fun_part.strip()}")
                                    else:
                                        st.markdown(insights)
                                    
                                    st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.warning("计算失败，请尝试其他问题")
                            
                    except json.JSONDecodeError:
                        st.warning("AI 理解有误，已切换到基础模式")
                        result_df, chart_type, summary, filter_msg = fallback_analyze(query, df)
                        if result_df is not None:
                            st.markdown("---")
                            st.markdown("### 📊 Analysis Results")
                            if filter_msg:
                                st.info(filter_msg)
                            st.markdown(f"### {summary}")
                            st.dataframe(result_df, use_container_width=True)
                else:
                    st.warning("AI 服务暂时不可用，已切换到基础模式")
                    result_df, chart_type, summary, filter_msg = fallback_analyze(query, df)
                    if result_df is not None:
                        st.markdown("---")
                        st.markdown("### 📊 Analysis Results")
                        if filter_msg:
                            st.info(filter_msg)
                        st.markdown(f"### {summary}")
                        st.dataframe(result_df, use_container_width=True)
        else:
            # 降级模式
            st.markdown('<div><span class="mode-badge-fallback">⚠️ 基础模式（AI 服务暂不可用）</span></div>', unsafe_allow_html=True)
            st.caption("当前支持：品类分布 · 销售额排名 · 筛选查询")
            
            result_df, chart_type, summary, filter_msg = fallback_analyze(query, df)
            
            if result_df is not None:
                st.markdown("---")
                st.markdown("### 📊 Analysis Results")
                if filter_msg:
                    st.info(filter_msg)
                st.markdown(f"### {summary}")
                
                col_left, col_right = st.columns([3, 2])
                with col_left:
                    st.dataframe(result_df, use_container_width=True)
                with col_right:
                    if chart_type != "none" and len(result_df) >= 2:
                        if chart_type == "pie":
                            fig = px.pie(result_df, names=result_df.columns[0], values=result_df.columns[1])
                            fig.update_traces(marker=dict(colors=px.colors.sequential.Blues_r))
                        else:
                            fig = px.bar(result_df, x=result_df.columns[0], y=result_df.columns[1])
                            fig.update_traces(marker_color=PRIMARY_BLUE)
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(summary)
    
    elif analyze_btn and not query:
        st.warning("💡 请输入一个问题～")

else:
    # 未上传文件时的引导页面
    st.markdown("""
    <div class="upload-area" style="text-align: center;">
        <div style="font-size: 48px; margin-bottom: 16px;">📁</div>
        <div style="font-size: 20px; font-weight: 500; margin-bottom: 8px;">上传你的数据文件</div>
        <div style="font-size: 14px; color: #666;">支持 Excel (.xlsx, .xls) 和 CSV 格式</div>
        <div style="font-size: 12px; color: #888; margin-top: 16px;">✨ 数据只在你的电脑处理，不上传任何服务器 ✨</div>
    </div>
    
    <div style="margin-top: 32px;">
        <div class="card" style="text-align: center;">
            <div style="font-size: 24px; margin-bottom: 16px;">🎯 能做什么</div>
            <div style="display: flex; justify-content: center; gap: 32px; flex-wrap: wrap;">
                <div>📊 品类分布</div>
                <div>📈 销售趋势</div>
                <div>🏆 销售额排名</div>
                <div>🔍 筛选查询</div>
                <div>🎯 决策建议</div>
            </div>
        </div>
    </div>
    
    <div style="margin-top: 24px; text-align: center; font-size: 14px; color: #888;">
        <p>💡 试试上传电商数据、销售订单或任何 Excel 文件</p>
        <p>🤖 问它：「各品类销售额分布」「连衣裙」「技术部」「给我一些决策建议」</p>
    </div>
    """, unsafe_allow_html=True)

# ==================== 底部 ====================
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 16px; font-size: 12px; color: #888;">
    ⚡ Made with ☕ by Tuotuo09 · Powered by DeepSeek AI<br>
    🔒 Your data stays on your computer · No uploads · 100% Private
</div>
""", unsafe_allow_html=True)
