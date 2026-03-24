"""
InsightFlow - AI 智能数据决策助手
作者：Tuotuo09
功能：上传任意 Excel，自然语言提问，AI 智能分析 + 智能图表 + 决策建议
     自带降级模式，API 异常时自动切换规则匹配
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
import time

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
    .fallback-card {{
        background: linear-gradient(135deg, #FFF3E0, white);
        border-radius: 16px;
        padding: 24px;
        border-left: 4px solid {WARNING_ORANGE};
    }}
    .fun-fact {{
        background-color: #FFF3E0;
        border-radius: 12px;
        padding: 12px 20px;
        margin-top: 16px;
        font-size: 14px;
        border-left: 3px solid {WARNING_ORANGE};
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
        💡 随便问 · AI 帮你分析 · 智能图表 · 决策建议
    </p>
</div>
""", unsafe_allow_html=True)

# ==================== 会话状态初始化 ====================
if 'api_available' not in st.session_state:
    st.session_state.api_available = None
if 'api_checked' not in st.session_state:
    st.session_state.api_checked = False

# ==================== DeepSeek API 配置 ====================
# 比赛演示时，在这里填入你的 API Key
# 或者部署到 Streamlit Cloud 后，在 Secrets 中配置
DEEPSEEK_API_KEY = "sk-52bcbd3d232945828250c3a1408598ff"

def check_api_availability():
    """检查 API 是否可用（只检查一次）"""
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
            {"role": "system", "content": "你是一个专业的数据分析专家。你需要理解用户的数据分析需求，并给出清晰的洞察和建议。回答要简洁、专业、有决策价值。请按指定格式输出。"},
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
        else:
            return None
    except:
        return None

def ai_analyze(query, df):
    """AI 模式：智能分析 + 智能图表"""
    # 准备数据摘要
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # 排除 ID 类字段
    id_keywords = ['id', '编号', '工号', '序号', '员工id']
    real_numeric_cols = [c for c in numeric_cols if not any(kw in c.lower() for kw in id_keywords)]
    if not real_numeric_cols:
        real_numeric_cols = numeric_cols
    
    # 数据摘要
    stats = f"数据共有{len(df)}行，{len(df.columns)}列。"
    stats += f"数值列：{', '.join(real_numeric_cols[:5])}。"
    stats += f"文本列：{', '.join(text_cols[:5])}。"
    
    # 关键统计（只发送统计结果）
    summary = {}
    for col in real_numeric_cols[:3]:
        summary[col] = {
            "平均值": round(df[col].mean(), 2),
            "最大值": df[col].max(),
            "最小值": df[col].min()
        }
    
    for col in text_cols[:3]:
        top_values = df[col].value_counts().head(5).to_dict()
        summary[col] = {"前5个值": top_values}
    
    # 准备示例数据（前10行，用于展示）
    sample_data = df.head(10).to_dict('records')
    
    # 构建 AI 提示词（要求返回 JSON 格式）
    prompt = f"""
用户问题：{query}

数据概况：
- {stats}
- 关键统计：{json.dumps(summary, ensure_ascii=False)}
- 示例数据（前10行）：{json.dumps(sample_data, ensure_ascii=False, default=str)}

请分析用户问题，并返回 JSON 格式的结果。JSON 必须包含以下字段：
- chart_type: 图表类型，可选值：'pie'(饼图，用于分布)、'bar'(柱状图，用于排名/对比)、'line'(折线图，用于趋势)、'none'(不需要图表)
- chart_x: 图表X轴字段名（如果有图表）
- chart_y: 图表Y轴字段名（如果有图表）
- insight: 数据洞察（100字以内）
- recommendation: 决策建议（150字以内，具体可操作）
- fun_fact: 趣味事实（50字以内）
- summary: 一句话总结分析结果（50字以内）

只返回 JSON，不要有其他内容。
"""
    
    response = call_deepseek(prompt)
    
    if response:
        try:
            # 尝试解析 JSON
            result = json.loads(response)
            return result
        except:
            # 如果 JSON 解析失败，返回默认结构
            return {
                "chart_type": "none",
                "insight": "AI 分析完成",
                "recommendation": "请查看数据详情",
                "fun_fact": "数据共" + str(len(df)) + "条记录",
                "summary": "分析完成"
            }
    return None

# ==================== 降级模式：规则匹配 ====================
def fallback_analyze(query, df):
    """降级模式：规则匹配分析"""
    query_lower = query.lower()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # 排除 ID 类字段
    id_keywords = ['id', '编号', '工号', '序号', '员工id']
    real_numeric_cols = [c for c in numeric_cols if not any(kw in c.lower() for kw in id_keywords)]
    if not real_numeric_cols:
        real_numeric_cols = numeric_cols
    
    result_df = None
    chart_type = None
    chart_x = None
    chart_y = None
    insight = ""
    summary = ""
    
    # 部门分布
    if "分布" in query_lower and text_cols:
        col = text_cols[0]
        result_df = df[col].value_counts().reset_index()
        result_df.columns = [col, "数量"]
        chart_type = "pie"
        chart_x = col
        chart_y = "数量"
        top = result_df.iloc[0][col]
        top_pct = result_df.iloc[0]["数量"] / len(df) * 100
        summary = f"{col}分布：{top}最多，占比{top_pct:.1f}%"
        insight = f"{col}分布情况，{top}占比最高"
    
    # 排名/前几名
    elif "前" in query_lower and real_numeric_cols:
        col = real_numeric_cols[0]
        nums = re.findall(r'\d+', query_lower)
        n = int(nums[0]) if nums else 5
        result_df = df.nlargest(n, col)[[col] + text_cols[:2]]
        chart_type = "bar"
        chart_x = text_cols[0] if text_cols else "index"
        chart_y = col
        summary = f"{col}最高的{n}名"
        insight = f"按{col}降序排列，显示前{n}名"
    
    # 平均
    elif "平均" in query_lower and real_numeric_cols:
        col = real_numeric_cols[0]
        avg_val = df[col].mean()
        summary = f"平均{col}：{avg_val:.0f}"
        insight = f"数据的平均{col}为{avg_val:.0f}"
        result_df = pd.DataFrame([{"指标": f"平均{col}", "数值": f"{avg_val:.0f}"}])
        chart_type = "none"
    
    # 筛选部门
    elif any(dept in query_lower for dept in ["技术部", "产品部", "市场部", "销售部", "人力资源部", "财务部", "运营部"]):
        for dept in ["技术部", "产品部", "市场部", "销售部", "人力资源部", "财务部", "运营部"]:
            if dept in query_lower and "部门" in df.columns:
                result_df = df[df["部门"] == dept]
                summary = f"{dept}共有{len(result_df)}人"
                insight = f"{dept}员工列表，共{len(result_df)}人"
                chart_type = "none"
                break
    
    # 总人数
    elif any(kw in query_lower for kw in ["多少人", "总人数", "一共", "总数"]):
        total = len(df)
        summary = f"数据共有{total}条记录"
        insight = f"当前数据共有{total}行"
        result_df = pd.DataFrame([{"统计项": "总记录数", "数值": total}])
        chart_type = "none"
    
    # 默认推荐
    else:
        suggestions = []
        if text_cols:
            suggestions.append(f"「{text_cols[0]}分布」")
        if real_numeric_cols:
            suggestions.append(f"「{real_numeric_cols[0]}前10」")
            suggestions.append(f"「平均{real_numeric_cols[0]}」")
        if "部门" in df.columns:
            suggestions.append("「技术部」")
        
        summary = "试试这些问题：" + " · ".join(suggestions[:4])
        insight = "基础模式支持的问题类型"
        result_df = pd.DataFrame([{"提示": "支持的问题", "示例": s} for s in suggestions[:5]])
        chart_type = "none"
    
    return result_df, chart_type, chart_x, chart_y, summary, insight

def create_chart(df, chart_type, x_col, y_col):
    """创建图表"""
    if chart_type == "none" or df is None:
        return None
    
    try:
        if chart_type == "pie" and x_col and y_col:
            fig = px.pie(df, names=x_col, values=y_col, title=f"{x_col}分布")
            fig.update_traces(marker=dict(colors=px.colors.sequential.Blues_r))
            return fig
        elif chart_type == "bar" and x_col and y_col:
            fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col}排行")
            fig.update_traces(marker_color=PRIMARY_BLUE)
            return fig
        elif chart_type == "line" and x_col and y_col:
            fig = px.line(df, x=x_col, y=y_col, title=f"{y_col}趋势")
            fig.update_traces(line_color=PRIMARY_BLUE)
            return fig
        elif len(df.columns) >= 2:
            # 自动选择
            fig = px.bar(df, x=df.columns[0], y=df.columns[1])
            return fig
    except:
        return None
    return None

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
    st.caption("随便问，AI 会帮你分析数据并给出决策建议")
    
    query = st.text_input("", placeholder="例如：哪个部门人最多？｜加班最多的员工？｜技术部薪资合理吗？", label_visibility="collapsed")
    
    col_btn, col_space = st.columns([1, 5])
    with col_btn:
        analyze_btn = st.button("🚀 Analyze", type="primary", use_container_width=True)
    
    st.markdown("💡 **试试这些**：部门分布 · 薪资前10 · 加班最多的员工 · 平均年龄 · 技术部")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 分析逻辑
    if analyze_btn and query:
        # 检查 API 可用性
        api_ok = check_api_availability()
        
        if api_ok:
            # AI 模式
            st.markdown('<div><span class="mode-badge-ai">🤖 AI 智能模式</span></div>', unsafe_allow_html=True)
            
            with st.spinner("AI 正在思考中..."):
                ai_result = ai_analyze(query, df)
                
                if ai_result:
                    # 获取 AI 返回的信息
                    chart_type = ai_result.get("chart_type", "none")
                    chart_x = ai_result.get("chart_x", None)
                    chart_y = ai_result.get("chart_y", None)
                    insight = ai_result.get("insight", "")
                    recommendation = ai_result.get("recommendation", "")
                    fun_fact = ai_result.get("fun_fact", "")
                    summary = ai_result.get("summary", "")
                    
                    # 如果有图表需求，准备数据
                    result_df = None
                    if chart_type != "none" and chart_x and chart_y and chart_x in df.columns and chart_y in df.columns:
                        if chart_type == "pie":
                            result_df = df.groupby(chart_x)[chart_y].sum().reset_index()
                        elif chart_type in ["bar", "line"]:
                            result_df = df.groupby(chart_x)[chart_y].sum().reset_index()
                            result_df = result_df.sort_values(chart_y, ascending=False).head(10)
                    
                    # 显示结果
                    st.markdown("---")
                    st.markdown("### 📊 Analysis Results")
                    
                    if summary:
                        st.markdown(f"### {summary}")
                    
                    # 左右布局：表格 + 图表
                    col_left, col_right = st.columns([3, 2])
                    
                    with col_left:
                        if result_df is not None:
                            st.dataframe(result_df, use_container_width=True)
                        else:
                            # 显示数据摘要
                            st.info("数据概览")
                            st.dataframe(df.head(20), use_container_width=True)
                    
                    with col_right:
                        if result_df is not None:
                            fig = create_chart(result_df, chart_type, chart_x, chart_y)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                        elif text_cols:
                            # 默认显示第一个文本列的分布
                            default_data = df[text_cols[0]].value_counts().head(5)
                            fig = px.pie(values=default_data.values, names=default_data.index, title=f"{text_cols[0]}分布")
                            fig.update_traces(marker=dict(colors=px.colors.sequential.Blues_r))
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # AI 决策建议区域
                    st.markdown("---")
                    st.markdown('<div class="ai-card">', unsafe_allow_html=True)
                    st.markdown("### 🧠 Tuotuo's AI Intelligence")
                    
                    if insight:
                        st.markdown(f"🔍 **洞察**：{insight}")
                    if recommendation:
                        st.markdown(f"🎯 **决策建议**：{recommendation}")
                    if fun_fact:
                        st.markdown(f"✨ **趣味事实**：{fun_fact}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.warning("⚠️ AI 服务响应异常，已切换到基础模式")
                    # 降级到规则匹配
                    result_df, chart_type, chart_x, chart_y, summary, insight = fallback_analyze(query, df)
                    if result_df is not None:
                        st.markdown("---")
                        st.markdown("### 📊 Analysis Results")
                        st.markdown(f"### {summary}")
                        
                        col_left, col_right = st.columns([3, 2])
                        with col_left:
                            st.dataframe(result_df, use_container_width=True)
                        with col_right:
                            if chart_type != "none" and chart_x and chart_y:
                                fig = create_chart(result_df, chart_type, chart_x, chart_y)
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown(f"🔍 **洞察**：{insight}")
        
        else:
            # 降级模式
            st.markdown('<div><span class="mode-badge-fallback">⚠️ 基础模式（AI 服务暂不可用）</span></div>', unsafe_allow_html=True)
            st.caption("当前支持：部门分布 · 薪资前10 · 平均年龄 · 部门筛选")
            
            result_df, chart_type, chart_x, chart_y, summary, insight = fallback_analyze(query, df)
            
            if result_df is not None:
                st.markdown("---")
                st.markdown("### 📊 Analysis Results")
                st.markdown(f"### {summary}")
                
                col_left, col_right = st.columns([3, 2])
                with col_left:
                    st.dataframe(result_df, use_container_width=True)
                with col_right:
                    if chart_type != "none" and chart_x and chart_y:
                        fig = create_chart(result_df, chart_type, chart_x, chart_y)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    elif text_cols and len(result_df) > 0:
                        # 自动生成简单图表
                        if len(result_df.columns) >= 2:
                            fig = create_chart(result_df, "bar", result_df.columns[0], result_df.columns[1])
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown(f"🔍 **洞察**：{insight}")
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
                <div>📊 数据分布</div>
                <div>📈 趋势分析</div>
                <div>🏆 排名对比</div>
                <div>🎯 决策建议</div>
                <div>✨ 趣味洞察</div>
                <div>📉 智能图表</div>
            </div>
        </div>
    </div>
    
    <div style="margin-top: 24px; text-align: center; font-size: 14px; color: #888;">
        <p>💡 试试上传一个人事数据、销售订单或任何 Excel 文件</p>
        <p>🤖 问它：「哪个部门人最多？」「加班最多的员工？」「给我一些决策建议」</p>
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
