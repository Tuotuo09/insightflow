"""
InsightFlow - 智能数据决策助手（最终通用版）
作者：Tuotuo09
功能：通用数据分析 + 多条件筛选 + 动态提示 + AI 洞察
特性：自动识别字段类型，用户问什么就分析什么
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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
if 'filter_desc' not in st.session_state:
    st.session_state.filter_desc = None
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
if 'analysis_summary' not in st.session_state:
    st.session_state.analysis_summary = ""
if 'group_col' not in st.session_state:
    st.session_state.group_col = None
if 'value_col' not in st.session_state:
    st.session_state.value_col = None
if 'agg_func' not in st.session_state:
    st.session_state.agg_func = "sum"

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
            {"role": "system", "content": "你是一个专业的数据分析专家。基于提供的准确数据，给出深刻的洞察和可执行的建议。请只基于提供的数据进行分析，不要编造任何不存在的人名或数据。"},
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

def detect_filters_from_query(query, df):
    """从用户问题中提取筛选条件（按词拆分，精确匹配，多条件AND）"""
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # 构建所有可能的筛选值映射
    value_to_cols = {}
    for col in text_cols:
        unique_vals = df[col].dropna().unique().tolist()
        for val in unique_vals:
            if val:
                if val not in value_to_cols:
                    value_to_cols[val] = []
                value_to_cols[val].append(col)
    
    # 将用户输入拆分成独立的词
    words = []
    # 按空格分词
    for part in query.split():
        if part:
            words.append(part)
    # 也尝试匹配整个短语
    words.append(query)
    # 去重
    words = list(set(words))
    
    # 筛选匹配的词
    filters = []
    seen_cols = set()
    
    for word in words:
        if word in value_to_cols:
            # 精确匹配
            for col in value_to_cols[word]:
                if col not in seen_cols:
                    filters.append((col, word))
                    seen_cols.add(col)
                    break
    
    return filters

def apply_filters(df, filters):
    """应用多个筛选条件（AND）"""
    filtered_df = df.copy()
    filter_desc_parts = []
    
    for col, val in filters:
        if col in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[col] == val]
            filter_desc_parts.append(f"{col}={val}")
    
    return filtered_df, "、".join(filter_desc_parts) if filter_desc_parts else None

def detect_metric_from_query(query, df):
    """从用户问题中提取分析指标（通用版）"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # 排除 ID 类字段
    id_keywords = ['id', '编号', '工号', '序号', '用户id', '员工id']
    real_numeric_cols = [c for c in numeric_cols if not any(kw in c.lower() for kw in id_keywords)]
    
    query_lower = query.lower()
    
    # 1. 检查用户是否在问文本字段分布（如「绩效等级」「部门」「岗位」）
    for col in text_cols:
        col_lower = col.lower()
        if col_lower in query_lower or any(kw in query_lower for kw in [f"{col_lower}分布", f"{col_lower}情况"]):
            return None, None, col  # 返回文本字段名
    
    # 2. 关键词映射（数值字段）
    keyword_map = [
        (['付费', '金额', '消费', '支付', 'payment', 'amount', '销售额', '销售', '收入', '薪资', '工资'], 'sum'),
        (['年龄', 'age'], 'mean'),
        (['单价', '价格', 'price'], 'mean'),
        (['时长', '时间', 'duration', 'time'], 'mean'),
        (['活跃', '天数', '次数', '数量', 'active', 'days', 'times'], 'sum'),
    ]
    
    # 遍历关键词，匹配用户问题
    for keywords, agg_func in keyword_map:
        if any(kw in query_lower for kw in keywords):
            for col in real_numeric_cols:
                col_lower = col.lower()
                if any(kw in col_lower for kw in keywords):
                    return col, agg_func, None
    
    # 3. 默认：返回第一个数值列
    default_col = real_numeric_cols[0] if real_numeric_cols else None
    if default_col:
        col_lower = default_col.lower()
        if any(kw in col_lower for kw in ['年龄', '单价', '价格', '时长']):
            agg_func = "mean"
        else:
            agg_func = "sum"
        return default_col, agg_func, None
    
    return None, None, None

def precompute_stats(df, value_col, agg_func):
    """工具准确计算各项统计"""
    total_rows = len(df)
    total_columns = len(df.columns)
    
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # 分组统计
    group_stats = None
    group_col = None
    if text_cols and value_col and value_col in df.columns:
        group_col = text_cols[0]
        
        if agg_func == "mean":
            grouped = df.groupby(group_col)[value_col].mean().reset_index()
            grouped.columns = [group_col, value_col]
            grouped = grouped.sort_values(value_col, ascending=False)
        else:
            grouped = df.groupby(group_col)[value_col].sum().reset_index()
            grouped.columns = [group_col, value_col]
            grouped = grouped.sort_values(value_col, ascending=False)
        
        group_stats = grouped
    
    # 数值列统计
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    id_keywords = ['id', '编号', '工号', '序号', '用户id', '员工id']
    real_numeric_cols = [c for c in numeric_cols if not any(kw in c.lower() for kw in id_keywords)]
    
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
        "value_col": value_col,
        "agg_func": agg_func
    }

def get_unit(value_col):
    """根据字段名获取单位"""
    col_lower = value_col.lower()
    
    if '年龄' in col_lower or 'age' in col_lower:
        return "岁"
    elif '单价' in col_lower or '价格' in col_lower or 'price' in col_lower:
        return "元"
    elif '小时' in col_lower:
        return "小时"
    elif '时长' in col_lower or '时间' in col_lower or 'duration' in col_lower:
        return "分钟"
    elif '天数' in col_lower or '天' in col_lower or 'days' in col_lower:
        return "天"
    elif '次数' in col_lower or '次' in col_lower:
        return "次"
    elif '付费' in col_lower or '金额' in col_lower or '薪资' in col_lower or '工资' in col_lower:
        return "元"
    else:
        return ""

def generate_text_distribution(df, text_col, filter_desc):
    """生成文本字段的分布统计"""
    value_counts = df[text_col].value_counts().reset_index()
    value_counts.columns = [text_col, "数量"]
    
    summary = ""
    if filter_desc:
        summary += f"📊 {text_col}分布（{filter_desc}）\n"
    else:
        summary += f"📊 {text_col}分布\n"
    
    for _, row in value_counts.iterrows():
        summary += f"• {row[text_col]}：{row['数量']}人\n"
    
    return summary, value_counts

def generate_analysis_summary(stats, filter_desc, df, query, text_col=None, value_col=None, agg_func=None):
    """生成分析结果文字总结（通用版）"""
    
    # 如果用户问的是文本字段分布
    if text_col and text_col in df.columns:
        summary, _ = generate_text_distribution(df, text_col, filter_desc)
        return summary
    
    # 否则处理数值字段
    if stats['group_stats'] is None or len(stats['group_stats']) == 0:
        return "数据中没有找到可分析的字段。"
    
    group_col = stats['group_col']
    value = stats['value_col'] if value_col is None else value_col
    agg = stats['agg_func'] if agg_func is None else agg_func
    unit = get_unit(value)
    
    summary = ""
    
    if filter_desc:
        # 有筛选条件时，显示详细数据
        summary += f"📊 整体数据\n"
        summary += f"• {filter_desc} 共有 {stats['total_rows']} 条记录\n"
        
        # 获取总额
        total_val = 0
        if stats['numeric_stats'] is not None:
            for _, row in stats['numeric_stats'].iterrows():
                if row['字段'] == value:
                    total_val = row['总和']
                    break
        
        if agg == "mean" or total_val == 0:
            avg_val = stats['group_stats'].iloc[0][value]
            summary += f"• 平均{value}：{avg_val:.1f}{unit}\n"
        else:
            avg_val = total_val / stats['total_rows'] if stats['total_rows'] > 0 else 0
            summary += f"• {value}总额：{total_val:,.0f}{unit}（人均 {avg_val:.0f}{unit}）\n"
    
    else:
        # 无筛选条件时，显示全面分析
        summary += f"📊 整体数据\n"
        summary += f"• 总记录数：{stats['total_rows']} 条\n"
        
        # 统计所有数值字段的平均值
        if stats['numeric_stats'] is not None and len(stats['numeric_stats']) > 0:
            for _, row in stats['numeric_stats'].head(3).iterrows():
                unit = get_unit(row['字段'])
                summary += f"• 平均{row['字段']}：{row['平均值']:.1f}{unit}\n"
        
        # 自动选择最有意义的排名（第一个文本列 + 第一个数值列）
        text_cols = df.select_dtypes(include=['object']).columns.tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if text_cols and numeric_cols:
            # 排除 ID 类字段
            id_keywords = ['id', '编号', '工号', '序号']
            real_numeric_cols = [c for c in numeric_cols if not any(kw in c.lower() for kw in id_keywords)]
            
            if real_numeric_cols:
                rank_col = real_numeric_cols[0]
                label_col = text_cols[0]
                
                summary += f"\n📈 {label_col}排名（前10）\n"
                top10 = df.nlargest(10, rank_col)[[label_col, rank_col]].drop_duplicates(subset=[label_col])
                for _, row in top10.iterrows():
                    unit = get_unit(rank_col)
                    summary += f"• {row[label_col]}：{row[rank_col]:,.0f}{unit}\n"
    
    return summary

def generate_ai_insight(query, df, stats, analysis_summary, filter_desc):
    """生成 AI 智能洞察（通用版）"""
    data_summary = f"用户问题：{query}\n\n"
    
    if filter_desc:
        data_summary += f"【筛选条件】当前分析的是「{filter_desc}」的数据\n\n"
    
    data_summary += f"【分析结果】\n{analysis_summary}\n\n"
    
    # 添加数据摘要
    data_summary += f"【数据概况】共 {stats['total_rows']} 条记录\n"
    
    if stats['group_stats'] is not None and len(stats['group_stats']) > 0:
        group_col = stats['group_col']
        value_col = stats['value_col']
        agg_func = stats['agg_func']
        unit = get_unit(value_col)
        data_summary += f"【{group_col}分布】\n"
        for _, row in stats['group_stats'].iterrows():
            if agg_func == "mean":
                data_summary += f"- {row[group_col]}: {row[value_col]:.1f}{unit}\n"
            else:
                data_summary += f"- {row[group_col]}: {row[value_col]:,.0f}{unit}\n"
    
    if stats['numeric_stats'] is not None and len(stats['numeric_stats']) > 0:
        data_summary += f"\n【数值统计】\n"
        for _, row in stats['numeric_stats'].head(5).iterrows():
            unit = get_unit(row['字段'])
            data_summary += f"- {row['字段']}: 平均{row['平均值']:.1f}{unit}, 总和{row['总和']:.0f}{unit}\n"
    
    # 添加排名数据（通用）
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if text_cols and numeric_cols:
        id_keywords = ['id', '编号', '工号', '序号']
        real_numeric_cols = [c for c in numeric_cols if not any(kw in c.lower() for kw in id_keywords)]
        
        if real_numeric_cols:
            rank_col = real_numeric_cols[0]
            label_col = text_cols[0]
            data_summary += f"\n【{label_col}排名前10（真实数据）】\n"
            top10 = df.nlargest(10, rank_col)[[label_col, rank_col]].drop_duplicates(subset=[label_col])
            for _, row in top10.iterrows():
                unit = get_unit(rank_col)
                data_summary += f"- {row[label_col]}: {row[rank_col]:,.0f}{unit}\n"
    
    prompt = f"""基于以下准确数据，给出洞察和建议。注意：请只基于提供的数据进行分析，不要编造任何不存在的人名或数据。

{data_summary}

请按以下格式输出：
【洞察】：（2-3点，基于数据发现的核心问题）
【建议】：（具体可执行）
【趣味发现】：（一个有趣的数据洞察）"""
    
    return call_deepseek(prompt)

def generate_dynamic_example(df):
    """动态生成示例提示"""
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not text_cols or not numeric_cols:
        return "💡 试试：「给我建议」"
    
    # 获取第一个文本列和第一个数值列
    sample_col = text_cols[0]
    sample_num = numeric_cols[0]
    
    # 获取该列的前两个非空值
    unique_vals = df[sample_col].dropna().unique().tolist()
    val1 = unique_vals[0] if len(unique_vals) > 0 else "示例"
    val2 = unique_vals[1] if len(unique_vals) > 1 else ""
    
    # 判断数值字段类型
    num_lower = sample_num.lower()
    if '年龄' in num_lower or '单价' in num_lower or '价格' in num_lower or '时长' in num_lower:
        # 平均值类型
        if val2:
            return f"💡 试试：「{sample_col}」「{val1}」「{val1}平均{sample_num}」「{val1}{val2}」「给我建议」"
        else:
            return f"💡 试试：「{sample_col}」「{val1}」「{val1}平均{sample_num}」「给我建议」"
    else:
        # 总额类型
        if val2:
            return f"💡 试试：「{sample_col}」「{val1}」「{val1}{sample_num}」「{val1}{val2}」「给我建议」"
        else:
            return f"💡 试试：「{sample_col}」「{val1}」「{val1}{sample_num}」「给我建议」"

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
        margin-bottom: 20px;
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
    
    /* 隐藏默认的上传提示文字 */
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
    /* 隐藏默认的上传提示文字 */
    [data-testid="stFileUploader"] > div:first-child > div:first-child {{
        display: none;
    }}
</style>
""", unsafe_allow_html=True)

# ==================== 标题 ====================
st.markdown("""
<div style="text-align: center; margin-bottom: 32px;">
    <h1 class="title">✨ InsightFlow · 智能数据决策助手</h1>
    <p style="font-size: 18px; color: #666; margin-top: -8px;">
        🤖 Built by Tuotuo09 · 自动适配任何数据 · 智能分析 · AI 解读
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
    
    # ==================== 输入区域 ====================
    st.markdown("---")
    
    query = st.text_input("", placeholder="例如：「部门」「绩效等级」「技术部」「给我建议」", label_visibility="collapsed")
    analyze_btn = st.button("🚀 开始分析", type="primary")
    
    # 动态生成示例提示
    example_text = generate_dynamic_example(df)
    st.caption(example_text)
    
    # ==================== 分析逻辑 ====================
    if analyze_btn and query:
        # 检测多个筛选条件（按词拆分，精确匹配）
        filters = detect_filters_from_query(query, df)
        
        if filters:
            display_df, filter_desc = apply_filters(df, filters)
            st.info(f"🔍 已筛选：{filter_desc}，共 {len(display_df)} 条记录")
        else:
            display_df = df
            filter_desc = None
        
        # 检测分析指标（返回数值字段或文本字段）
        value_col, agg_func, text_col = detect_metric_from_query(query, display_df)
        
        # 计算统计
        if text_col:
            # 用户问的是文本字段分布
            stats = None
            analysis_summary, group_stats = generate_text_distribution(display_df, text_col, filter_desc)
        else:
            # 用户问的是数值字段
            stats = precompute_stats(display_df, value_col, agg_func)
            analysis_summary = generate_analysis_summary(stats, filter_desc, display_df, query, text_col, value_col, agg_func)
            group_stats = stats['group_stats'] if stats else None
        
        # 保存到 session_state
        st.session_state.has_result = True
        st.session_state.filtered_df = display_df
        st.session_state.filter_desc = filter_desc
        st.session_state.group_stats = group_stats
        st.session_state.numeric_stats = stats['numeric_stats'] if stats else None
        st.session_state.total_rows = stats['total_rows'] if stats else len(display_df)
        st.session_state.total_columns = stats['total_columns'] if stats else len(display_df.columns)
        st.session_state.analysis_summary = analysis_summary
        st.session_state.group_col = stats['group_col'] if stats else None
        st.session_state.value_col = stats['value_col'] if stats else None
        st.session_state.agg_func = stats['agg_func'] if stats else None
        
        # ==================== 显示分析结果 ====================
        st.markdown("---")
        st.markdown("### 📊 分析结果")
        st.markdown(analysis_summary)
        
        # ==================== AI 智能洞察 ====================
        api_ok = check_api_availability()
        if api_ok:
            with st.spinner(random.choice(LOADING_MESSAGES)):
                if text_col:
                    # 文本字段分布，生成简单的 AI 洞察
                    ai_response = f"【洞察】\n{text_col}分布显示，各类型的分布情况如上表所示。\n\n【建议】\n可根据分布情况，关注占比最高的类型，分析其特点和优势。\n\n【趣味发现】\n{text_col}分布中，{display_df[text_col].value_counts().index[0]}占比最高。"
                else:
                    ai_response = generate_ai_insight(query, display_df, stats, analysis_summary, filter_desc)
                
                if ai_response:
                    st.session_state.ai_response = ai_response
                    st.markdown("### 🧠 Tuotuo's AI 智能洞察")
                    
                    if "【洞察】" in ai_response:
                        parts = ai_response.split("【洞察】")
                        if len(parts) > 1:
                            insight_part = parts[1].split("【建议】")[0] if "【建议】" in parts[1] else parts[1]
                            st.markdown(f"**🔍 洞察**\n{insight_part.strip()}")
                    
                    if "【建议】" in ai_response:
                        rec_part = ai_response.split("【建议】")[1].split("【趣味发现】")[0] if "【趣味发现】" in ai_response else ai_response.split("【建议】")[1]
                        st.markdown(f"\n**💡 建议**\n{rec_part.strip()}")
                    
                    if "【趣味发现】" in ai_response:
                        fun_part = ai_response.split("【趣味发现】")[1]
                        st.markdown(f"\n**✨ 趣味发现**\n{fun_part.strip()}")
                else:
                    st.warning("🤖 AI 服务繁忙，请稍后再试")
        else:
            st.warning("🤖 AI 服务暂不可用，请检查 API Key")
        
        # ==================== 数据明细 & 图表 ====================
        if group_stats is not None and len(group_stats) > 0:
            st.markdown("---")
            st.markdown("### 📊 数据明细 & 图表")
            
            col_left, col_right = st.columns([3, 2])
            
            with col_left:
                st.markdown("**📋 数据明细**")
                st.dataframe(group_stats, use_container_width=True)
            
            with col_right:
                st.markdown("**📈 图表**")
                fig = px.bar(group_stats, x=group_stats.columns[0], y=group_stats.columns[1], title=f"{group_stats.columns[0]}分布")
                fig.update_traces(marker_color=PRIMARY_BLUE)
                fig.update_layout(
                    title_font_size=14,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # 数据明细分页
        if len(display_df) > 0:
            st.markdown("---")
            st.markdown(f"### 📋 全部数据明细（共 {len(display_df)} 条）")
            
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
    
    # 显示上次结果
    elif st.session_state.has_result and not analyze_btn:
        st.markdown("---")
        st.markdown("### 📊 分析结果")
        st.markdown(st.session_state.analysis_summary)
        
        if st.session_state.ai_response:
            st.markdown("### 🧠 Tuotuo's AI 智能洞察")
            response = st.session_state.ai_response
            
            if "【洞察】" in response:
                parts = response.split("【洞察】")
                if len(parts) > 1:
                    insight_part = parts[1].split("【建议】")[0] if "【建议】" in parts[1] else parts[1]
                    st.markdown(f"**🔍 洞察**\n{insight_part.strip()}")
            
            if "【建议】" in response:
                rec_part = response.split("【建议】")[1].split("【趣味发现】")[0] if "【趣味发现】" in response else response.split("【建议】")[1]
                st.markdown(f"\n**💡 建议**\n{rec_part.strip()}")
            
            if "【趣味发现】" in response:
                fun_part = response.split("【趣味发现】")[1]
                st.markdown(f"\n**✨ 趣味发现**\n{fun_part.strip()}")
        
        if st.session_state.group_stats is not None and len(st.session_state.group_stats) > 0:
            st.markdown("---")
            st.markdown("### 📊 数据明细 & 图表")
            
            col_left, col_right = st.columns([3, 2])
            with col_left:
                st.markdown("**📋 数据明细**")
                st.dataframe(st.session_state.group_stats, use_container_width=True)
            with col_right:
                st.markdown("**📈 图表**")
                group_col = st.session_state.group_stats.columns[0]
                value_col = st.session_state.group_stats.columns[1]
                fig = px.bar(st.session_state.group_stats, x=group_col, y=value_col, title=f"{group_col}分布")
                fig.update_traces(marker_color=PRIMARY_BLUE)
                fig.update_layout(
                    title_font_size=14,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        if st.session_state.filtered_df is not None:
            st.markdown("---")
            st.markdown(f"### 📋 全部数据明细（共 {len(st.session_state.filtered_df)} 条）")
            
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
            <div>📊 自动识别字段</div>
            <div>📈 智能统计分析</div>
            <div>🔍 多条件筛选</div>
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
    ⚡ Made with ☕ by Tuotuo09 · 自动适配任何数据 · AI 智能解读<br>
    🔒 数据本地处理 · 只发送统计结果
</div>
""", unsafe_allow_html=True)
