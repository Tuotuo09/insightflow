"""
InsightFlow - 智能数据决策助手（最终版）
作者：Tuotuo09
功能：通用数据分析 + 多条件筛选 + 动态提示 + AI 洞察 + 隐私模式
特性：分析结果横向展示，简洁清晰
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
if 'privacy_mode' not in st.session_state:
    st.session_state.privacy_mode = True

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
            {"role": "system", "content": "你是一个专业的数据分析专家。基于提供的准确数据，给出深刻的洞察和可执行的建议。请直接使用【准确统计数据】中提供的数值，不要重新计算。"},
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

def is_sensitive_field(col_name):
    """判断是否为敏感字段（个人标识字段）"""
    sensitive_keywords = ['姓名', '名字', '名称', '员工', '用户', '客户', '手机', '电话', '邮箱', '地址', 'id', '编号', '工号', '账号']
    col_lower = col_name.lower()
    for kw in sensitive_keywords:
        if kw in col_lower:
            return True
    return False

def should_show_sum(field_name):
    """判断字段是否应该显示总和（年龄、单价等不应该显示总和）"""
    no_sum_keywords = ['年龄', 'age', '单价', '价格', 'price', '时长', '比例', '率']
    field_lower = field_name.lower()
    for kw in no_sum_keywords:
        if kw in field_lower:
            return False
    return True

def get_rank_label_col(df, privacy_mode):
    """获取排名用的标签列（隐私模式用部门/岗位，普通模式用姓名）"""
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if privacy_mode:
        # 隐私模式：找第一个非敏感的文本列（部门、岗位、品类等）
        for col in text_cols:
            if not is_sensitive_field(col):
                return col
        return text_cols[0] if text_cols else None
    else:
        # 普通模式：优先使用姓名列，否则用第一个文本列
        for col in text_cols:
            if '姓名' in col or '名字' in col:
                return col
        return text_cols[0] if text_cols else None

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
    
    # 0. 检查是否是对比类问题（如「各部门薪酬对比」）
    if '对比' in query_lower or '比较' in query_lower:
        if text_cols and real_numeric_cols:
            return real_numeric_cols[0], 'sum', None
    
    # 1. 检查用户是否在问文本字段分布
    for col in text_cols:
        col_lower = col.lower()
        if col_lower in query_lower or any(kw in query_lower for kw in [f"{col_lower}分布", f"{col_lower}情况"]):
            return None, None, col
    
    # 2. 关键词映射（数值字段）
    keyword_map = [
        (['薪酬', '薪资', '工资', '收入', '金额', '付费', '销售额', '销售'], 'sum'),
        (['年龄', 'age'], 'mean'),
        (['单价', '价格', 'price'], 'mean'),
        (['时长', '时间', 'duration', 'time'], 'mean'),
        (['活跃', '天数', '次数', '数量', 'active', 'days', 'times'], 'mean'),
        (['简历', '面试', '录用', '入职'], 'sum'),
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
    elif '薪酬' in col_lower or '薪资' in col_lower or '工资' in col_lower or '金额' in col_lower or '付费' in col_lower:
        return "元"
    elif '简历' in col_lower or '面试' in col_lower or '录用' in col_lower or '入职' in col_lower:
        return "人"
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

def extract_target_field_from_query(query):
    """从用户问题中提取目标字段关键词"""
    field_keywords = ['薪酬', '薪资', '工资', '年龄', '出勤', '迟到', '加班', '活跃', '登录', '付费', '销售额', '简历', '面试', '录用', '入职']
    for kw in field_keywords:
        if kw in query:
            return kw
    return None

def format_number(num):
    """格式化数字，添加千分位分隔符"""
    return f"{num:,.0f}"

def generate_analysis_summary(stats, filter_desc, df, query, text_col=None, value_col=None, agg_func=None, privacy_mode=True):
    """生成分析结果文字总结（横向展示版）"""
    
    # 如果用户问的是文本字段分布
    if text_col and text_col in df.columns:
        summary, _ = generate_text_distribution(df, text_col, filter_desc)
        return summary
    
    # 提取用户指定的目标字段
    target_field = extract_target_field_from_query(query)
    
    summary = ""
    
    if filter_desc:
        summary += f"📊 整体数据\n"
        summary += f"• {filter_desc} 共有 {stats['total_rows']} 条记录\n"
    else:
        summary += f"📊 整体数据\n"
        summary += f"• 总记录数：{stats['total_rows']} 条\n"
    
    # 遍历所有数值字段，横向展示统计
    if stats['numeric_stats'] is not None and len(stats['numeric_stats']) > 0:
        for _, row in stats['numeric_stats'].iterrows():
            field_name = row['字段']
            unit = get_unit(field_name)
            show_sum = should_show_sum(field_name)
            
            # 如果用户指定了目标字段，只显示匹配的字段
            if target_field:
                if target_field not in field_name and field_name not in target_field:
                    continue
            
            # 横向展示各项指标
            items = []
            items.append(f"平均值：{row['平均值']:.1f}{unit}")
            items.append(f"最高值：{format_number(row['最大值'])}{unit}")
            items.append(f"最低值：{format_number(row['最小值'])}{unit}")
            if show_sum:
                items.append(f"总和：{format_number(row['总和'])}{unit}")
            
            summary += f"\n📈 {field_name}\n"
            summary += " | ".join(items) + "\n"
    
    # 添加排名（只显示前5名，横向展示）
    rank_label_col = get_rank_label_col(df, privacy_mode)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    id_keywords = ['id', '编号', '工号', '序号', '用户id', '员工id']
    real_numeric_cols = [c for c in numeric_cols if not any(kw in c.lower() for kw in id_keywords)]
    
    if rank_label_col and real_numeric_cols:
        rank_col = real_numeric_cols[0]
        unit = get_unit(rank_col)
        
        # 计算各分组平均值用于排名
        grouped_mean = df.groupby(rank_label_col)[rank_col].mean().sort_values(ascending=False)
        
        if len(grouped_mean) > 0:
            rank_items = [f"{label}：{format_number(value)}{unit}" for label, value in grouped_mean.head(5).items()]
            summary += f"\n🏆 {rank_label_col}排名（前5）\n"
            summary += " | ".join(rank_items) + "\n"
    
    return summary

def generate_ai_insight(query, df, stats, analysis_summary, filter_desc, privacy_mode=True):
    """生成 AI 智能洞察（隐私模式下不发送个人数据）"""
    data_summary = f"用户问题：{query}\n\n"
    
    if filter_desc:
        data_summary += f"【筛选条件】当前分析的是「{filter_desc}」的数据\n\n"
    
    data_summary += f"【分析结果】\n{analysis_summary}\n\n"
    
    # 添加准确的统计数据（只发送汇总统计，不发送个人排名）
    data_summary += f"【准确统计数据（请直接使用以下数据，不要重新计算）】\n"
    
    if stats['numeric_stats'] is not None and len(stats['numeric_stats']) > 0:
        for _, row in stats['numeric_stats'].iterrows():
            unit = get_unit(row['字段'])
            data_summary += f"- 平均{row['字段']}：{row['平均值']:.1f}{unit}\n"
            data_summary += f"- 总{row['字段']}：{row['总和']:.0f}{unit}\n"
            data_summary += f"- 最高{row['字段']}：{row['最大值']:.0f}{unit}\n"
            data_summary += f"- 最低{row['字段']}：{row['最小值']:.0f}{unit}\n"
    
    # 隐私模式下：不发送分组统计中的个人数据
    if not privacy_mode and stats['group_stats'] is not None and len(stats['group_stats']) > 0:
        group_col = stats['group_col']
        value_col = stats['value_col']
        agg_func = stats['agg_func']
        unit = get_unit(value_col)
        data_summary += f"\n【{group_col}分布】\n"
        for _, row in stats['group_stats'].iterrows():
            if agg_func == "mean":
                data_summary += f"- {row[group_col]}: {row[value_col]:.1f}{unit}\n"
            else:
                data_summary += f"- {row[group_col]}: {row[value_col]:,.0f}{unit}\n"
    
    # 构建提示词
    if privacy_mode:
        privacy_instruction = "⚠️ 重要：当前为隐私模式，数据中不包含个人姓名。请只基于部门、岗位等分类数据进行洞察，绝对不要编造或提及任何具体个人姓名。"
    else:
        privacy_instruction = ""
    
    prompt = f"""基于以下准确数据，给出洞察和建议。请直接使用【准确统计数据】中提供的数值，不要重新计算。

{privacy_instruction}

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
        if val2:
            return f"💡 试试：「{sample_col}」「{val1}」「{val1}平均{sample_num}」「{val1}{val2}」「给我建议」"
        else:
            return f"💡 试试：「{sample_col}」「{val1}」「{val1}平均{sample_num}」「给我建议」"
    else:
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

# ==================== 隐私模式开关 ====================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    privacy_mode = st.checkbox(
        "🔒 隐私模式（开启后自动隐藏姓名、ID等个人标识字段）",
        value=st.session_state.privacy_mode,
        help="开启后，分析结果、数据明细、AI洞察中都不会显示个人敏感信息"
    )
    st.session_state.privacy_mode = privacy_mode
    
    if privacy_mode:
        st.info("🔒 隐私模式已开启，个人姓名、ID等字段将不会出现在任何地方")
    else:
        st.info("🔓 隐私模式已关闭，将显示完整数据")

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
        # 隐私模式下过滤敏感字段
        if privacy_mode:
            display_cols = [col for col in df.columns if not is_sensitive_field(col)]
            display_df_preview = df[display_cols]
        else:
            display_df_preview = df
        st.dataframe(display_df_preview.head(100), use_container_width=True)
        st.caption(f"共 {len(df)} 行，{len(df.columns)} 列")
    
    # ==================== 输入区域 ====================
    st.markdown("---")
    
    query = st.text_input("", placeholder="例如：「部门」「绩效等级」「技术部」「各部门薪酬对比」「给我建议」", label_visibility="collapsed")
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
            analysis_summary = generate_analysis_summary(stats, filter_desc, display_df, query, text_col, value_col, agg_func, privacy_mode)
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
                    top_value = display_df[text_col].value_counts().index[0]
                    top_count = display_df[text_col].value_counts().values[0]
                    ai_response = f"【洞察】\n{text_col}分布显示，{top_value}占比最高，共{top_count}人。\n\n【建议】\n可根据分布情况，关注占比最高的类型，分析其特点和优势。\n\n【趣味发现】\n{text_col}分布中，{top_value}是最常见的类型。"
                else:
                    ai_response = generate_ai_insight(query, display_df, stats, analysis_summary, filter_desc, privacy_mode)
                
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
            
            # 隐私模式下过滤敏感字段
            if privacy_mode:
                display_columns = [col for col in group_stats.columns if not is_sensitive_field(col)]
                display_group_stats = group_stats[display_columns]
            else:
                display_group_stats = group_stats
            
            col_left, col_right = st.columns([3, 2])
            
            with col_left:
                st.markdown("**📋 数据明细**")
                st.dataframe(display_group_stats, use_container_width=True)
            
            with col_right:
                st.markdown("**📈 图表**")
                if len(display_group_stats.columns) >= 2:
                    chart_col = display_group_stats.columns[0]
                    value_col = display_group_stats.columns[1]
                    fig = px.bar(display_group_stats, x=chart_col, y=value_col, title=f"{chart_col}分布")
                    fig.update_traces(marker_color=PRIMARY_BLUE)
                    fig.update_layout(
                        title_font_size=14,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # ==================== 全部数据明细 ====================
        if len(display_df) > 0:
            st.markdown("---")
            st.markdown(f"### 📋 全部数据明细（共 {len(display_df)} 条）")
            
            # 隐私模式下过滤敏感字段
            if privacy_mode:
                display_cols = [col for col in display_df.columns if not is_sensitive_field(col)]
                display_data = display_df[display_cols]
            else:
                display_data = display_df
            
            rows_per_page = 10
            total_pages = (len(display_data) + rows_per_page - 1) // rows_per_page
            page_key = "data_page"
            if page_key not in st.session_state:
                st.session_state[page_key] = 1
            
            current_page = st.session_state[page_key]
            start_idx = (current_page - 1) * rows_per_page
            end_idx = min(start_idx + rows_per_page, len(display_data))
            
            st.dataframe(display_data.iloc[start_idx:end_idx], use_container_width=True)
            
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
            
            # 隐私模式下过滤敏感字段
            if privacy_mode:
                display_columns = [col for col in st.session_state.group_stats.columns if not is_sensitive_field(col)]
                display_group_stats = st.session_state.group_stats[display_columns]
            else:
                display_group_stats = st.session_state.group_stats
            
            col_left, col_right = st.columns([3, 2])
            with col_left:
                st.markdown("**📋 数据明细**")
                st.dataframe(display_group_stats, use_container_width=True)
            with col_right:
                st.markdown("**📈 图表**")
                if len(display_group_stats.columns) >= 2:
                    group_col = display_group_stats.columns[0]
                    value_col = display_group_stats.columns[1]
                    fig = px.bar(display_group_stats, x=group_col, y=value_col, title=f"{group_col}分布")
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
            
            # 隐私模式下过滤敏感字段
            if privacy_mode:
                display_cols = [col for col in st.session_state.filtered_df.columns if not is_sensitive_field(col)]
                display_data = st.session_state.filtered_df[display_cols]
            else:
                display_data = st.session_state.filtered_df
            
            rows_per_page = 10
            total_pages = (len(display_data) + rows_per_page - 1) // rows_per_page
            page_key = "data_page"
            if page_key not in st.session_state:
                st.session_state[page_key] = 1
            
            current_page = st.session_state[page_key]
            start_idx = (current_page - 1) * rows_per_page
            end_idx = min(start_idx + rows_per_page, len(display_data))
            
            st.dataframe(display_data.iloc[start_idx:end_idx], use_container_width=True)
            
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
