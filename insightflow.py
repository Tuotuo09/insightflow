import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
import re

st.set_page_config(page_title="InsightFlow - 智能数据洞察助手", layout="wide")

st.title("📊 InsightFlow")
st.caption("智能数据洞察助手 · 上传Excel，用自然语言提问，AI智能分析")

uploaded_file = st.file_uploader("上传数据文件", type=['xlsx', 'xls', 'csv'])

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    # 数据概览
    st.subheader("📋 数据概览")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # 排除ID类字段（员工ID、编号等）
    id_keywords = ['id', '编号', '工号', '序号']
    real_numeric_cols = []
    for col in numeric_cols:
        is_id = False
        for kw in id_keywords:
            if kw in col.lower():
                is_id = True
                break
        if not is_id:
            real_numeric_cols.append(col)
    
    # 如果没有其他数值列，才用ID列
    if not real_numeric_cols:
        real_numeric_cols = numeric_cols
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("总行数", len(df))
    with col2:
        st.metric("总列数", len(df.columns))
    with col3:
        st.metric("数值列", len(real_numeric_cols))
    
    with st.expander("🔍 查看数据详情"):
        st.dataframe(df.head(100))
        st.write("**字段列表**")
        st.write(df.dtypes.reset_index().rename(columns={'index': '字段', 0: '类型'}))
    
    st.divider()
    st.subheader("💬 自然语言查询")
    st.caption("随便问，AI会智能分析你的问题")
    
    query = st.text_input("输入你的问题", placeholder="例如：部门分布、薪资前10、平均年龄、技术部、哪个部门人最多...")
    
    if st.button("🔍 分析", type="primary"):
        with st.spinner("🤖 AI正在智能分析中..."):
            query_lower = query.lower()
            result_df = None
            chart_type = None
            insight = ""
            answer = ""
            
            # ========== 智能意图识别 ==========
            
            # 1. 判断是否问平均（优先匹配具体字段名）
            if "平均" in query_lower:
                # 尝试找到用户想平均的字段
                target_col = None
                for col in real_numeric_cols:
                    if col.lower() in query_lower:
                        target_col = col
                        break
                # 如果没找到，用第一个非ID数值列
                if target_col is None and real_numeric_cols:
                    target_col = real_numeric_cols[0]
                
                if target_col:
                    avg_val = df[target_col].mean()
                    answer = f"📈 平均{target_col}：{avg_val:.0f}"
                    insight = f"数据的平均{target_col}为 {avg_val:.0f}"
                    result_df = pd.DataFrame([{"指标": f"平均{target_col}", "数值": f"{avg_val:.0f}"}])
            
            # 2. 判断是否问分布
            elif any(kw in query_lower for kw in ["分布", "占比", "比例", "构成"]):
                if text_cols:
                    col = text_cols[0]
                    result_df = df[col].value_counts().reset_index()
                    result_df.columns = [col, "数量"]
                    total = len(df)
                    top = result_df.iloc[0][col]
                    top_pct = result_df.iloc[0]["数量"] / total * 100
                    answer = f"📊 {col}分布：{top}最多，占比 {top_pct:.1f}%"
                    insight = f"{col}分布情况，{top}占比最高"
                    chart_type = "pie"
            
            # 3. 判断是否问最多/最大
            elif any(kw in query_lower for kw in ["最多", "最大", "最高", "哪个"]):
                if text_cols and real_numeric_cols:
                    col = text_cols[0]
                    val_col = real_numeric_cols[0]
                    max_row = df.loc[df[val_col].idxmax()]
                    answer = f"🏆 {col}中【{max_row[col]}】的{val_col}最高，为 {max_row[val_col]:,}"
                    insight = f"{val_col}最高的是 {max_row[col]}"
                    result_df = df.nlargest(5, val_col)[[col, val_col]]
                    chart_type = "bar"
            
            # 4. 判断是否问排名/前几名
            elif any(kw in query_lower for kw in ["前", "排名", "top"]):
                if real_numeric_cols:
                    col = real_numeric_cols[0]
                    nums = re.findall(r'\d+', query_lower)
                    n = int(nums[0]) if nums else 5
                    result_df = df.nlargest(n, col)[[col] + text_cols[:2]]
                    answer = f"🏆 {col}最高的{n}名"
                    insight = f"按{col}降序排列，显示前{n}名"
                    chart_type = "bar"
            
            # 5. 判断是否问总和
            elif any(kw in query_lower for kw in ["总和", "合计", "总共"]):
                if real_numeric_cols:
                    col = real_numeric_cols[0]
                    total = df[col].sum()
                    answer = f"💰 {col}总和：{total:,.0f}"
                    insight = f"{col}的总和为 {total:,.0f}"
                    result_df = pd.DataFrame([{"指标": f"{col}总和", "数值": f"{total:,.0f}"}])
            
            # 6. 判断是否问数量/人数
            elif any(kw in query_lower for kw in ["多少", "几个", "人数", "数量", "总数", "总计"]):
                total = len(df)
                answer = f"📊 数据共有 **{total}** 条记录"
                insight = f"当前数据共有 {total} 行"
                result_df = pd.DataFrame([{"统计项": "总记录数", "数值": total}])
            
            # 7. 判断是否问筛选某个部门/岗位
            elif any(dept in query_lower for dept in ["技术部", "产品部", "市场部", "销售部", "人力资源部", "财务部", "运营部", "人事部"]):
                for dept in ["技术部", "产品部", "市场部", "销售部", "人力资源部", "财务部", "运营部", "人事部"]:
                    if dept in query_lower and "部门" in df.columns:
                        result_df = df[df["部门"] == dept]
                        answer = f"📁 {dept}共有 **{len(result_df)}** 人"
                        insight = f"{dept}员工列表，共{len(result_df)}人"
                        st.dataframe(result_df)
                        break
            
            # 8. 默认：智能推荐
            else:
                suggestions = []
                if text_cols:
                    suggestions.append(f"「{text_cols[0]}分布」")
                if real_numeric_cols:
                    suggestions.append(f"「{real_numeric_cols[0]}前10」")
                    suggestions.append(f"「平均{real_numeric_cols[0]}」")
                if "部门" in df.columns:
                    suggestions.append("「技术部」")
                
                answer = f"💡 我暂时无法理解「{query}」，试试这些问题："
                insight = "智能推荐问题"
                for s in suggestions[:4]:
                    answer += f"\n   • {s}"
                result_df = pd.DataFrame([{"提示": "支持的问题类型", "示例": s} for s in suggestions[:5]])
            
            # ========== 显示结果 ==========
            if result_df is not None:
                st.subheader("📈 分析结果")
                st.markdown(f"### {answer}")
                
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    st.dataframe(result_df, use_container_width=True)
                
                with col2:
                    if chart_type == "pie" and len(result_df) >= 2:
                        fig = px.pie(result_df, names=result_df.columns[0], values=result_df.columns[1])
                        st.plotly_chart(fig, use_container_width=True)
                    elif chart_type == "bar" and len(result_df) >= 2:
                        fig = px.bar(result_df, x=result_df.columns[0], y=result_df.columns[1])
                        st.plotly_chart(fig, use_container_width=True)
                
                st.divider()
                st.subheader("🤖 AI 洞察")
                st.info(insight)
                
                st.caption("💡 试试继续问：部门分布、薪资排名、平均年龄")
else:
    st.info("👈 请先上传 Excel 或 CSV 文件开始分析")
    
    st.markdown("""
    ### 🚀 InsightFlow 功能亮点
    
    | 功能 | 说明 |
    |------|------|
    | 📁 **任意Excel** | 支持上传任何表格数据 |
    | 💬 **任何问题** | 随便问，AI都会给出反馈 |
    | 📊 **自动图表** | 智能匹配饼图/柱状图 |
    | 🤖 **AI洞察** | 每次分析都有智能总结 |
    | 🔒 **数据安全** | 所有处理在本地完成 |
    
    ### 💡 试试这样提问
    
    - 「部门分布」→ 显示各部门人数 + 饼图
    - 「薪资前10」→ 显示薪资最高的10人
    - 「平均年龄」→ 显示平均年龄
    - 「平均薪资」→ 显示平均薪资
    - 「技术部」→ 筛选技术部员工
    - 「哪个部门人最多」→ 智能分析
    """)
