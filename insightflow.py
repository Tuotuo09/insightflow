import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
import re

st.set_page_config(page_title="InsightFlow", layout="wide")

st.title("📊 InsightFlow")
st.caption("智能数据洞察助手 · 上传Excel，用自然语言提问")

uploaded_file = st.file_uploader("上传数据文件", type=['xlsx', 'xls', 'csv'])

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    st.subheader("数据预览")
    st.dataframe(df.head(100))
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    st.info(f"📊 共 {len(df)} 行，{len(df.columns)} 列")
    st.write(f"📈 数值列：{', '.join(numeric_cols[:5])}")
    st.write(f"📝 文本列：{', '.join(text_cols[:5])}")
    
    st.divider()
    st.subheader("💬 自然语言查询")
    
    query = st.text_input("输入问题", placeholder="例如：部门分布、薪资前10、平均年龄")
    
    if st.button("分析", type="primary"):
        with st.spinner("分析中..."):
            query_lower = query.lower()
            
            if "分布" in query_lower and text_cols:
                col = text_cols[0]
                result = df[col].value_counts().reset_index()
                result.columns = [col, "数量"]
                st.dataframe(result)
                fig = px.pie(result, names=col, values="数量")
                st.plotly_chart(fig)
                st.success(f"📊 {col}分布完成")
            
            elif "前" in query_lower and numeric_cols:
                col = numeric_cols[0]
                n = 5
                result = df.nlargest(n, col)[[col] + text_cols[:2]]
                st.dataframe(result)
                fig = px.bar(result, x=text_cols[0], y=col)
                st.plotly_chart(fig)
                st.success(f"🏆 {col}最高的{n}条")
            
            elif "平均" in query_lower and numeric_cols:
                col = numeric_cols[0]
                avg_val = df[col].mean()
                st.metric(f"平均{col}", f"{avg_val:.2f}")
                st.success(f"📈 平均{col}：{avg_val:.2f}")
            
            else:
                st.warning("试试这些：'部门分布'、'薪资前10'、'平均年龄'")
else:
    st.info("👈 请先上传 Excel 文件")