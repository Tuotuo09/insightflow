def detect_filter_from_query(query, df):
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
    if filter_col == "渠道来源":
        # 渠道付费金额柱状图
        if "累计付费金额" in df.columns:
            channel_pay = df.groupby("渠道来源")["累计付费金额"].sum().reset_index()
            if len(channel_pay) > 0:
                fig = px.bar(channel_pay, x="渠道来源", y="累计付费金额", title=f"筛选后：{filter_val}渠道付费金额分布")
                fig.update_traces(marker_color=PRIMARY_BLUE)
                fig.update_layout(title_font_size=14)
                return fig
    
    elif filter_col == "城市等级":
        if "累计付费金额" in df.columns:
            city_pay = df.groupby("城市等级")["累计付费金额"].sum().reset_index()
            if len(city_pay) > 0:
                fig = px.bar(city_pay, x="城市等级", y="累计付费金额", title=f"筛选后：{filter_val}城市付费金额分布")
                fig.update_traces(marker_color=PRIMARY_BLUE)
                fig.update_layout(title_font_size=14)
                return fig
    
    elif filter_col == "用户类型":
        if "累计付费金额" in df.columns:
            type_pay = df.groupby("用户类型")["累计付费金额"].sum().reset_index()
            if len(type_pay) > 0:
                fig = px.pie(type_pay, names="用户类型", values="累计付费金额", title=f"筛选后：{filter_val}用户付费分布")
                fig.update_traces(marker=dict(colors=px.colors.sequential.Blues_r))
                fig.update_layout(title_font_size=14)
                return fig
    
    elif filter_col == "留存状态":
        if "留存状态" in df.columns:
            retention_count = df["留存状态"].value_counts().reset_index()
            retention_count.columns = ["留存状态", "用户数"]
            if len(retention_count) > 0:
                fig = px.bar(retention_count, x="留存状态", y="用户数", title=f"筛选后：{filter_val}留存状态分布")
                fig.update_traces(marker_color=PRIMARY_BLUE)
                fig.update_layout(title_font_size=14)
                return fig
    
    return None
