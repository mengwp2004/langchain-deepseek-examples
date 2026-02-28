import streamlit as st
import pandas as pd

# 1. 添加一个标题
st.title("我的第一个 Streamlit App")

# 2. 创建一个滑块组件
x = st.slider("选择一个数字")

# 3. 显示计算结果
st.write(x, "的平方是", x * x)

# 4. 创建一个数据表格
df = pd.DataFrame({
    '第一列': [1, 2, 3, 4],
    '第二列': [10, 20, 30, 40]
})
st.write("这是一个数据表格：")
st.dataframe(df)
