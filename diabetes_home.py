import streamlit as st

def app(diabetes_df):
    st.markdown(
        "<p style='color:red'>Diabetes is a chronic (long-lasting) health condition that affects how your body turns food into energy."
        "There isn’t a cure yet for diabetes, but losing weight, eating healthy food, and being active can really help in reducing the impact of diabetes."
        "This Web app will help you to predict whether a person has diabetes or is prone to get diabetes in future by analysing the values of several features using the Decision Tree Classifier.",
        unsafe_allow_html=True)
    with st.expander('View Database'):
        st.dataframe(diabetes_df)
    st.subheader('Columns description')
    col_name, col_dtype, col_display = st.columns(3)
    if col_name.checkbox('Show all column names'):
        st.table(diabetes_df.columns)
    if col_dtype.checkbox('View column datatype'):
        st.write(list(diabetes_df.dtypes))
    if col_display.checkbox('View Column data'):
        col = st.selectbox('Select columns', diabetes_df.columns)
        st.table(diabetes_df[col])