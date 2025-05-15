import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# App Title and Configuration
st.set_page_config(page_title="Data Science Portfolio Project", layout="wide")

st.title("📊 Data Science Project Showcase")

# Sidebar Navigation
st.sidebar.title("📂 Navigation")
pages = ["Project Overview", "Dataset Explorer", "Visualization", "Modeling", "Conclusion"]
page = st.sidebar.radio("Go to", pages)

# Global DataFrame container
@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df

# Page 1: Project Overview
if page == "Project Overview":
    st.subheader("📌 Overview")
    st.markdown("""
    Welcome to my data science portfolio project!  
    This app demonstrates my ability to explore data, build machine learning models, and create insightful visualizations.

    **Project Objectives**:
    - Data Exploration & Cleaning
    - Exploratory Data Analysis (EDA)
    - Model Building & Evaluation
    - Drawing Conclusions

    *Please use the sidebar to navigate through the different stages of the project.*
    """)

# Page 2: Dataset Explorer
elif page == "Dataset Explorer":
    st.subheader("📁 Dataset Explorer")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.write("### First 5 Rows of the Data")
        st.dataframe(df.head())

        st.write("### Data Summary")
        st.write(df.describe())

        st.write("### Data Types")
        st.write(df.dtypes)

        st.write("### Missing Values")
        st.write(df.isnull().sum())

# Page 3: Visualization
elif page == "Visualization":
    st.subheader("📈 Exploratory Data Analysis")

    if 'df' in locals():
        col1, col2 = st.columns(2)

        with col1:
            selected_col = st.selectbox("Choose a column for histogram", df.select_dtypes(include=['int64', 'float64']).columns)
            fig = px.histogram(df, x=selected_col)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            num_cols = df.select_dtypes(include=['int64', 'float64']).columns
            if len(num_cols) >= 2:
                x_axis = st.selectbox("X-axis", num_cols, index=0)
                y_axis = st.selectbox("Y-axis", num_cols, index=1)
                fig2 = px.scatter(df, x=x_axis, y=y_axis)
                st.plotly_chart(fig2, use_container_width=True)

        st.write("### Correlation Heatmap")
        fig3, ax = plt.subplots()
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig3)
    else:
        st.warning("Please upload a dataset in the 'Dataset Explorer' section.")

# Page 4: Modeling
elif page == "Modeling":
    st.subheader("🤖 Machine Learning Model")

    if 'df' in locals():
        target = st.selectbox("Select the target column", df.columns)
        features = st.multiselect("Select features", [col for col in df.columns if col != target])

        if features and target:
            X = df[features]
            y = df[target]

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.write("### Classification Report")
            st.text(classification_report(y_test, y_pred))
        else:
            st.info("Please select both features and a target to build the model.")
    else:
        st.warning("Please upload a dataset in the 'Dataset Explorer' section.")

# Page 5: Conclusion
elif page == "Conclusion":
    st.subheader("📌 Conclusion")
    st.markdown("""
    Thank you for reviewing this project!

    **Key Takeaways**:
    - Demonstrated end-to-end data science workflow
    - Built interactive visualizations
    - Applied machine learning modeling

    **Next Steps**:
    - Improve model performance with tuning
    - Try different algorithms
    - Deploy with Streamlit Cloud or HuggingFace Spaces

    *Feel free to reach out for collaboration or questions!*
    """)
