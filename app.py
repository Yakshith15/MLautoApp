# from operator import index
# Before
# from pydantic import BaseSettings

# After
# from pydantic_settings import BaseSettings


import streamlit as st
# import plotly.express as px
from pycaret.regression import setup, compare_models, pull, save_model, load_model
# from pydantic_settings import BaseSettings
# import pandas_profiling
import pandas as pd
# from pandas_profiling import ProfileReport
# import streamlit_pandas_profiling
# from streamlit_pandas_profiling import st_profile_report
import os 

if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar: 
    st.image("https://www.google.com/url?sa=i&url=https%3A%2F%2Funsplash.com%2Fs%2Fphotos%2Fmachine-learning&psig=AOvVaw1eJjK4LENtLEWiLNUhkJAh&ust=1706537129585000&source=images&cd=vfe&opi=89978449&ved=0CBMQjRxqFwoTCPjSkLCggIQDFQAAAAAdAAAAABAD")
    st.title("MLApp")
    choice = st.radio("Navigation", ["Upload","Profiling","Modelling", "Download"])
    st.info("This project application helps you build and explore your data.")

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

# if choice == "Profiling": 
#     st.title("Exploratory Data Analysis")
#     profile_df = df.profile_report()
#     st_profile_report(profile_df)

# if choice == "Modelling": 
#     chosen_target = st.selectbox('Choose the Target Column', df.columns)
#     if st.button('Run Modelling'): 
#         setup(df, target=chosen_target, )
#         setup_df = pull()
#         st.dataframe(setup_df)
#         best_model = compare_models()
#         compare_df = pull()
#         st.dataframe(compare_df)
#         save_model(best_model, 'best_model')
        
# if choice == "Modelling": 
#     chosen_target = st.selectbox('Choose the Target Column', df.columns)
#     if st.button('Run Modelling'): 
#         # Handle categorical columns
#         df_encoded = pd.get_dummies(df)  # Use appropriate encoding method

#         # Handle missing values
#         df_encoded.dropna(inplace=True)

#         setup(df_encoded, target=chosen_target)
#         setup_df = pull()
#         st.dataframe(setup_df)

#         best_model = compare_models()
#         compare_df = pull()
#         st.dataframe(compare_df)

#         save_model(best_model, 'best_model')
        
if choice == "Modelling": 
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    if st.button('Run Modelling'): 
        # Handle categorical columns
        df_encoded = pd.get_dummies(df)  # Use appropriate encoding method

        # Handle missing values
        df_encoded.dropna(inplace=True)

        # Check the dataset before calling setup
        st.dataframe(df_encoded.head())

        # Verify the target column
        if chosen_target not in df_encoded.columns:
            st.error(f"Target column '{chosen_target}' not found in the dataset.")
        else:
            setup(df_encoded, target=chosen_target)
            setup_df = pull()
            st.dataframe(setup_df)

            best_model = compare_models()
            compare_df = pull()
            st.dataframe(compare_df)

            save_model(best_model, 'best_model')



# if choice == "Download": 
#     with open('best_model.pkl', 'rb') as f: 
#         st.download_button('Download Model', f, file_name="best_model.pkl")

# import streamlit as st

# st.write("Hello world!")