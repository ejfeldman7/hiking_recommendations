import pandas as pd
import streamlit as st

def load_data(filepath, file_type):
	@st.cache_data
	if file_type == 'parquet':
		df = pd.read_parquet(filepath)
	elif file_type == 'csv':
		df = pd.read_csv(filepath)
	return df
