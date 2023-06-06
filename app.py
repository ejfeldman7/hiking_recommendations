import streamlit as st
import pandas as pd
import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect('https://github.com/ejfeldman7/hiking_recommendations/raw/main/hikes.db')

# Query the database
query = "SELECT * FROM hikes"
df = pd.read_sql_query(query, conn)

# Create sliders for numeric fields
st.sidebar.markdown("### Numeric Filters")
length_min = st.sidebar.slider("Minimum Length", float(df['Length'].min()), float(df['Length'].max()), float(df['Length'].min()))
length_max = st.sidebar.slider("Maximum Length", float(df['Length'].min()), float(df['Length'].max()), float(df['Length'].max()))
elevation_min = st.sidebar.slider("Minimum Elevation Gain", int(df['Elevation Gain'].min()), int(df['Elevation Gain'].max()), int(df['Elevation Gain'].min()))
elevation_max = st.sidebar.slider("Maximum Elevation Gain", int(df['Elevation Gain'].min()), int(df['Elevation Gain'].max()), int(df['Elevation Gain'].max()))

# Create checkboxes for categorical fields
st.sidebar.markdown("### Categorical Filters")
regions = df['Region'].unique()
selected_regions = st.sidebar.multiselect("Regions", regions, default=regions)
tags = df['Tags'].explode().unique()
selected_tags = st.sidebar.multiselect("Tags", tags, default=tags)

# Apply filters
filtered_df = df[(df['Length'] >= length_min) & (df['Length'] <= length_max) &
                 (df['Elevation Gain'] >= elevation_min) & (df['Elevation Gain'] <= elevation_max) &
                 (df['Region'].isin(selected_regions)) & (df['Tags'].apply(lambda x: any(tag in selected_tags for tag in x)))]

# Display filtered results
st.dataframe(filtered_df)

# Close the database connection
conn.close()
