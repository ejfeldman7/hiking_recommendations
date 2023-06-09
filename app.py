import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import sklearn
from sklearn.metrics import pairwise_distances
from geopy.geocoders import Nominatim
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

from .utils.processing import scaler
from .utils.feature_lists import object_cols, numeric_cols, tags_cols


with open('./pickle_barrel/vectorizer.pkl', 'rb') as read_file:
    vectorizer = pickle.load(read_file)
with open('./pickle_barrel/tfidf_matrix.pkl', 'rb') as read_file:
    tfidf_matrix = pickle.load(read_file)
with open('./pickle_barrel/nmf_model.pkl', 'rb') as read_file:
    nmf_model = pickle.load(read_file)

# create a button in the side bar that will move to the next page/radio button choice
# next = st.sidebar.button('Next on list')

# will use this list and next button to increment page, MUST BE in the SAME order
# as the list passed to the radio button
new_choice = ['Home','Visualizations','Filter','Recommender']

# This is what makes this work, check directory for a pickled file that contains
# the index of the page you want displayed, if it exists, then you pick up where the
#previous run through of your Streamlit Script left off,
# if it's the first go it's just set to 0
if os.path.isfile('next.p'):
    next_clicked = pickle.load(open('next.p', 'rb'))
    # check if you are at the end of the list of pages
    if next_clicked == len(new_choice):
        next_clicked = 0 # go back to the beginning i.e. homepage
else:
    next_clicked = 0 #the start

# this is the second tricky bit, check to see if the person has clicked the
# next button and increment our index tracker (next_clicked)
# if next:
#     #increment value to get to the next page
#     next_clicked = next_clicked +1

#     # check if you are at the end of the list of pages again
#     if next_clicked == len(new_choice):
#         next_clicked = 0 # go back to the beginning i.e. homepage

# create your radio button with the index that we loaded
choice = st.sidebar.radio("go to",('Home','Visualizations','Filter','Recommender'), index=next_clicked)

st.sidebar.write(
    '''
    __About__ \n
    This project was built from the amazing hiking resource of the [Washington Trails Association](https://www.wta.org/go-outside/hikes). Specs from the hikes and NMF vectors from the descriptions are used to compare hikes. 
    \n
    This site was created by Ethan Feldman. You can find him on [GitHub](https://github.com/ejfeldman7), [LinkedIn](https://www.linkedin.com/in/feldmanethan/), [Medium/TDS](https://ethan-feldman.medium.com/) and his [website](ejfeldman.com)!
    ''')
# pickle the index associated with the value, to keep track if the radio button has been used
pickle.dump(new_choice.index(choice), open('next.p', 'wb'))

# finally get to whats on each page
if choice == 'Home':
    st.title('Welcome to a prototype tool for finding your next hike in Washington!')
    '''
    This project was built from the amazing hiking resource of the Washington Trails Association: https://www.wta.org/go-outside/hikes. 
    The detailed hike descriptions were used to create nine-dimensional hike vectors using non-negative matrix factorization on a TF-IDF encoding of each hike's description. This enabled comparison between hikes by their difference or similarity across characteristics.
    These vectors and additional features were then used for recommendations of hikes with similar vectors, predicting scores, and more.  \r\n
    __On the side bar__ on the left you will find a few different application  \r\n
    __Below__ is a quick table of contents for the different pages of the site
    '''
    '''
    1. This is the __Home Page__
    2. Use the __Filter__ page to sort through hikes by a variety of factors
    3. Use the __Recommender__ app to get a hike recommendation based on your favorite hike
    \r\n
    This site was created by Ethan Feldman. You can find him on [GitHub](https://github.com/ejfeldman7), [LinkedIn](https://www.linkedin.com/in/feldmanethan/),
    [Medium/TDS](https://ethan-feldman.medium.com/) and on his [website](https://www.ejfeldman.com/)  \r\n
    '''
    
elif choice == 'Filter':

	df = pd.read_parquet('./data/hikes.parquet')
	st.markdown("##### Filter by the Option Below to Reveal Potential Hikes")
	# Create sliders for numeric fields
	st.markdown("##### Numeric Filters")
	# Set the minimum and maximum values for the slider
	min_length = float(df['Length'].min())
	max_length = float(15.0) #float(df['Length'].max())

	# Display the selected values
	length_value = st.slider('Length Range (0 to 15+ mi)', min_length, max_length, (min_length, max_length), format='%.1f')

	# Adjust the displayed value if it exceeds 15
	length_max = length_value[1] if length_value[1] < 15 else float(1200.0)

	# Display the selected Range	
	# st.write('Selected Hike Length Range (mi):', length_value[0], 'to', length_max)

	# Set the minimum and maximum values for the slider
	min_gain = float(df['Elevation Gain'].min())
	max_gain = float(10000) #float(df['Length'].max())

	# Display the selected values
	gain_value = st.slider('Elevation Gain Range (0 to 10k+ ft)', min_gain, max_gain, (min_gain, max_gain), format='%.1f')

	# Adjust the displayed value if it exceeds 10000
	gain_max = gain_value[1] if gain_value[1] < 10000 else float(27000.0)

	# Display the selected Range	
	# st.write('Selected Elevation Gain Range (ft):', gain_value[0], 'to', gain_max)

	# Set the minimum and maximum values for the slider
	min_rating = float(0)
	max_rating = float(5) #float(df['Length'].max())

	# Display the selected values
	rating_value = st.slider('Star Rating Range (0 to 5)', min_rating, max_rating, (min_rating, max_rating), format='%.1f')

	# Display the selected Range	
	# st.write('Selected Star Rating Range:', rating_value[0], 'to', rating_value[1])
	
	# Create checkboxes for categorical fields
	st.markdown("##### Categorical Filters")
	regions = df['General Region'].unique()
	selected_regions = st.multiselect("Regions", regions, default=regions)
	tags = df['Tags'].explode().unique()
	selected_tags = st.multiselect("Tags", tags, default=tags)
	difficulty = df['Difficulty'].unique()
	selected_difficulty = st.multiselect("Difficulty", difficulty, default=difficulty)


	# Apply filters
	filtered_df = df[(df['Length'] >= length_value[0]) & (df['Length'] <= length_max) &
	                 (df['Elevation Gain'] >= gain_value[0]) & (df['Elevation Gain'] <= gain_max) &
	                 (df['Rating'] >= rating_value[0]) & (df['Rating'] <= rating_value[1]) &
	                 (df['Difficulty'].isin(selected_difficulty)) &
	                 (df['General Region'].isin(selected_regions)) & (df['Tags'].apply(lambda x: any(tag in selected_tags for tag in x)))]

	# Add a checkbox to show/hide the location input fields
	show_location_input = st.checkbox("Enter location")

	# Process the input and geocoder	
	# if show_location_input:

		# # Create a geocoder instance
		# geolocator = Nominatim(user_agent="my_geocoder")

		# # Add an input field for city and state or zip code
		# location_input = st.text_input("Enter a city and state or a zip code:")

		# # Add an input field for the distance in kilometers
		# distance_input_miles = st.number_input("Enter the distance in miles:", min_value=0)

		# # Process the input and geocode
		# if location_input and distance_input_miles:
		# 	location = geolocator.geocode(location_input)
		# if location is not None:
		# 	distance_input_km = distance_input_miles * 1.60934
		# 	filtered_df = df[
		# 		df.apply(
		# 			lambda row: haversine(
		# 				(location.latitude, location.longitude),
		# 				(row["latitude"], row["longitude"]),
		# 				unit=Unit.MIL,
		# 			)
		# 			<= distance_input_km,
		# 			axis=1,
		# 		)
		# 	]
		# else:
		# 	st.write("Invalid location input")

	# Display filtered results
	st.dataframe(filtered_df)

elif choice == 'Visualizations':
	df = pd.read_parquet('./data/hikes.parquet')
	
	data = df[['General Region','Specific Region','Rating','Votes','Length','Elevation Gain',
 		'One Way','Round Trip','Highest Point','Tags', 'Difficulty']]
	
	# Example: Radio buttons to choose the visualization type
	visualization_type = st.radio('Select visualization type', ['Bar Plot', 'Scatter Plot', 'Box Plot'])

	# Example: Selectbox to choose the feature
	# if visualization_type == 'Line Plot':
	# 	x = st.selectbox('Select the x-axis variable:', data.select_dtypes(include=['int', 'float']).columns)
	# 	y = st.selectbox('Select the y-axis variable:', data.select_dtypes(include=['int', 'float']).columns)
	# 	plt.plot(data[x], data[y])
	# 	plt.xlabel(x)
	# 	plt.ylabel(y)
	# 	st.pyplot()

	if visualization_type == 'Bar Plot':
		selected_feature = st.selectbox('Select a feature', data.select_dtypes(include=['int', 'float']).columns)
		freq = data[selected_feature].value_counts()
		st.bar_chart(freq)

	elif chart_type == 'Scatter Plot':
		x = st.selectbox('Select the x-axis variable:', data.select_dtypes(include=['int', 'float']).columns)
		y = st.selectbox('Select the y-axis variable:', data.select_dtypes(include=['int', 'float']).columns)

		plt.scatter(df[x], df[y])
		plt.xlabel(x)
		plt.ylabel(y)
		st.pyplot()

	elif chart_type == 'Box Plot':
		selected_feature = st.selectbox('Select a feature', data.select_dtypes(include=['int', 'float']).columns)
		sns.boxplot(data=data, x=feature)
		st.pyplot()

elif choice == 'Recommender':
	df = pd.read_parquet('./data/hikes.parquet')
	st.markdown("# Get a Hike Recommendation - Similar Descriptions")
	st.markdown("####**Not Using Hike Stats Yet (length, elevation, etc)**")
	st.markdown("##### First, type in the name of a hike and hit enter")
	st.markdown("##### Then, select the hike from the drop down")

	# Create a text entry field
	user_input = ''
	user_input = st.text_input('Type Here')

	# Filter the dataframe based on the user input
	filtered_df = df[df['Name'].str.contains(user_input, case=False)]

	# Extract the suggestions from the filtered dataframe
	suggestions = filtered_df['Name'].tolist()

	# Show the suggestions as a multiselect or selectbox widget
	if user_input == '': 
		st.write('''Excited to recommend a hike for you!''') 
	else:
		selected_suggestions = st.selectbox('Suggestions', suggestions)
		st.write(f'Searching for recommendation based off of {selected_suggestions}...')
		text_df = pd.read_parquet('./data/text_data.parquet')
		hike_to_find = selected_suggestions
		text = [text_df[text_df.Name == hike_to_find]['cleaned_text'].iloc[0]]

		## Limit Recommendations By Hike Features
		# Select the input vector
		input_index = df[df.Name == hike_to_find].index[0]

		# Select the features to use for distance calculation
		features_df = scaler(df, scaler_type = 'MinMax', numeric_cols, object_cols, tags_cols)

		# Create the input vector from the transformed dataframe
		input_vector = features_df.iloc[input_index]

		# Calculate pairwise distances between the input vector and all records in the dataframe
		distances = pairwise_distances(features_df[features].values, input_vector.values)

		# Get the indices of the 25 closest records
		closest_indices = np.argsort(distances.flatten())[1:25]

		# Get the closest records from the dataframe
		closest_records = df.iloc[closest_indices]

		# Filter NMF matrix to records above
		nmf_features = nmf_model.transform(tfidf_matrix)
		vt = vectorizer.transform(text).todense()
		tt1 = nmf_model.transform(np.asarray(vt))

		## Find Recommendations By Topic Model
		indices = pairwise_distances(tt1.reshape(1,-1),nmf_features[closest_indices],metric='cosine').argsort()
		recs = list(indices[0][1:4])

		if len(text_df.iloc[recs[0]]['Name'])>1:
			st.write('Based on your input hike, I recommend you try:','\n\n',text_df.iloc[recs[0]]['Name'],'\n\n','It could be desribed as:','\n\n',text_df.iloc[recs[0]].Description)
			st.write('For more information, visit:','\n\n',text_df.iloc[recs[0]]['Hike ID'])

			st.write('Or, another option I would recommend you try is:','\n\n',text_df.iloc[recs[1]]['Name'],'\n\n','It could be desribed as:','\n\n',text_df.iloc[recs[1]].Description)
			st.write('For more information, visit:','\n\n',text_df.iloc[recs[1]]['Hike ID'])

