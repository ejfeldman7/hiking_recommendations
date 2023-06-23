import streamlit as st
import pandas as pd
import numpy as np
from streamlit_chat import message
import os
import pickle
import sklearn
from sklearn.metrics import pairwise_distances
from geopy.geocoders import Nominatim
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import openai

from utils.processing import scaler
from utils.feature_lists import object_cols, numeric_cols, tag_cols, fewer_numeric, fewer_object
from utils.data import load_data


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
new_choice = ['Home','Visualizations','Filter','Recommender','Chat HikeGPT']

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
choice = st.sidebar.radio("go to",('Home','Visualizations','Filter','Recommender','Chat HikeGPT'), index=next_clicked)

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
    2. Use the __Visualization__ page to dig a little deeper in data about the hikes
    3. Use the __Filter__ page to sort through hikes by a variety of factors
    4. Use the __Recommender__ app to get a hike recommendation based on your favorite hike
    5. Use the __Chat HikeGTP__ page if you have an OpenAI API key to chat!
    \r\n
    This site was created by Ethan Feldman. You can find him on [GitHub](https://github.com/ejfeldman7), [LinkedIn](https://www.linkedin.com/in/feldmanethan/),
    [Medium/TDS](https://ethan-feldman.medium.com/) and on his [website](https://www.ejfeldman.com/)  \r\n
    '''
    
elif choice == 'Filter':
	df = load_data('./data/hikes.parquet', 'parquet')
	# st.dataframe(df)
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
	                 (df['General Region'].isin(selected_regions)) & 
	                 (df['Tags'].apply(lambda x: any(tag in selected_tags for tag in x)))]

	# Add a checkbox to show/hide the location input fields
	show_location_input = st.checkbox("Enter location")

	# Process the input and geocoder	
	if show_location_input:

		# Create a geocoder instance
		geolocator = Nominatim(user_agent="my_geocoder")

		# Add an input field for city and state or zip code
		location_input = st.text_input("Enter a city and state or a zip code:")

	# 	# Add an input field for the distance in kilometers
	# 	distance_input_miles = st.number_input("Enter the distance in miles:", min_value=0)

	# 	# Process the input and geocode
	# 	if location_input and distance_input_miles:
	# 		location = geolocator.geocode(location_input)
	# 	if location is not None:
	# 		distance_input_km = distance_input_miles * 1.60934
	# 		filtered_df = df[
	# 			df.apply(
	# 				lambda row: haversine(
	# 					(location.latitude, location.longitude),
	# 					(row["latitude"], row["longitude"]),
	# 					unit=Unit.MIL,
	# 				)
	# 				<= distance_input_km,
	# 				axis=1,
	# 			)
	# 		]
	# 	else:
	# 		st.write("Invalid location input")

	# Display filtered results
	st.dataframe(filtered_df)

elif choice == 'Visualizations':
	df = load_data('./data/hikes.parquet', 'parquet')
	
	data = df[['General Region','Specific Region','Rating','Votes','Length','Elevation Gain',
 		'One Way','Round Trip','Highest Point','Tags', 'Difficulty']]
	
	# Example: Radio buttons to choose the visualization type
	visualization_type = st.radio('Select visualization type', ['Frequency Plot/Histogram', 'Scatter Plot', 'Box Plot'])

	# Example: Selectbox to choose the feature
	# if visualization_type == 'Line Plot':
	# 	x = st.selectbox('Select the x-axis variable:', data.select_dtypes(include=['int', 'float']).columns)
	# 	y = st.selectbox('Select the y-axis variable:', data.select_dtypes(include=['int', 'float']).columns)
	# 	plt.plot(data[x], data[y])
	# 	plt.xlabel(x)
	# 	plt.ylabel(y)
	# 	st.pyplot()

	if visualization_type == 'Frequency Plot/Histogram':
		selected_feature = st.selectbox('Select a feature', data.select_dtypes(include=['int', 'float']).columns)
		# max_val = int(df[selected_feature].dropna().max().astype(str).str.split('.')[0])
		bin_size = st.slider('Select bin size', min_value=0, max_value=20, value=10)  # Adjust the min/max values as needed
		if selected_feature == 'Length':
			data = data[data[selected_feature] < 20]
			st.write('Limited visual to trails under 20 miles')
		elif selected_feature == 'Elevation Gain':
			data = data[data[selected_feature] < 5000]
			st.write('Limited visual to trails under 5k ft gained')
		freq, bins = np.histogram(data[selected_feature].dropna(), bins=bin_size)
		fig, ax = plt.subplots()
		ax.bar(bins[:-1], freq, width=np.diff(bins), align='edge')
		st.pyplot(fig)

	elif visualization_type == 'Scatter Plot':
		x = st.selectbox('Select the x-axis variable:', data.select_dtypes(include=['int', 'float']).columns)
		y = st.selectbox('Select the y-axis variable:', data.select_dtypes(include=['int', 'float']).columns)
		if x == 'Length' or y == 'Length':
			data = data[data[x] < 20]
			st.write('Limited visual to trails under 20 miles')
		elif x == 'Elevation Gain' or y == 'Elevation Gain':
			data = data[data[y] < 5000]
			st.write('Limited visual to trails under 5k ft gained')		
		plt.scatter(data[x], data[y])
		plt.xlabel(x)
		plt.ylabel(y)
		st.pyplot()

	elif visualization_type == 'Box Plot':
		selected_feature = st.selectbox('Select a feature', data.select_dtypes(include=['int', 'float']).columns)
		if selected_feature == 'Length':
			data = data[data[selected_feature] < 20]
			st.write('Limited visual to trails under 20 miles')
		elif selected_feature == 'Elevation Gain':
			data = data[data[selected_feature] < 5000]
			st.write('Limited visual to trails under 5k ft gained')		
		sns.boxplot(data=data, x=selected_feature)
		st.set_option('deprecation.showPyplotGlobalUse', False)
		st.pyplot()

elif choice == 'Recommender':
	df = load_data('./data/hikes.parquet', 'parquet')
	st.markdown("# Get a Hike Recommendation")

	# Create checkboxes for categorical fields
	st.markdown("#### Select any region in which you'd like to include in recommendations")
	regions = df['General Region'].unique()

	selected_regions = st.multiselect("Regions", regions, default=regions)
	st.markdown("##### First, type in the name of a hike and hit enter")
	st.markdown("##### Then, select the hike from the drop down")

	# Create a text entry field
	user_input = ''
	user_input = st.text_input('Type Here')

	# Filter the dataframe based on the user input
	df = df.dropna()
	filtered_df = df[df['Name'].str.contains(user_input, case=False)]

	# Extract the suggestions from the filtered dataframe
	suggestions = filtered_df['Name'].tolist()

	# Show the suggestions as a multiselect or selectbox widget
	if user_input == '': 
		st.write('''Excited to recommend a hike for you!''') 
	else:
		selected_suggestions = st.selectbox('Suggestions', suggestions)
		st.write(f'Searching for recommendation based off of {selected_suggestions}...')
		elevation_mult = st.slider('How important is having a similar elevation_gain?', 0, 5, 3)
		length_mult = st.slider('How important is having a similar length of hike?', 0, 5, 3)
		text_df = load_data('./data/text_data.parquet', 'parquet')
		hike_to_find = selected_suggestions
		text = [text_df[text_df.Name == hike_to_find]['cleaned_text'].iloc[0]]

		## Limit Recommendations By Hike Features
		# Select the input vector
		input_index = df[df.Name == hike_to_find].index[0]
		
		# Select the features to use for distance calculation , 
		features_df = scaler(df, scaler_type = 'MinMax', numeric_cols = fewer_numeric, object_cols = fewer_object, tag_cols = tag_cols)

		# Weight some features more heavily
		features_df['Length'] = features_df['Length'] * length_mult
		features_df['Elevation Gain'] = features_df['Elevation Gain'] * 2 * elevation_mult

		# Create the input vector from the transformed dataframe
		input_vector = features_df.iloc[input_index].values

		# Remove rows with null values
		features_df = features_df.dropna()
		encoded_regions = ['General Region_' + x for x in selected_regions]
		
		# Filter to selected region(s)
		# filtered_df = features_df[(features_df['General Region'].isin(selected_regions))] 
		filtered_df = features_df[(features_df[encoded_regions]==1).any(axis='columns')]

		# Calculate pairwise distances between the input vector and all records in the dataframe
		distances = pairwise_distances(input_vector.reshape(1, -1), filtered_df.values)

		# Get the indices of the 25 closest records
		closest_indices = np.argsort(distances.flatten())[1:10]

		# Get the closest records from the dataframe using the extracted index
		closest_records = df.iloc[closest_indices]

		# Filter NMF matrix to records above
		nmf_features = nmf_model.transform(tfidf_matrix)
		vt = vectorizer.transform(text).todense()
		tt1 = nmf_model.transform(np.asarray(vt))

		## Find Recommendations By Topic Model
		indices = pairwise_distances(tt1.reshape(1,-1),nmf_features[closest_indices],metric='cosine').argsort()
		recs = list(indices[0][0:6])

		if len(text_df.iloc[recs[0]]['Name'])>1:
			st.write('Based on your input hike, I recommend you try:',text_df.iloc[recs[0]]['Name'],'\n\n','It could be desribed as:','\n',text_df.iloc[recs[0]].Description)
			st.write('For more information, visit:',text_df.iloc[recs[0]]['Hike ID'])
			st.write('Or, another option I would recommend you try is:',text_df.iloc[recs[1]]['Name'],'\n\n','It could be desribed as:','\n',text_df.iloc[recs[1]].Description)
			st.write('For more information, visit:',text_df.iloc[recs[1]]['Hike ID'],'\n')
			st.write('Other choices include:', text_df.iloc[recs[2]]['Hike ID'],', ',text_df.iloc[recs[3]]['Hike ID'],', ',text_df.iloc[recs[4]]['Hike ID'])

elif choice == 'Chat HikeGPT':

	openai_api_key = st.text_input('OpenAI API Key',key='chatbot_api_key')
	
	st.title("HikeChat Streamlit GPT")
	#openai.api_key = st.secrets.openai_api_key
	if "messages" not in st.session_state:
		st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

	with st.form("chat_input", clear_on_submit=True):
		a, b = st.columns([4, 1])
		user_input = a.text_input(
			label="Your message:",
			placeholder="Ask me about finding a hike!",
			label_visibility="collapsed",
		)
		b.form_submit_button("Send", use_container_width=True)

	for msg in st.session_state.messages:
		message(msg["content"], is_user=msg["role"] == "user")

	if user_input and not openai_api_key:
		st.info("Please add your OpenAI API key to continue.")
	    
	if user_input and openai_api_key:
		openai.api_key = openai_api_key
		st.session_state.messages.append({"role": "user", "content": user_input})
		message(user_input, is_user=True)
		response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
		msg = response.choices[0].message
		st.session_state.messages.append(msg)
		message(msg.content)
