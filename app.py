import streamlit as st
import pandas as pd
import os
import pickle

# create a button in the side bar that will move to the next page/radio button choice
next = st.sidebar.button('Next on list')

# will use this list and next button to increment page, MUST BE in the SAME order
# as the list passed to the radio button
new_choice = ['Home','Filter','Recommender']

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
if next:
    #increment value to get to the next page
    next_clicked = next_clicked +1

    # check if you are at the end of the list of pages again
    if next_clicked == len(new_choice):
        next_clicked = 0 # go back to the beginning i.e. homepage

# create your radio button with the index that we loaded
choice = st.sidebar.radio("go to",('Home','Filter','Recommender'), index=next_clicked)

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
    This project was built from the amazing hiking resource of the [Washington Trails Association]. 
    The detailed hike descriptions were used to create nine-dimensional hike vectors using non-negative matrix factorization on a TF-IDF encoding of each coffee's review. This enabled comparison between coffees by their difference or similarity across the derived flavor spectrum.
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

	# Create sliders for numeric fields
	st.sidebar.markdown("### Numeric Filters")
	# Set the minimum and maximum values for the slider
	min_value = float(df['Length'].min())
	max_value = float(df['Length'].max())

	# Adjust the displayed value if it exceeds 100
	slider_max = max_value if slider_value[1] > 100 else slider_value[1]

	# Display the selected values
	start_value, end_value = st.sidebar.slider('Length Range', min_value, max_value, (min_value, max_value))

	# Display the selected Range	
	st.write('Selected Range:', slider_value[0], 'to', slider_max)
	# length_min = st.sidebar.slider("", ), float(df['Length'].max()), float(df['Length'].min()))
	# length_max = st.sidebar.slider("Maximum Length", float(df['Length'].min()), float(df['Length'].max()), float(df['Length'].max()))
	elevation_min = st.sidebar.slider("Minimum Elevation Gain", int(df['Elevation Gain'].min()), int(df['Elevation Gain'].max()), int(df['Elevation Gain'].min()))
	elevation_max = st.sidebar.slider("Maximum Elevation Gain", int(df['Elevation Gain'].min()), int(df['Elevation Gain'].max()), int(df['Elevation Gain'].max()))

	# Create checkboxes for categorical fields
	st.sidebar.markdown("### Categorical Filters")
	regions = df['General Region'].unique()
	selected_regions = st.sidebar.multiselect("Regions", regions, default=regions)
	tags = df['Tags'].explode().unique()
	selected_tags = st.sidebar.multiselect("Tags", tags, default=tags)

	# Apply filters
	filtered_df = df[(df['Length'] >= length_min) & (df['Length'] <= length_max) &
	                 (df['Elevation Gain'] >= elevation_min) & (df['Elevation Gain'] <= elevation_max) &
	                 (df['General Region'].isin(selected_regions)) & (df['Tags'].apply(lambda x: any(tag in selected_tags for tag in x)))]

	# Display filtered results
	st.dataframe(filtered_df)

elif choice == 'Recommender':
	df = pd.read_parquet('./data/hikes.parquet')

	# Create a text entry field
	user_input = st.text_input('Type in the Name of a Hike You Have Enjoyed')

	# Filter the dataframe based on the user input
	filtered_df = df[df['Name'].str.contains(user_input, case=False)]

	# Extract the suggestions from the filtered dataframe
	suggestions = filtered_df['Name'].tolist()

	# Show the suggestions as a multiselect or selectbox widget
	selected_suggestions = st.multiselect('Suggestions', suggestions)

	# You can access the selected suggestions using 'selected_suggestions' variable
	st.write('Selected Suggestions:', selected_suggestions)
