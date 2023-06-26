# hiking_recommendations
Data from WTA website pulled to be used for filtering/recommending hikes via a Streamlit app 

## Process
Data was pulled from the [WTA website](https://www.wta.org/go-hiking/hikes/) using BeautifulSoup. This includes features available including hike length, elevation gain, highest point, region, tags, and a written description of the hike. 

This dataset was then minimally processed to create two parquet files. The first contains normalized numeric features and one hot encoded categorical features. The second is a TF-IDF corpus of the descriptions. 

Recommendations are them made by finding them closest hike to a user specified reference hike, comparing these using the cosine distance of the features in the first dataset to find similarly featured hikes and then narrowing down a final recommendation based on an NMF topic model vector of the description. Users can further filter this by limiting recommendations to particular regions.

## Deployment 
Datasets, NMF model, and recommendation pipeline were deployed using [Streamlit](https://ejfeldman7-hiking-recommendations-app-4kri38.streamlit.app/), along with a few additional pages for exploring hikes. 

