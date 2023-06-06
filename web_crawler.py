import requests
import pandas as pd
from bs4 import BeautifulSoup
import sqlite3

counter = 0

def get_hike_details(url):
  hike_id =  soup.select_one('.listing-image')['data-hikeuid']
  name = soup.select_one('.listitem-title a span').text.strip()
  region_div = soup.select_one('.region')

  region_text = region_div.text.strip()
  general_region, specific_region = map(str.strip, region_text.split('>'))

  rating_div = soup.select_one('.hike-rating .star-rating')
  rating = float(rating_div.select_one('.current-rating')['style'].split(':')[1].strip('%;')) / 100
  votes = int(soup.select_one('.rating-count').text.strip().split()[0][1:])

  length_text = soup.select_one('.hike-length dd').text.strip()
  length, length_type = map(str.strip, length_text.split(','))

  elevation_gain_text = soup.select_one('.hike-gain dd').text.strip()
  elevation_gain, elevation_gain_unit = map(str.strip, elevation_gain_text.split())

  highest_point_text = soup.select_one('.hike-highpoint dd').text.strip()
  highest_point, highest_point_unit = map(str.strip, highest_point_text.split())

  out_and_back = bool(soup.select_one('.route-icon.out-and-back-icon'))
  round_trip = bool(soup.select_one('.route-icon.loop-icon'))

  tags = set([tag.select_one('.wta-icon__label').text.strip() for tag in soup.select('.wta-icon-list li')])

  review = soup.select_one('.listing-summary').text.strip()

  hike_data = {
          'Hike ID': hike_id,
          'Name': name,
          'General Region': general_region,
          'Specific Region': specific_region,
          'Rating': rating,
          'Votes': votes,
          'Length': length,
          'Length Type': length_type,
          'Elevation Gain': elevation_gain,
          'Elevation Gain Unit': elevation_gain_unit,
          'Out and Back': out_and_back,
          'Round Trip': round_trip,
          'Highest Point': highest_point,
          'Highest Point Unit': highest_point_unit,
          'Tags': tags,
          'Review': review
      }
  return hike_data


def crawl_wta_hikes(counter):
    if counter % 50 == 0:
      print(counter)
    base_url = 'https://www.wta.org/go-outside/hikes'
    hike_urls = []
    hike_data = []

    # Fetch all hike URLs from pagination
    page = 1
    while True:
        url = f'{base_url}?page={page}'
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        hikes = soup.select('.listitem-title a')[:4]
        if not hikes:
            break
        hike_urls.extend([hike['href'] for hike in hikes])
        page += 1

    # Fetch hike details from each URL
    for url in hike_urls:
        hike_data.append(get_hike_details(url))
    counter += 1
    return hike_data

# Run the crawler and store the results in a DataFrame
hike_data = crawl_wta_hikes(counter)
df = pd.DataFrame(hike_data)

conn = sqlite3.connect('hikes.db')
df.to_sql('hikes', conn, if_exists='replace', index=False)
