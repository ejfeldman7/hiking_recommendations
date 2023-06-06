import sqlite3
import requests
import pandas as pd
from bs4 import BeautifulSoup
import json

def get_hike_details(url):
  response = requests.get(url)
  soup = BeautifulSoup(response.content, 'html.parser')
  script_tag = soup.select_one('script[type="application/ld+json"]')
  ld_json = json.loads(script_tag.string)

  hike_id = ld_json['url']
  name = ld_json['name']

  rating = ld_json['aggregateRating']['ratingValue']
  votes = ld_json['aggregateRating']['ratingCount']

  description = ld_json['description']
  sidebar = soup.select_one('.wta-sidebar-layout__sidebar')
  if sidebar:
      sidebar.extract()

  hike_body_text = soup.select_one('#hike-body-text')
  # Remove the sidebar section from between the paragraphs
  if hike_body_text:
      for sidebar_element in hike_body_text.select('.sidebar'):
          sidebar_element.extract()
  # full_description = hike_body_text.get_text(separator='\n')
  paragraphs = hike_body_text.find_all('p')
  full_description = ' '.join([p.get_text(strip=True) for p in paragraphs])

  latitude = ld_json['geo']['latitude']
  longitude = ld_json['geo']['longitude']
  region_div = soup.select_one('.wta-icon-headline__text')
  region_text = region_div.text.strip()
  general_region, specific_region = map(str.strip, region_text.split('>'))

  # Extract hike details
  hike_stats = soup.select('.hike-stats__stat')
  length = hike_stats[0].dd.get_text(strip=True)
  one_way = 'one way' in length
  round_trip = 'roundtrip' in length
  length = float(length.split(' ')[0])
  elevation_gain = hike_stats[1].dd.get_text(strip=True)
  elevation_gain = float(elevation_gain.split(' ')[0].replace(',',''))
  highest_point = hike_stats[2].dd.get_text(strip=True)
  highest_point = float(highest_point.split(' ')[0].replace(',',''))

  tags = set([tag.select_one('.wta-icon__label').text.strip() for tag in soup.select('.wta-icon-list li')])

  difficulty = soup.select_one('.wta-pill').text.strip()

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
          'Description': description,
          'Full Description' : full_description,
          'Difficulty' : difficulty
      }

  return hike_data

def get_urls(0):
    base_url = 'https://www.wta.org/go-outside/hikes'
    hike_urls = []
    hike_data = []

    # Fetch all hike URLs from pagination
    page = 1
    while True:
        if page == 1:
          url = base_url
        else:
          url = f'{base_url}?b_start:int={page*30}'
        page += 1
        
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        hikes = soup.select('.listitem-title a')
        if not hikes:
            break
        hike_urls.extend([hike['href'] for hike in hikes])
        counter += 30
        if counter % 90 == 0:
          print(counter)
        if counter - 30 > len(hike_urls):
          break

def crawl_wta_hikes(hike_urls, counter):
    # Fetch hike details from each URL
    for url in hike_urls:
      hike_data.append(get_hike_details(url))
      counter +=1
      if counter % 100 == 0:
          print(counter)
    return hike_data

# Run the crawler and store the results in a DataFrame
hike_urls = get_urls(counter)
with open('urls.txt', 'w') as f:
    for line in hike_urls:
        f.write(f"{line}\n")

hike_data = crawl_wta_hikes(hike_urls, 0)
df = pd.DataFrame(hike_data)

conn = sqlite3.connect('hikes.db')
df.to_sql('hikes', conn, if_exists='replace', index=False)
df.to_parquet('hikes.parquet')
