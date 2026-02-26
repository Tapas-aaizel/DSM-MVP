
import sys
import os
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import re
import json

# Add AWS-Data to path to potentially import, but let's just copy the relevant scrape logic 
# to avoid strict dependency issues and just test the site content.

BASE_URL = "https://www.weather-india.in"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0"}

def get_station_link():
    # Find one station in Rajasthan/Gujarat to test
    url = "https://www.weather-india.in/en/india/rajasthan/jaipur/weather-hourly"
    return url

def test_station_data(url):
    print(f"Testing URL: {url}")
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(r.content, 'html.parser')
        
        # Debug: Check for any hourly data container
        # weather.py looked for 'values-parent'
        
        # Look for typical weather keywords in classes
        print("Searching for weather data containers...")
        
        # Try finding 'hourly' related divs
        hour_divs = soup.find_all('div', class_=re.compile(r'hourly'))
        if hour_divs:
             print(f"Found {len(hour_divs)} divs with 'hourly' in class.")
             
        # Just Dump the first few IDs or Classes of main divs
        print("Top Level Classes found:")
        for div in soup.find_all('div', class_=True, limit=20):
             print(f" - {div.get('class')}")

        h_cont = soup.find_all('div', class_='values-parent')
        print(f"Found {len(h_cont)} daily tabs/containers.")
        
        found_dates = []
        for idx, cont in enumerate(h_cont):
            # Code from weather.py logic
            date_obj = datetime.now().date() + timedelta(days=idx)
            
            # Check if there's actual data
            hrs = cont.find_all('div', class_=re.compile(r'\bhour\b'))
            if hrs:
                found_dates.append(date_obj.strftime('%Y-%m-%d'))
                
        return found_dates
    except Exception as e:
        print(f"Error: {e}")
        return []

if __name__ == "__main__":
    url = get_station_link()
    dates = test_station_data(url)
    print("\n--- RESUTLS ---")
    print(f"Available Dates detected from scraping logic (Today+Index):")
    for d in dates:
        print(f" - {d}")
        
    if dates:
        print(f"\nTotal: {len(dates)} days of data available starting from {dates[0]}.")
    else:
        print("No data found.")
