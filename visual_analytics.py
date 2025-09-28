# Install required packages (run in terminal once)
# pip install folium pandas

import folium
import pandas as pd
from folium.plugins import MarkerCluster
from folium.features import DivIcon

# Load the dataset directly with pandas
URL = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/spacex_launch_geo.csv'
spacex_df = pd.read_csv(URL)

# Select relevant columns
spacex_df = spacex_df[['Launch Site', 'Lat', 'Long', 'class']]

# Get unique launch sites
launch_sites_df = spacex_df.groupby('Launch Site', as_index=False).first()[['Launch Site', 'Lat', 'Long']]

# Start location: NASA Johnson Space Center
nasa_coordinate = [29.559684888503615, -95.0830971930759]

# Create Folium map
site_map = folium.Map(location=nasa_coordinate, zoom_start=5)

# Add NASA JSC marker
circle = folium.Circle(
    nasa_coordinate, radius=1000, color='#d35400', fill=True
).add_child(folium.Popup('NASA Johnson Space Center'))

marker = folium.map.Marker(
    nasa_coordinate,
    icon=DivIcon(
        icon_size=(20,20),
        icon_anchor=(0,0),
        html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % 'NASA JSC',
    )
)

site_map.add_child(circle)
site_map.add_child(marker)

# Add each launch site as a circle marker
for _, row in launch_sites_df.iterrows():
    folium.CircleMarker(
        location=[row['Lat'], row['Long']],
        radius=10,
        color='blue',
        fill=True,
        fill_color='blue',
        popup=row['Launch Site']
    ).add_to(site_map)

# Save map to HTML
site_map.save('spacex_launch_sites_map.html')
print("Map has been saved as 'spacex_launch_sites_map.html'")
