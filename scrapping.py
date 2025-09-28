import sys
import requests
from bs4 import BeautifulSoup
import re
import unicodedata
import pandas as pd

def date_time(table_cells):
    """
    This function returns the date and time from the HTML table cell
    Input: the element of a table data cell extracts extra row
    """
    return [data_time.strip() for data_time in list(table_cells.strings)][0:2]

def booster_version(table_cells):
    """
    This function returns the booster version from the HTML table cell 
    Input: the element of a table data cell extracts extra row
    """
    out = ''.join([booster_version for i, booster_version in enumerate(table_cells.strings) if i % 2 == 0][0:-1])
    return out

def landing_status(table_cells):
    """
    This function returns the landing status from the HTML table cell 
    Input: the element of a table data cell extracts extra row
    """
    out = [i for i in table_cells.strings][0]
    return out

def get_mass(table_cells):
    """
    Extract payload mass (kg) from the HTML table cell
    """
    mass = unicodedata.normalize("NFKD", table_cells.text).strip()
    if mass and "kg" in mass:
        new_mass = mass[0:mass.find("kg")+2]
    else:
        new_mass = 0
    return new_mass

def extract_column_from_header(row):
    """
    This function returns the column name from the HTML table header cell 
    Input: the element of a table header cell extracts extra row
    """
    if row.br:
        row.br.extract()
    if row.a:
        row.a.extract()
    if row.sup:
        row.sup.extract()
        
    colunm_name = ' '.join(row.contents)
    
    # Filter the digit and empty names
    if not(colunm_name.strip().isdigit()):
        colunm_name = colunm_name.strip()
        return colunm_name    

# Wikipedia Falcon 9 launches (static oldid version)
static_url = "https://en.wikipedia.org/w/index.php?title=List_of_Falcon_9_and_Falcon_Heavy_launches&oldid=1027686922"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/91.0.4472.124 Safari/537.36"
}

# Fetch the page
response = requests.get(static_url, headers=headers)

# Check if request successful
if response.status_code != 200:
    print(f"Failed to fetch page. Status code: {response.status_code}")
    sys.exit(1)

# Create BeautifulSoup object
soup = BeautifulSoup(response.text, "html.parser")

# Print page title
print("Page Title:", soup.title.string)

# Find all tables with 'wikitable' class (Falcon 9 launch history)
launch_tables = soup.find_all("table", class_="wikitable")

print(f"Found {len(launch_tables)} launch tables.")

# Example: Extract headers from the first table
first_table = launch_tables[0]
headers = [extract_column_from_header(th) for th in first_table.find_all('th')]
headers = [h for h in headers if h is not None]

print("Table Headers:", headers)

# Convert first table rows into a DataFrame (just as an example)
rows = []
for tr in first_table.find_all("tr")[1:]:  # skip header row
    cells = tr.find_all(["td", "th"])
    row = [unicodedata.normalize("NFKD", c.get_text(strip=True)) for c in cells]
    if row:
        rows.append(row)

df = pd.DataFrame(rows, columns=headers[:len(rows[0])])  # align headers to columns
#print(df.head())

# Use the find_all function in the BeautifulSoup object, with element type `table`
# Assign the result to a list called html_tables
html_tables = soup.find_all("table")
# Let's print the third table and check its content
first_launch_table = html_tables[2]
print(first_launch_table)