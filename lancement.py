# Importation des bibliothèques nécessaires
import requests
import pandas as pd
import numpy as np
import datetime

# Affichage complet des colonnes et du contenu
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

# Définition des fonctions pour enrichir les données via l'API
def getBoosterVersion(data):
    for x in data['rocket']:
        if x:
            response = requests.get(f"https://api.spacexdata.com/v4/rockets/{x}").json()
            BoosterVersion.append(response.get('name'))

def getLaunchSite(data):
    for x in data['launchpad']:
        if x:
            response = requests.get(f"https://api.spacexdata.com/v4/launchpads/{x}").json()
            Longitude.append(response.get('longitude'))
            Latitude.append(response.get('latitude'))
            LaunchSite.append(response.get('name'))

def getPayloadData(data):
    for load in data['payloads']:
        if load:
            response = requests.get(f"https://api.spacexdata.com/v4/payloads/{load}").json()
            PayloadMass.append(response.get('mass_kg'))
            Orbit.append(response.get('orbit'))

def getCoreData(data):
    for core in data['cores']:
        if core.get('core') is not None:
            response = requests.get(f"https://api.spacexdata.com/v4/cores/{core['core']}").json()
            Block.append(response.get('block'))
            ReusedCount.append(response.get('reuse_count'))
            Serial.append(response.get('serial'))
        else:
            Block.append(None)
            ReusedCount.append(None)
            Serial.append(None)
        Outcome.append(f"{core.get('landing_success')} {core.get('landing_type')}")
        Flights.append(core.get('flight'))
        GridFins.append(core.get('gridfins'))
        Reused.append(core.get('reused'))
        Legs.append(core.get('legs'))
        LandingPad.append(core.get('landpad'))

# Chargement des données depuis l'URL statique
static_json_url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/API_call_spacex_api.json'
response = requests.get(static_json_url)
data = response.json()

# Conversion en DataFrame
df = pd.json_normalize(data)
print(df.head())

# Sélection des colonnes utiles
data = pd.DataFrame(data)[['rocket', 'payloads', 'launchpad', 'cores', 'flight_number', 'date_utc']]

# Filtrage des lignes avec un seul core et une seule payload
data = data[data['cores'].map(len) == 1]
data = data[data['payloads'].map(len) == 1]

# Extraction des valeurs uniques des listes
data['cores'] = data['cores'].map(lambda x: x[0])
data['payloads'] = data['payloads'].map(lambda x: x[0])

# Conversion de la date
data['date'] = pd.to_datetime(data['date_utc']).dt.date

# Filtrage par date
data = data[data['date'] <= datetime.date(2020, 11, 13)]

# Initialisation des listes globales
BoosterVersion, PayloadMass, Orbit = [], [], []
LaunchSite, Outcome, Flights = [], [], []
GridFins, Reused, Legs = [], [], []
LandingPad, Block, ReusedCount, Serial = [], [], [], []
Longitude, Latitude = [], []

# Appels des fonctions d'enrichissement
getBoosterVersion(data)
getLaunchSite(data)
getPayloadData(data)
getCoreData(data)

# Création du dictionnaire final
launch_dict = {
    'FlightNumber': list(data['flight_number']),
    'Date': list(data['date']),
    'BoosterVersion': BoosterVersion,
    'PayloadMass': PayloadMass,
    'Orbit': Orbit,
    'LaunchSite': LaunchSite,
    'Outcome': Outcome,
    'Flights': Flights,
    'GridFins': GridFins,
    'Reused': Reused,
    'Legs': Legs,
    'LandingPad': LandingPad,
    'Block': Block,
    'ReusedCount': ReusedCount,
    'Serial': Serial,
    'Longitude': Longitude,
    'Latitude': Latitude
}

# Création du DataFrame final
data_falcon9 = pd.DataFrame(launch_dict)

# Filtrage pour exclure Falcon 1
data_falcon9 = data_falcon9[data_falcon9['BoosterVersion'] != 'Falcon 1']

# Réindexation des numéros de vol
data_falcon9['FlightNumber'] = range(1, len(data_falcon9) + 1)

# Affichage des valeurs manquantes
print(data_falcon9.isnull().sum())

# Remplacement des valeurs manquantes dans PayloadMass par la moyenne
payload_mean = data_falcon9['PayloadMass'].mean()
data_falcon9['PayloadMass'].fillna(payload_mean, inplace=True)

# Exportation en CSV
data_falcon9.to_csv('dataset_part_1.csv', index=False)