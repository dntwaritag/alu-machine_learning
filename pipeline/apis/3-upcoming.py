import requests
from datetime import datetime

def get_upcoming_launch():
    url = 'https://api.spacexdata.com/v4/launches/upcoming'
    response = requests.get(url).json()

    upcoming_launch = sorted(response, key=lambda x: x['date_unix'])[0]

    launch_name = upcoming_launch['name']
    launch_date = datetime.utcfromtimestamp(upcoming_launch['date_unix']).strftime('%Y-%m-%dT%H:%M:%S')
    rocket_name = upcoming_launch['rocket']['name']
    launchpad_name = upcoming_launch['launchpad']['name']
    launchpad_locality = upcoming_launch['launchpad']['locality']

    print(f"{launch_name} ({launch_date}) {rocket_name} - {launchpad_name} ({launchpad_locality})")

if __name__ == "__main__":
    get_upcoming_launch()
