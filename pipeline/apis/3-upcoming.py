#!/usr/bin/env python3
"""Pipeline API: Fetch the next upcoming SpaceX launch"""

import requests
from datetime import datetime

if __name__ == '__main__':
    url = "https://api.spacexdata.com/v4/launches/upcoming"
    response = requests.get(url)

    if response.status_code != 200:
        print("Error: Unable to fetch upcoming launches.")
        exit(1)

    launches = response.json()
    
    if not launches:
        print("No upcoming launches found.")
        exit(1)

    # Find the soonest launch
    next_launch = min(launches, key=lambda x: x["date_unix"])

    launch_name = next_launch["name"]
    date = next_launch["date_local"]
    rocket_id = next_launch["rocket"]
    launchpad_id = next_launch["launchpad"]

    # Fetch Rocket Name
    rocket_url = f"https://api.spacexdata.com/v4/rockets/{rocket_id}"
    rocket_response = requests.get(rocket_url)

    if rocket_response.status_code == 200:
        rocket_name = rocket_response.json().get("name", "Unknown Rocket")
    else:
        rocket_name = "Unknown Rocket"

    # Fetch Launchpad Info
    launchpad_url = f"https://api.spacexdata.com/v4/launchpads/{launchpad_id}"
    launchpad_response = requests.get(launchpad_url)

    if launchpad_response.status_code == 200:
        launchpad_data = launchpad_response.json()
        launchpad_name = launchpad_data.get("name", "Unknown Launchpad")
        launchpad_location = launchpad_data.get("locality", "Unknown Location")
    else:
        launchpad_name = "Unknown Launchpad"
        launchpad_location = "Unknown Location"

    # Format Output String
    output = f"{launch_name} ({date}) {rocket_name} - {launchpad_name} ({launchpad_location})"
    print(output)
