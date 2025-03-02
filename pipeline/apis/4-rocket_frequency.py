#!/usr/bin/env python3

"""Count past launches per SpaceX rocket type"""

import requests
from collections import defaultdict


def count_rocket_launches():
    """
    Fetches past SpaceX launches and counts the number of times each rocket has been used.
    """
    url = "https://api.spacexdata.com/v4/launches/past"
    response = requests.get(url)

    if response.status_code != 200:
        print("Error fetching launch data")
        return

    launches = response.json()
    rocket_count = defaultdict(int)

    # Count launches per rocket ID
    for launch in launches:
        rocket_count[launch["rocket"]] += 1

    # Fetch rocket names
    rocket_names = {}
    for rocket_id in rocket_count.keys():
        rocket_url = f"https://api.spacexdata.com/v4/rockets/{rocket_id}"
        rocket_data = requests.get(rocket_url).json()
        rocket_names[rocket_id] = rocket_data["name"]

    # Print formatted results
    for rocket_id, count in rocket_count.items():
        print(f"{rocket_names[rocket_id]}: {count}")

if __name__ == "__main__":
    count_rocket_launches()
