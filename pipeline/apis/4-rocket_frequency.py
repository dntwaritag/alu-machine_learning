#!/usr/bin/env python3
"""Pipeline API: Count SpaceX launches per rocket"""

import requests

if __name__ == '__main__':
    url = "https://api.spacexdata.com/v4/launches"
    response = requests.get(url)

    if response.status_code != 200:
        print("Error: Unable to fetch data from SpaceX API")
        exit(1)

    rocket_dict = {}

    for launch in response.json():
        rocket_id = launch["rocket"]
        if rocket_id in rocket_dict:
            rocket_dict[rocket_id] += 1
        else:
            rocket_dict[rocket_id] = 1

    # Sort by launch count (descending) and then alphabetically by rocket ID
    sorted_rockets = sorted(rocket_dict.items(), key=lambda kv: (-kv[1], kv[0]))

    for rocket_id, count in sorted_rockets:
        rocket_url = f"https://api.spacexdata.com/v4/rockets/{rocket_id}"
        rocket_response = requests.get(rocket_url)

        if rocket_response.status_code == 200:
            rocket_name = rocket_response.json().get("name", "Unknown Rocket")
            print(f"{rocket_name}: {count}")
