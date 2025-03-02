#!/usr/bin/env python3
"""
Uses the (unofficial) SpaceX API to print the number of launches per rocket as:
<rocket name>: <number of launches>
ordered by the number of launches in descending order or,
if rockets have the same amount of launches, in alphabetical order.
"""

import requests

if __name__ == "__main__":
    # SpaceX API URL for launches
    url = 'https://api.spacexdata.com/v4/launches'
    rocketDict = {}

    # Fetch all launch data (handling pagination)
    while url:
        results = requests.get(url).json()
        for launch in results:
            rocket_id = launch.get('rocket')
            if rocket_id:
                # Increment the count for the rocket
                rocketDict[rocket_id] = rocketDict.get(rocket_id, 0) + 1
        # Get the URL for the next page of results, if available
        url = results.get('next')

    # Now fetch rocket names in bulk
    rockets_url = 'https://api.spacexdata.com/v4/rockets'
    rockets = requests.get(rockets_url).json()
    rocket_name_map = {rocket['id']: rocket['name'] for rocket in rockets}

    # Prepare a sorted list of rockets by launch count
    rocketList = [(rocket_name_map[rocket_id], count) for rocket_id, count in rocketDict.items()]
    rocketList = sorted(rocketList, key=lambda kv: (kv[1], kv[0]), reverse=True)

    # Print the sorted rockets and their launch counts
    for rocket in rocketList:
        print(f"{rocket[0]}: {rocket[1]}")
