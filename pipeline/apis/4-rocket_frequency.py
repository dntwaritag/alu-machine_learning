#!/usr/bin/env python3
"""
    Script that displays the number of launches per rocket.
"""

import requests
from collections import Counter


def get_launch_count_by_rocket():
    """
    Get all launches and count them per rocket.
    """
    url = "https://api.spacexdata.com/v4/launches"
    response = requests.get(url)
    launches = response.json()

    # Count the launches per rocket ID
    rocket_counts = Counter(launch["rocket"] for launch in launches)

    # Get rocket names
    url = "https://api.spacexdata.com/v4/rockets"
    response = requests.get(url)
    rockets = response.json()
    rocket_names = {rocket["id"]: rocket["name"] for rocket in rockets}

    # Create a list of (rocket_name, count) and sort it
    rocket_launch_counts = [
        (rocket_names[rocket_id], count)
        for rocket_id, count in rocket_counts.items()
    ]
    rocket_launch_counts.sort(key=lambda x: (-x[1], x[0]))

    return rocket_launch_counts


if __name__ == "__main__":
    rocket_launch_counts = get_launch_count_by_rocket()
    for rocket_name, count in rocket_launch_counts:
        print("{}: {}".format(rocket_name, count))
