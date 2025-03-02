#!/usr/bin/env python3

"""Fetch upcoming SpaceX launch details"""

import requests


def get_upcoming_launch():
    """
    Fetches the next upcoming SpaceX launch and displays its details.
    """
    url = "https://api.spacexdata.com/v4/launches/upcoming"
    response = requests.get(url)

    if response.status_code != 200:
        print("Error fetching data")
        return

    launches = response.json()
    recent = None

    for launch in launches:
        launch_time = int(launch["date_unix"])
        if recent is None or launch_time < recent:
            recent = launch_time
            launch_name = launch["name"]
            date = launch["date_local"]
            rocket_id = launch["rocket"]
            launchpad_id = launch["launchpad"]

    # Fetch rocket name
    rocket_url = "https://api.spacexdata.com/v4/rockets/{}".format(rocket_id)
    rocket_name = requests.get(rocket_url).json()["name"]

    # Fetch launchpad details
    launchpad_url = "https://api.spacexdata.com/v4/launchpads/{}".format(launchpad_id)
    launchpad_data = requests.get(launchpad_url).json()
    launchpad_name = launchpad_data["name"]
    launchpad_location = launchpad_data["locality"]

    result = "{} ({}) {} - {} ({})".format(
        launch_name, date, rocket_name, launchpad_name, launchpad_location
    )

    print(result)


if __name__ == "__main__":
    get_upcoming_launch()
