#!/usr/bin/env python3
"""
This module provides a function to get a list of starships
that can hold a specified number of passengers.
"""

import requests


def availableShips(passengerCount):
    """
    Retrieve a list of starships that can hold
    at least `passengerCount` passengers.

    Args:
        passengerCount (int): The minimum number of passengers
        the starship should be able to hold.

    Returns:
        list: A list of starship names that can hold
        the specified number of passengers.
    """
    url = "https://swapi-api.alx-tools.com/api/starships/"
    ships = []

    while url:
        response = requests.get(url)
        data = response.json()

        for ship in data["results"]:
            passengers = ship["passengers"]
            if passengers not in ["n/a", "unknown", "0", "none"]:
                passengers = passengers.replace(",", "")
                if int(passengers) >= passengerCount:
                    ships.append(ship["name"])

        url = data["next"]

    return ships


if __name__ == "__main__":
    ships = availableShips(4)
    for ship in ships:
        print(ship)
