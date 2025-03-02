#!/usr/bin/env python3
"""
This script fetches the location of a specific GitHub user using
the GitHub API. It handles rate limiting, missing location data, and user not found.
"""

import requests
import sys
import time


def fetch_user_location(url):
    """Fetch the location of the user from the GitHub API.

    Args:
        url (str): The API URL to fetch user data from.

    Returns:
        str: Location of the user or None if the user doesn't exist.
    """
    try:
        response = requests.get(url)

        if response.status_code == 403:
            # If rate limited, compute the time to wait before retrying
            rate_limit_reset = int(response.headers.get('X-Ratelimit-Reset', time.time()))
            current_time = int(time.time())
            wait_time = rate_limit_reset - current_time
            if wait_time > 0:
                print(f"Rate limit hit. Reset in {wait_time // 60} minutes.")
                time.sleep(wait_time + 1)  # Wait for the reset time, and retry the request
            return fetch_user_location(url)  # Recursively retry

        elif response.status_code == 404:
            print("User not found.")
            return None

        elif response.status_code == 200:
            # If status is OK, return the location from the user data
            user_data = response.json()
            location = user_data.get('location', None)
            
            if location is None:
                print("Location is not set in the user's GitHub profile.")
            return location

        else:
            print(f"Unexpected status code: {response.status_code}")
            return None

    except requests.RequestException as e:
        print(f"Error during request: {e}")
        return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 2-user_location.py <GitHub API URL>")
        sys.exit(1)

    # Get the URL from the command line arguments
    url = sys.argv[1]
    
    # Fetch the location
    location = fetch_user_location(url)
    
    # If a location was found, print it
    if location:
        print(location)
    else:
        print("Location not found or unavailable.")

