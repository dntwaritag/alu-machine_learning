#!/usr/bin/env python3

"""Return the location of a specific GitHub user"""

import requests
import sys
import time


def get_user_location(api_url):
    """
    Fetches the location of a specific GitHub user from the provided API URL.

    Parameters:
    api_url (str): The API endpoint URL to fetch user data from.

    Returns:
    None: Prints the user's location or an error message if not found.

    Raises:
    requests.exceptions.RequestException: If there's an HTTP request issue.
    """
    try:
        res = requests.get(api_url)

        if res.status_code == 403:  # Rate limit exceeded
            rate_limit_reset = int(res.headers.get('X-Ratelimit-Reset', 0))
            current_time = int(time.time())
            wait_time = (rate_limit_reset - current_time) // 60
            print("Reset in {} min".format(wait_time))
            sys.exit(1)

        elif res.status_code == 404:  # User not found
            print("Not found")
            sys.exit(1)

        elif res.status_code == 200:  # Successful request
            user_data = res.json()
            location = user_data.get('location', 'Location not provided')
            print(location)

        else:  # Other unexpected status codes
            print("Error: Status code {}".format(res.status_code))
            sys.exit(1)

    except requests.exceptions.RequestException as e:
        print("An error occurred: {}".format(e))
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: ./2-user_location.py <API_URL>")
        sys.exit(1)

    api_url = sys.argv[1]
    get_user_location(api_url)
