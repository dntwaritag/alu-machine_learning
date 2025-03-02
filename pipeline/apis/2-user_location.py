#!/usr/bin/env python3

""" Return the location of a specific GitHub user """

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

        if res.status_code == 403:
            rate_limit_reset = int(res.headers.get('X-Ratelimit-Reset', 0))
            current_time = int(time.time())
            diff = (rate_limit_reset - current_time) // 60
            print("Reset in {} min".format(diff))
        elif res.status_code == 404:
            print("Not found")
        elif res.status_code == 200:
            user_data = res.json()
            location = user_data.get('location', 'Location not provided')
            print(location)
        else:
            error_msg = "Error: Status code {}".format(res.status_code)
            print(error_msg)

    except requests.exceptions.Request)



