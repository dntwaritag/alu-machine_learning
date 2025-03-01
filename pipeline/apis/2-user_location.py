#!/usr/bin/env python3

""" Fetch and print the location of a GitHub user """

import requests
import sys
import time

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: ./2-user_location.py https://api.github.com/rate_limit")
        sys.exit(1)

    url = sys.argv[1]
    res = requests.get(url)

    if res.status_code == 403:  # Rate limit exceeded
        rate_limit = int(res.headers.get("X-Ratelimit-Reset", 0))
        current_time = int(time.time())
        diff = (rate_limit - current_time) // 60
        print("Reset in {} min".format(diff))
    
    elif res.status_code == 404:  # User not found
        print("Not found")

    elif res.status_code == 200:  # Success
        data = res.json()
        print(data.get("location", "No location available"))
    
    else:  # Handle unexpected errors
        print("Error:", res.status_code)
