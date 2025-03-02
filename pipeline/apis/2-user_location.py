#!/usr/bin/env python3
'''
Prints the location of a user
'''


import sys
import requests
import time


def get_user_location(api_url):
    """
    Fetch and print the location of a GitHub user.

    :param api_url: The API URL for the user
    """
    try:
        response = requests.get(api_url)

        if response.status_code == 200:
            user_data = response.json()
            location = user_data.get('location')
            if location:
                print(location)
            else:
                print('Location not available')
        elif response.status_code == 404:
            print('Not found')
        elif response.status_code == 403:
            reset_time = int(
                response.headers.get('X-RateLimit-Reset', time.time()))
            current_time = int(time.time())
            wait_time = (reset_time - current_time) // 60
            print('Reset in {} min'.format(wait_time))
        else:
            print('Error: {}'.format(response.status_code))
    except requests.RequestException as e:
        print('An error occurred: {}'.format(e))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: ./2-user_location.py <api_url>')
        sys.exit(1)

    api_url = sys.argv[1]
    get_user_location(api_url)
