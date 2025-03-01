import requests
import sys
import time

def get_user_location(url):
    response = requests.get(url)
    if response.status_code == 403:
        reset_time = int(response.headers['X-Ratelimit-Reset'])
        wait_time = reset_time - int(time.time())
        print(f"Reset in {wait_time // 60} min")
    elif response.status_code == 404:
        print("Not found")
    elif response.status_code == 200:
        user_data = response.json()
        print(user_data.get('location', 'Location not available'))

if __name__ == "__main__":
    get_user_location(sys.argv[1])
