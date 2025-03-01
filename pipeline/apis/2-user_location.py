#!/usr/bin/env python3

import requests
import sys
import time

def check_rate_limit(res):
    """Check the rate limit and sleep if necessary."""
    if res.status_code == 403:
        rate_limit_reset_time = int(res.headers.get('X-Ratelimit-Reset'))
        current_time = int(time.time())
        wait_time = rate_limit_reset_time - current_time
        if wait_time > 0:
            print(f"Rate limit exceeded. Waiting for {wait_time} seconds...")
            time.sleep(wait_time)
            return True
    return False

def main():
    """Main function to handle API requests and responses."""
    if len(sys.argv) != 2:
        print("Usage: python3 2-user_location.py <API_URL>")
        sys.exit(1)
    
    url = sys.argv[1]

    retries = 0
    MAX_RETRIES = 3

    while retries < MAX_RETRIES:
        res = requests.get(url)
        
        if res.status_code == 200:
            try:
                res_data = res.json()
                print(res_data['location'])
                break
            except Exception as e:
                print("Error processing the response:", e)
                print("Response content:", res.text)
                break
        elif res.status_code == 403:
            if not check_rate_limit(res):
                print("Forbidden: Possible API rate limit or permission issue.")
                break
            retries += 1
        elif res.status_code == 404:
            print("Resource not found. Ensure the URL is correct.")
            break
        else:
            print(f"Unexpected status code {res.status_code}")
            break

if __name__ == "__main__":
    main()
