#!/usr/bin/env python3
"""
Runner script for the GitHub user location module.
"""

import sys
from user_location import get_user_location

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python3 run_user_location.py <github_api_user_url>")
        sys.exit(1)

    api_url = sys.argv[1]
    result = get_user_location(api_url)
    print(result)