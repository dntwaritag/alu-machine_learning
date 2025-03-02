#!/usr/bin/env python3
"""Pipeline Api"""
import requests


if __name__ == '__main__':
    """pipeline api"""
    url = "https://api.spacexdata.com/v4/launches"
    r = requests.get(url)
    rocket_dict = {"5e9d0d95eda69955f709d1eb": 0}

    for launch in r.json():
        if launch["rocket"] in rocket_dict:
            rocket_dict[launch["rocket"]] += 1
        else:
            rocket_dict[launch["rocket"]] = 1
    # Sort by the count first and then by the name alphabetically.
    sorted_rockets = sorted(
        rocket_dict.items(), key=lambda kv: (-kv[1], kv[0])
    )

    for key, value in sorted_rockets:
        rurl = "https://api.spacexdata.com/v4/rockets/" + key
        req = requests.get(rurl)

        print(req.json()["name"] + ": " + str(value))
