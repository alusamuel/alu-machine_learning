#!/usr/bin/env python3
"""
Print the location of a GitHub user, handling 404 and 403 (rate limit).
"""

import sys
import time
import requests


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(0)

    url = sys.argv[1]

    res = requests.get(url)
    status = res.status_code

    # 404: user not found
    if status == 404:
        print("Not found")
        sys.exit(0)

    # 403: rate limit exceeded
    if status == 403:
        reset_ts = res.headers.get("X-RateLimit-Reset")
        if reset_ts is not None:
            reset_ts = int(reset_ts)
            now = int(time.time())
            minutes = int((reset_ts - now) / 60)
            if minutes < 0:
                minutes = 0
            print(f"Reset in {minutes} min")
        else:
            print("Reset in 0 min")
        sys.exit(0)

    # Other errors
    if status != 200:
        print("Not found")
        sys.exit(0)

    data = res.json()
    location = data.get("location")

    if location:
        print(location)
    else:
        print("Not found")
