#!/usr/bin/env python3
"""
Print the location of a GitHub user, handling 404 and 403 (rate limit).
"""

import sys
import time
import requests


def get_reset_minutes(reset_timestamp):
    """
    Calculates the number of minutes until a rate limit reset.

    Args:
        reset_timestamp: Unix timestamp returned by the GitHub API.

    Returns:
        The number of whole minutes until reset.
    """
    reset_timestamp = int(reset_timestamp)
    seconds = reset_timestamp - time.time()
    if seconds <= 0:
        return 0
    return int(seconds / 60)


def print_user_location(url):
    """
    Prints the location for a GitHub user API URL.

    Args:
        url: GitHub API URL for the requested user.
    """
    res = requests.get(url)
    status = res.status_code

    if status == 404:
        print("Not found")
        return

    if status == 403:
        reset_ts = res.headers.get("X-RateLimit-Reset")
        if reset_ts is not None:
            print("Reset in {} min".format(get_reset_minutes(reset_ts)))
        else:
            print("Reset in 0 min")
        return

    if status != 200:
        print("Not found")
        return

    data = res.json()
    location = data.get("location")

    if location:
        print(location)
    else:
        print("Not found")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        print_user_location(sys.argv[1])
