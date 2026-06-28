#!/usr/bin/env python3
"""
Display the upcoming SpaceX launch (soonest from now),
with name, local date, rocket name, and launchpad name+locality.
"""

import requests
import sys
from datetime import datetime


def get_upcoming_launch():
    """
    Returns a dict describing the upcoming launch.
    """
    base_url = "https://api.spacexdata.com/v4"

    # Get all upcoming launches
    launches_res = requests.get(f"{base_url}/launches/upcoming")
    launches_res.raise_for_status()
    launches = launches_res.json()

    # Choose launch with smallest date_unix
    upcoming = None
    for launch in launches:
        date_unix = launch.get("date_unix")
        if date_unix is None:
            continue
        if upcoming is None or date_unix < upcoming.get("date_unix"):
            upcoming = launch

    return upcoming


if __name__ == "__main__":
    base_url = "https://api.spacexdata.com/v4"
    upcoming = get_upcoming_launch()
    if not upcoming:
        sys.exit(0)

    name = upcoming.get("name")
    date_unix = upcoming.get("date_unix")

    # Convert to local time ISO string with offset
    dt_local = datetime.fromtimestamp(date_unix).astimezone()
    date_str = dt_local.isoformat()

    # Rocket
    rocket_id = upcoming.get("rocket")
    rocket_name = ""
    if rocket_id:
        r_res = requests.get(f"{base_url}/rockets/{rocket_id}")
        if r_res.status_code == 200:
            rocket_name = r_res.json().get("name", "")

    # Launchpad
    launchpad_id = upcoming.get("launchpad")
    launchpad_name = ""
    locality = ""
    if launchpad_id:
        l_res = requests.get(f"{base_url}/launchpads/{launchpad_id}")
        if l_res.status_code == 200:
            lp = l_res.json()
            launchpad_name = lp.get("name", "")
            locality = lp.get("locality", "")

    # Format: "<name> (<local_date>) <rocket> - <launchpad> (<locality>)"
    print(f"{name} ({date_str}) {rocket_name} - {launchpad_name} ({locality})")
