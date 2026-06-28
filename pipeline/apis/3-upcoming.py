#!/usr/bin/env python3
"""
Display the upcoming SpaceX launch (soonest from now),
with name, local date, rocket name, and launchpad name+locality.
"""

import requests


BASE_URL = "https://api.spacexdata.com/v4"


def get_upcoming_launch(base_url=BASE_URL):
    """
    Gets the next upcoming SpaceX launch.

    Args:
        base_url: Base URL for the SpaceX API.

    Returns:
        The launch dictionary with the earliest date_unix value, or None.
    """
    launches_res = requests.get("{}/launches/upcoming".format(base_url))
    launches_res.raise_for_status()
    launches = launches_res.json()

    upcoming = None
    for launch in launches:
        date_unix = launch.get("date_unix")
        if date_unix is None:
            continue
        if upcoming is None or date_unix < upcoming.get("date_unix"):
            upcoming = launch

    return upcoming


def get_rocket_name(rocket_id, base_url=BASE_URL):
    """
    Gets the name of a SpaceX rocket.

    Args:
        rocket_id: SpaceX rocket ID.
        base_url: Base URL for the SpaceX API.

    Returns:
        The rocket name, or an empty string if it cannot be found.
    """
    if not rocket_id:
        return ""
    res = requests.get("{}/rockets/{}".format(base_url, rocket_id))
    if res.status_code != 200:
        return ""
    return res.json().get("name", "")


def get_launchpad(launchpad_id, base_url=BASE_URL):
    """
    Gets launchpad information from the SpaceX API.

    Args:
        launchpad_id: SpaceX launchpad ID.
        base_url: Base URL for the SpaceX API.

    Returns:
        A tuple containing launchpad name and locality.
    """
    if not launchpad_id:
        return "", ""
    res = requests.get("{}/launchpads/{}".format(base_url, launchpad_id))
    if res.status_code != 200:
        return "", ""
    launchpad = res.json()
    return launchpad.get("name", ""), launchpad.get("locality", "")


def print_upcoming_launch(base_url=BASE_URL):
    """
    Prints the next upcoming SpaceX launch.

    Args:
        base_url: Base URL for the SpaceX API.
    """
    upcoming = get_upcoming_launch(base_url)
    if not upcoming:
        return

    name = upcoming.get("name")
    date_local = upcoming.get("date_local")
    rocket_name = get_rocket_name(upcoming.get("rocket"), base_url)
    launchpad_name, locality = get_launchpad(upcoming.get("launchpad"),
                                             base_url)

    print("{} ({}) {} - {} ({})".format(name, date_local, rocket_name,
                                        launchpad_name, locality))


if __name__ == "__main__":
    print_upcoming_launch()
