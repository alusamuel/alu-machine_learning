#!/usr/bin/env python3
"""
Display the number of launches per rocket using the SpaceX API.
"""

import requests


BASE_URL = "https://api.spacexdata.com/v4"


def get_launches(base_url=BASE_URL):
    """
    Gets all SpaceX launches.

    Args:
        base_url: Base URL for the SpaceX API.

    Returns:
        A list of launch dictionaries.
    """
    launches_res = requests.get("{}/launches".format(base_url))
    launches_res.raise_for_status()
    return launches_res.json()


def get_rockets(base_url=BASE_URL):
    """
    Gets all SpaceX rockets.

    Args:
        base_url: Base URL for the SpaceX API.

    Returns:
        A list of rocket dictionaries.
    """
    rockets_res = requests.get("{}/rockets".format(base_url))
    rockets_res.raise_for_status()
    return rockets_res.json()


def count_launches_by_rocket(launches):
    """
    Counts launches for each rocket ID.

    Args:
        launches: List of SpaceX launch dictionaries.

    Returns:
        A dictionary mapping rocket IDs to launch counts.
    """
    rocket_counts = {}
    for launch in launches:
        rocket_id = launch.get("rocket")
        if not rocket_id:
            continue
        rocket_counts[rocket_id] = rocket_counts.get(rocket_id, 0) + 1
    return rocket_counts


def get_rocket_frequencies(base_url=BASE_URL):
    """
    Builds sorted launch frequencies by rocket name.

    Args:
        base_url: Base URL for the SpaceX API.

    Returns:
        A list of tuples containing rocket name and launch count.
    """
    launches = get_launches(base_url)
    rockets = get_rockets(base_url)
    rocket_counts = count_launches_by_rocket(launches)
    id_to_name = {r["id"]: r["name"] for r in rockets}

    freq = []
    for rid, count in rocket_counts.items():
        name = id_to_name.get(rid, rid)
        freq.append((name, count))

    freq.sort(key=lambda x: (-x[1], x[0]))
    return freq


def print_rocket_frequencies(base_url=BASE_URL):
    """
    Prints launch frequencies by rocket.

    Args:
        base_url: Base URL for the SpaceX API.
    """
    for name, count in get_rocket_frequencies(base_url):
        print("{}: {}".format(name, count))


if __name__ == "__main__":
    print_rocket_frequencies()
