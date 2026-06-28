#!/usr/bin/env python3
"""
List ships that can hold at least a given number of passengers
using the SWAPI starships endpoint.
"""

import requests


def availableShips(passengerCount):
    """
    Returns a list of ship names that can hold at least passengerCount.
    Uses pagination over the /starships endpoint.
    """
    url = "https://swapi-api.alx-tools.com/api/starships/"
    ships = []

    while url:
        res = requests.get(url)
        if res.status_code != 200:
            break

        data = res.json()
        for ship in data.get("results", []):
            passengers_str = ship.get("passengers", "0")

            # Ignore values that are "unknown" or similar
            if not passengers_str.replace(",", "").isdigit():
                continue

            passengers = int(passengers_str.replace(",", ""))

            if passengers >= passengerCount:
                ships.append(ship.get("name"))

        url = data.get("next")

    return ships
