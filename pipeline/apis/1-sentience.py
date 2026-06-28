#!/usr/bin/env python3
"""
List names of home planets of all sentient species
using the SWAPI species endpoint.
"""

import requests


def sentientPlanets():
    """
    Returns a list of planet names that are homeworld
    of species classified as sentient.
    Uses pagination over /species.
    """
    species_url = "https://swapi-api.alx-tools.com/api/species/"
    planets = set()

    while species_url:
        res = requests.get(species_url)
        if res.status_code != 200:
            break

        data = res.json()
        for spec in data.get("results", []):
            classification = spec.get("classification", "").lower()
            designation = spec.get("designation", "").lower()

            if ("sentient" not in classification and
                    "sentient" not in designation):
                continue

            homeworld_url = spec.get("homeworld")
            if not homeworld_url:
                continue

            p_res = requests.get(homeworld_url)
            if p_res.status_code != 200:
                continue

            planet_data = p_res.json()
            name = planet_data.get("name")
            if name:
                planets.add(name)

        species_url = data.get("next")

    return list(planets)
