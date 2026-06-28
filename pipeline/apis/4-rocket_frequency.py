#!/usr/bin/env python3
"""
Display the number of launches per rocket using the SpaceX API.
"""

import requests


if __name__ == "__main__":
    base_url = "https://api.spacexdata.com/v4"

    # Get all launches
    launches_res = requests.get(f"{base_url}/launches")
    launches_res.raise_for_status()
    launches = launches_res.json()

    # Count launches per rocket id
    rocket_counts = {}
    for launch in launches:
        rocket_id = launch.get("rocket")
        if not rocket_id:
            continue
        rocket_counts[rocket_id] = rocket_counts.get(rocket_id, 0) + 1

    # Fetch rocket names
    rockets_res = requests.get(f"{base_url}/rockets")
    rockets_res.raise_for_status()
    rockets = rockets_res.json()

    id_to_name = {r["id"]: r["name"] for r in rockets}

    # Build list (name, count)
    freq = []
    for rid, count in rocket_counts.items():
        name = id_to_name.get(rid, rid)
        freq.append((name, count))

    # Sort: by count desc, then name asc
    freq.sort(key=lambda x: (-x[1], x[0]))

    # Print "<name>: <count>" per line
    for name, count in freq:
        print(f"{name}: {count}")
