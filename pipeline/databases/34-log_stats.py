#!/usr/bin/env python3
"""
Provide some stats about Nginx logs stored in MongoDB.
"""

from pymongo import MongoClient


def print_stats():
    """
    Prints stats about logs.nginx collection.
    """
    client = MongoClient('mongodb://127.0.0.1:27017')
    collection = client.logs.nginx

    total = collection.count_documents({})
    print("{} logs".format(total))

    print("Methods:")
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    for m in methods:
        count = collection.count_documents({"method": m})
        print("\tmethod {}: {}".format(m, count))

    status_count = collection.count_documents(
        {"method": "GET", "path": "/status"}
    )
    print("{} status check".format(status_count))


if __name__ == "__main__":
    print_stats()
