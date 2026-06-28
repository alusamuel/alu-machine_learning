#!/usr/bin/env python3
"""
List all documents in a MongoDB collection.
"""


def list_all(mongo_collection):
    """
    Lists all documents in the given collection.

    mongo_collection: pymongo collection object
    Returns a list of documents (empty list if none).
    """
    return list(mongo_collection.find())
