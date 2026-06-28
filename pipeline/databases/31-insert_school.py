#!/usr/bin/env python3
"""
Insert a new school document in a MongoDB collection.
"""


def insert_school(mongo_collection, **kwargs):
    """
    Inserts a new document in the collection based on kwargs.

    mongo_collection: pymongo collection object
    Returns the new _id.
    """
    result = mongo_collection.insert_one(kwargs)
    return result.inserted_id
