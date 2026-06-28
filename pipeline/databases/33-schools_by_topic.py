#!/usr/bin/env python3
"""
Return list of schools having a specific topic.
"""


def schools_by_topic(mongo_collection, topic):
    """
    Returns list of school documents that contain given topic.

    mongo_collection: pymongo collection object
    topic: string topic searched
    """
    return list(mongo_collection.find({"topics": topic}))
