import numpy as np
from geopy.distance import geodesic

def calculate_similarity(user1, user2):
    # Food Preferences (Jaccard Similarity)
    common_cuisines = set(user1['foodPreferences']['cuisines']).intersection(user2['foodPreferences']['cuisines'])
    total_cuisines = set(user1['foodPreferences']['cuisines']).union(user2['foodPreferences']['cuisines'])
    cuisine_score = len(common_cuisines) / len(total_cuisines)

    # Location (Geodesic Distance)
    loc1 = (user1['location']['latitude'], user1['location']['longitude'])
    loc2 = (user2['location']['latitude'], user2['location']['longitude'])
    distance = geodesic(loc1, loc2).kilometers
    location_score = max(0, 1 - (distance / 50))  # Normalize within 50 km range

    # Spending Limit (Overlap Percentage)
    spending_overlap = min(user1['spendingLimit']['max'], user2['spendingLimit']['max']) - max(user1['spendingLimit']['min'], user2['spendingLimit']['min'])
    spending_score = max(0, spending_overlap) / (user1['spendingLimit']['max'] - user1['spendingLimit']['min'])

    # Aggregate Weighted Score
    weights = {'cuisine': 0.4, 'location': 0.3, 'spending': 0.3}
    total_score = (
        weights['cuisine'] * cuisine_score +
        weights['location'] * location_score +
        weights['spending'] * spending_score
    )

    return total_score

# Example Usage
user1 = { ... }  # JSON data
user2 = { ... }  # JSON data
score = calculate_similarity(user1, user2)
print(f"Compatibility Score: {score}")
