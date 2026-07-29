"""Zero-shot prompt bank for exploratory review slices.

Each image is assigned at most one prompt within a group, but the concepts are
not a ground-truth taxonomy and the winning margin is not an accuracy estimate.
Each entry maps a review label to text embedded by SigLIP. Classification is a
dot product against image embeddings that already exist, so adding a group
costs nothing at request time — edit this file and re-run
`python -m app.analyze --only attributes`.
"""

LABEL_BANK: dict[str, dict[str, str]] = {
    "setting": {
        "indoor": "a photo taken indoors",
        "outdoor": "a photo taken outdoors",
    },
    "time_of_day": {
        "day": "a photo taken during the day",
        "night": "a photo taken at night",
        "dusk": "a photo taken at sunset or dusk",
    },
    "environment": {
        "beach/water": "a photo at the beach or in the water",
        "snow": "a photo in the snow",
        "street": "a photo on a city street",
        "forest": "a photo in a forest or woods",
        "field/park": "a photo in a grassy field or park",
        "mountains": "a photo in the mountains or on rocky terrain",
        "indoors": "a photo inside a building or room",
    },
    "main_subject": {
        "dog": "a photo of a dog",
        "person": "a photo of one person",
        "group": "a photo of a group of people",
        "child": "a photo of a child",
        "other animal": "a photo of an animal that is not a dog",
        "vehicle": "a photo of a vehicle",
    },
}

SOFTMAX_SCALE = 100.0  # standard CLIP-style logit scale for cosine similarities
