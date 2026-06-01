ACTIVE_BEHAVIOR_LABELS = [
    "Eat",
    "Talk",
    "TV",
    "Computer/Phone",
    "Other Activity",
]

ACTIVE_BEHAVIOR_TO_ID = {
    label: i for i, label in enumerate(ACTIVE_BEHAVIOR_LABELS)
}

INACTIVE_BEHAVIORS = {"Sleep/Rest", "Inactive"}

ACTIVE_VS_INACTIVE_LABELS = ["Active", "Inactive"]
ACTIVE_VS_INACTIVE_TO_ID = {
    label: i for i, label in enumerate(ACTIVE_VS_INACTIVE_LABELS)
}

LEGACY_BEHAVIOR_LABEL_ALIASES = {
    "Computer/phone": "Computer/Phone",
    "Other activity": "Other Activity",
    "Sleep/rest": "Sleep/Rest",
}
