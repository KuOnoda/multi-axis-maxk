"""Fixed CLIP prompt ensembles used by the reported training recipes."""

RACE5_MODES = ("white", "black", "asian", "indian", "latino")
RACE5_PROMPTS = (
    (
        "a photo of a White person",
        "a portrait of a White face",
        "a photo of a Caucasian person",
        "a portrait of a Caucasian face",
        "a photo of a European person",
        "a portrait of a European face",
        "a headshot of a White person",
        "a photo of a light-skinned European person",
    ),
    (
        "a photo of a Black person",
        "a portrait of a Black face",
        "a photo of an African person",
        "a photo of an African American person",
        "a portrait of a Black African person",
        "a headshot of a Black person",
        "a photo of a person of African descent",
        "a portrait of a dark-skinned Black person",
    ),
    (
        "a photo of an East Asian person",
        "a portrait of an East Asian face",
        "a photo of a Southeast Asian person",
        "a portrait of a Southeast Asian face",
        "a headshot of an East Asian person",
        "a photo of a person from East Asia",
    ),
    (
        "a photo of an Indian person",
        "a portrait of an Indian face",
        "a photo of a South Asian person",
        "a headshot of a South Asian person",
        "a photo of a Bangladeshi person",
        "a photo of a person from the Indian subcontinent",
    ),
    (
        "a photo of a Latino person",
        "a portrait of a Latino face",
        "a photo of a Hispanic person",
        "a portrait of a Hispanic face",
        "a photo of a Latin American person",
        "a headshot of a Hispanic person",
    ),
)

GENDER2_MODES = ("male", "female")
GENDER2_PROMPTS = (
    (
        "a photo of a male person",
        "a portrait of a male person",
        "a close-up portrait of a man",
        "a headshot of a man",
        "a photo of a man",
        "a face photo of a male adult",
    ),
    (
        "a photo of a female person",
        "a portrait of a female person",
        "a close-up portrait of a woman",
        "a headshot of a woman",
        "a photo of a woman",
        "a face photo of a female adult",
    ),
)
