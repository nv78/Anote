
from anote import predict

API_KEY = "aaae81afd4872a53b1ee884e50b8c422a"
"""
Hand Labeled Example Text:
Cut the pears (fruit) into slices
Grill the beef (meat) with salt.
Boil the broccoli (vegetable) until tender.
Marinate the pork (meat) in soy sauce and garlic.
Bake the apples (fruit) until they are soft.
Steam the spinach (vegetable) until wilted.
Grill the chicken (meat) until it is cooked thoroughly.
Peel the oranges (fruit) and squeeze them to make juice
Slice the carrots (vegetable) and steam them until tender.
"""

ENTITIES = ["fruit", "vegetable", "meat", "unknown"]

EXAMPLES = {
    "fruit": ["apple", "pear", "orange"],
    "vegetable": ["broccoli", "spinach", "carrots"],
    "meat": ["beef", "pork", "chicken"]
}

text = """Begin by heating up some oil in a spacious pan,
    and then toss in the onions, celery, and carrots.
    Allow them to gently cook over medium-low heat for about 10 minutes,
    or until they become tender. Next, introduce the courgette,
    garlic, red peppers, and oregano to the pan, and sauté them for 2-3 minutes.
    Finally, incorporate slices of oranges and chunks of chicken into the mixture.
"""

prediction = predict(
    categories=ENTITIES,
    examples=EXAMPLES,
    text=text
)

print(prediction)
"""
Output: Begin by heating up some oil in a spacious pan,
    and then toss in the onions (vegetable), celery (vegetable), and carrots (vegetable).
    Allow them to gently cook over medium-low heat for about 10 minutes,
    or until they become tender. Next, introduce the courgette (unknown),
    garlic (unknown), red peppers (vegetable), and oregano (unknown) to the pan, and sauté them for 2-3 minutes.
    Finally, incorporate slices of oranges (fruit) and chunks of chicken (meat) into the mixture.
"""