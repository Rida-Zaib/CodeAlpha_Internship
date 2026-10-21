"""
Single source of truth for the target menus and for the 'Auto' (detect everything) vocabulary.

* CATEGORIES  - what the website's dropdown menus list (ordered; the first three share the top row).
* COCO_80     - the 80 classes the standard YOLO models know.
* UNIVERSAL_VOCAB - every name in CATEGORIES (+ a few extras). YOLO-World is baked once with this list so that
  Auto mode can name things that are not in COCO (lion, tiger, pigeon, whiteboard ...).

The website's menu markup is generated from CATEGORIES by tools/gen_class_menus.py.
"""
from __future__ import annotations

COCO_80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

CATEGORIES = [
    ("people_vehicles", "People & Vehicles", [
        "person", "child", "baby", "cyclist", "police officer", "soldier", "worker",
        "bicycle", "car", "motorcycle", "bus", "truck", "train", "airplane", "boat",
        "taxi", "van", "pickup truck", "jeep", "ambulance", "fire truck", "police car", "tractor", "scooter",
        "auto rickshaw", "tram", "helicopter", "ship", "kayak", "canoe", "trolley", "cart", "wheelchair",
        "stroller", "bulldozer", "crane", "horse carriage", "jet ski"]),
    ("animals", "Animals", [
        "dog", "cat", "horse", "sheep", "cow", "goat", "donkey", "pig", "rabbit", "buffalo", "camel",
        "lion", "tiger", "leopard", "cheetah", "wolf", "fox", "deer", "bear", "panda", "monkey", "gorilla",
        "chimpanzee", "elephant", "giraffe", "zebra", "kangaroo", "hippo", "rhino", "squirrel", "hedgehog", "rat",
        "bird", "pigeon", "sparrow", "crow", "seagull", "eagle", "owl", "parrot", "duck", "goose", "chicken",
        "rooster", "peacock", "swan", "flamingo", "penguin",
        "fish", "shark", "dolphin", "whale", "turtle", "jellyfish", "crab",
        "snake", "crocodile", "lizard", "frog",
        "butterfly", "bee", "ant", "spider", "dragonfly", "ladybug"]),
    ("carried", "Carried Items", [
        "backpack", "handbag", "suitcase", "umbrella", "tie", "cell phone", "laptop", "bottle", "cup",
        "wallet", "purse", "shopping bag", "school bag", "water bottle", "coffee cup", "tablet", "camera",
        "headphones", "keys", "id card", "paper", "pen", "power bank", "luggage trolley"]),
    ("clothing", "Clothing & Wear", [
        "hat", "cap", "helmet", "sunglasses", "eyeglasses", "watch", "scarf", "gloves", "face mask", "jacket",
        "coat", "shirt", "dress", "jeans", "shorts", "uniform", "shoes", "sneakers", "boots", "sandals",
        "necklace", "ring", "bracelet", "earrings", "belt", "turban", "hijab"]),
    ("street", "Street & Outdoor", [
        "traffic light", "stop sign", "fire hydrant", "parking meter", "bench",
        "road sign", "traffic cone", "street lamp", "pole", "signboard", "billboard", "fence", "gate", "barrier",
        "bridge", "building", "house", "tower", "tent", "statue", "fountain", "bus stop", "trash can", "mailbox",
        "flag", "speed bump", "manhole", "railway track", "tree", "flower", "bush", "rock", "pond", "boat dock"]),
    ("sports_toys", "Sports & Toys", [
        "sports ball", "frisbee", "kite", "skateboard", "surfboard", "skis", "snowboard", "tennis racket",
        "baseball bat", "baseball glove", "football", "soccer ball", "basketball", "volleyball", "cricket bat",
        "cricket ball", "golf club", "hockey stick", "badminton racket", "boxing glove", "dumbbell",
        "goal post", "swing", "slide", "seesaw", "trampoline", "skipping rope", "toy", "doll", "balloon",
        "teddy bear", "toy car", "puzzle"]),
    ("food", "Food & Kitchen", [
        "banana", "apple", "orange", "broccoli", "carrot", "sandwich", "hot dog", "pizza", "donut", "cake",
        "burger", "bread", "egg", "rice", "noodles", "soup", "salad", "grapes", "strawberry", "watermelon",
        "mango", "lemon", "tomato", "potato", "onion", "cucumber", "mushroom", "cookie", "ice cream",
        "chocolate", "juice", "milk", "tea cup", "bowl", "plate", "glass", "wine glass", "fork", "knife",
        "spoon", "chopsticks", "pan", "pot", "kettle", "tray", "basket"]),
    ("home", "Home & Furniture", [
        "chair", "couch", "bed", "dining table", "table", "desk", "toilet", "sink", "potted plant", "vase",
        "scissors", "hair drier", "toothbrush", "shelf", "bookshelf", "cabinet", "wardrobe", "lamp", "curtain",
        "carpet", "door", "window", "mirror", "pillow", "blanket", "fan", "air conditioner", "stairs",
        "mattress", "bathtub", "towel", "candle", "picture frame", "fireplace", "clothes hanger", "bucket"]),
    ("electronics", "Electronics & Office", [
        "tv", "computer monitor", "keyboard", "mouse", "remote", "microwave", "oven", "toaster", "refrigerator",
        "washing machine", "vacuum cleaner", "printer", "speaker", "projector", "router", "light bulb", "charger",
        "cable", "drone", "smartwatch", "game controller", "clock", "book", "notebook", "textbook", "pencil",
        "whiteboard", "blackboard", "eraser", "ruler", "calculator", "stapler", "folder", "envelope", "globe", "map",
        "marker"]),
]

# ── merge the big vocabulary (vocab_big.py) and the vehicle catalogue (vehicles.py) into the menus ──
from vocab_big import EXTRA as _EXTRA  # noqa: E402
from vehicles import VEHICLE_MENU_NAMES as _VEH_NAMES  # noqa: E402

_cats = {k: (k, t, list(i)) for k, t, i in CATEGORIES}
for _k, _t, _items in _EXTRA:
    if _k in _cats:
        _cats[_k][2].extend(_items)
    else:
        _cats[_k] = (_k, _t, list(_items))
_cats["people_vehicles"][2].extend(_VEH_NAMES)
_used: set[str] = set()
CATEGORIES = []
for _k, _t, _items in _cats.values():
    _clean = []
    for _n in _items:
        if _n not in _used:  # one menu per name, first menu wins
            _used.add(_n)
            _clean.append(_n)
    CATEGORIES.append((_k, _t, _clean))

# the website only offers names it can actually look for
_seen: set[str] = set()
UNIVERSAL_VOCAB: list[str] = []
for _k, _label, _items in CATEGORIES:
    for _n in _items:
        if _n not in _seen:
            _seen.add(_n)
            UNIVERSAL_VOCAB.append(_n)
for _n in COCO_80:  # safety net: every COCO class is always searchable
    if _n not in _seen:
        _seen.add(_n)
        UNIVERSAL_VOCAB.append(_n)

COCO_SET = set(COCO_80)


# Scenery and "stuff" (no clear outline) make a detector invent boxes, so Auto never searches for these.
# They stay in the menus: if you pick one yourself, a dedicated search is built for it.
STUFF = set("""road, highway, street, sidewalk, pavement, sky, cloud, sun, moon, grass, lawn, meadow, hill, mountain, valley, cliff, cave, waterfall,
river, stream, lake, sea, beach, sand, pebbles, snow, ice, forest, jungle, rainforest, savanna, tundra, desert, dune, canyon, plateau, garden, park,
botanical garden, brick wall, stone wall, concrete wall, hedge, fog, smoke, steam, dust, flame, spark, ember, wave, foam, farm field, wheat field,
rice paddy, corn field, tea plantation, vineyard, orchard, swamp, marsh, island, peninsula, volcano, glacier, iceberg, delta, pond water, leaf,
branch, tree trunk, grass blade, reed, vine, creeper, moss, fern, seaweed, kelp, coral, reef, puddle, rainbow, star, galaxy, planet, comet, lightning,
tornado, storm cloud, graffiti, mural, road marking, lane marking, stop line, arrow marking, crosswalk, zebra crossing, pothole, speed breaker,
construction site, scaffolding, shop, storefront, playground, stadium, grandstand, parking lot, parking space, crowd, herd, flock of birds, bee hive,
spider web, bonfire, campfire, fireworks, wall, floor, ceiling, mango tree, banyan tree, oak tree, pine tree, palm tree""".replace("\n", " ").split(","))
STUFF = {x.strip() for x in STUFF if x.strip()}
DETECT_VOCAB: list[str] = [n for n in UNIVERSAL_VOCAB if n not in STUFF or n in COCO_SET]
DETECT_SET = set(DETECT_VOCAB)
