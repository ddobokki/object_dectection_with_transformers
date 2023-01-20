from dataclasses import dataclass


@dataclass
class Folder:
    data = "./data/"
    """
    data folder의 상대경로
    """
    data_train = "./data/Train"
    """
    data/Train 의 상대경로
    """
    data_test = "./data/Test"
    """
    data/Test 상대경로
    """
    data_preprocess = "./data/preprocess/"
    """
    data/preprocess/의 상대경로
    """


@dataclass
class JsonKeys:
    info = "info"
    licenses = "licenses"
    images = "images"
    annotations = "annotations"
    categories = "categories"


@dataclass
class RawDataColumns:
    id = "id"
    file_name = "file_name"
    height = "height"
    width = "width"
    image_id = "image_id"
    bbox = "bbox"
    area = "area"
    category_id = "category_id"
    iscrowd = "iscrowd"


@dataclass
class DatasetColumns:
    pixel_values = "pixel_values"
    """pixel_values"""
    boxes = "boxes"
    """boxes"""
    class_labels = "class_labels"
    """class_labels"""
    labels = "labels"
    """labels"""


Label2Id = {
    "Potato Rice": 0,
    "Gondrebab": 1,
    "Rice ball": 2,
    "Fried rice": 3,
    "Albab": 4,
    "pine mushroom rice": 5,
    "Japtangbap": 6,
    "Sundae gukbap": 7,
    "Jeonju Bean Sprout Soup": 8,
    "Rice with Sashimi": 9,
    "Pork and rice soup": 10,
    "hayashi Rice": 11,
    "Kimchi Gimbap": 12,
    "Octopus sushi": 13,
    "Shrimp sushi": 14,
    "Ribs Triangular Gimbap": 15,
    "Regular Gimbap": 16,
    "Chicken Kalguksu": 17,
    "Black Bean Noodles": 22,
    "Jjamppong": 19,
    "Kongguksu": 20,
    "Tteokguk": 21,
    "Meat Dumpling": 23,
    "Dumpling soup": 24,
    "Crab meat porridge": 25,
    "Abalone porridge": 26,
    "Vegetable Porridge": 27,
    "Corn Soup": 28,
    "Tomato soup": 29,
    "Oyster soup": 30,
    "Egg soup": 31,
    "Roong doenjang soup": 32,
    "Short Rib Soup": 33,
    "Kkori Gomtang": 34,
    "Blue Crab Stew": 35,
    "Braised Spicy Chicken": 36,
    "Samgyetang": 37,
    "Seafood stew": 38,
    "Mackerel stew": 39,
    "Budae-jjigae": 40,
    "Tofu Stew": 41,
    "Pumpkin stew": 42,
    "Gochujang-jjigae": 43,
    "Octopus soup": 44,
    "Steamed monkfish": 45,
    "Steamed yellow corvina": 46,
    "Braised Short Ribs": 47,
    "Jjimdak": 48,
    "Jokbal": 49,
    "Bulgogi": 50,
    "Seasoned King Rib": 51,
    "Chicken teriyaki": 52,
    "Grilled Cabbage": 53,
    "Oyster pancake": 54,
    "Round meatballs": 55,
    "Hampan": 56,
    "Minarijeon": 57,
    "Cabbage pancake": 58,
    "Mushroom pancake": 59,
    "Stir-fried octopus": 60,
    "Stir-fried anchovies": 61,
    "Stir-fried squid": 62,
    "Stir-fried oyster mushroom": 63,
    "Stir-fried pork": 64,
    "Stir-fried pork skin": 65,
    "Stir-fried sausage": 66,
    "Duck Bulgogi": 67,
    "Rap-bokki": 68,
    "Mapa Tofu": 69,
    "Braised hairtail": 70,
    "Braised Pollack": 71,
    "Braised dried pollack": 72,
    "Boiled Quail Eggs": 73,
    "Braised potatoes": 74,
    "Braised burdock": 75,
    "Boiled beans": 76,
    "Fish cutlet": 77,
    "Sweet and Sour Pork": 78,
    "Fried laver": 79,
    "Radish Nocturne": 80,
    "Seasoned Deodeok": 81,
    "Gorinamul": 82,
    "Chwinamul": 83,
    "Bean sprouts": 84,
    "Dried dried pollack": 85,
    "Seasoned sea snail noodles": 86,
    "Japchae": 87,
    "Dongchimi": 88,
    "Fresh Cabbage": 89,
    "White Kimchi": 90,
    "Young radish kimchi": 91,
    "Cucumber kimchi": 92,
    "Pickled Red Pepper": 93,
    "Sesame Leaf Pickled": 94,
    "Seasoned Cucumber": 95,
    "Cucumber Pickle": 96,
    "Pickled radish": 97,
    "Garaetteok": 98,
    "Chalddeok": 99,
}

Id2Label = {
    0: "Potato Rice",
    1: "Gondrebab",
    2: "Rice ball",
    3: "Fried rice",
    4: "Albab",
    5: "pine mushroom rice",
    6: "Japtangbap",
    7: "Sundae gukbap",
    8: "Jeonju Bean Sprout Soup",
    9: "Rice with Sashimi",
    10: "Pork and rice soup",
    11: "hayashi Rice",
    12: "Kimchi Gimbap",
    13: "Octopus sushi",
    14: "Shrimp sushi",
    15: "Ribs Triangular Gimbap",
    16: "Regular Gimbap",
    17: "Chicken Kalguksu",
    18: "Black Bean Noodles",
    19: "Jjamppong",
    20: "Kongguksu",
    21: "Tteokguk",
    22: "Black Bean Noodles",
    23: "Meat Dumpling",
    24: "Dumpling soup",
    25: "Crab meat porridge",
    26: "Abalone porridge",
    27: "Vegetable Porridge",
    28: "Corn Soup",
    29: "Tomato soup",
    30: "Oyster soup",
    31: "Egg soup",
    32: "Roong doenjang soup",
    33: "Short Rib Soup",
    34: "Kkori Gomtang",
    35: "Blue Crab Stew",
    36: "Braised Spicy Chicken",
    37: "Samgyetang",
    38: "Seafood stew",
    39: "Mackerel stew",
    40: "Budae-jjigae",
    41: "Tofu Stew",
    42: "Pumpkin stew",
    43: "Gochujang-jjigae",
    44: "Octopus soup",
    45: "Steamed monkfish",
    46: "Steamed yellow corvina",
    47: "Braised Short Ribs",
    48: "Jjimdak",
    49: "Jokbal",
    50: "Bulgogi",
    51: "Seasoned King Rib",
    52: "Chicken teriyaki",
    53: "Grilled Cabbage",
    54: "Oyster pancake",
    55: "Round meatballs",
    56: "Hampan",
    57: "Minarijeon",
    58: "Cabbage pancake",
    59: "Mushroom pancake",
    60: "Stir-fried octopus",
    61: "Stir-fried anchovies",
    62: "Stir-fried squid",
    63: "Stir-fried oyster mushroom",
    64: "Stir-fried pork",
    65: "Stir-fried pork skin",
    66: "Stir-fried sausage",
    67: "Duck Bulgogi",
    68: "Rap-bokki",
    69: "Mapa Tofu",
    70: "Braised hairtail",
    71: "Braised Pollack",
    72: "Braised dried pollack",
    73: "Boiled Quail Eggs",
    74: "Braised potatoes",
    75: "Braised burdock",
    76: "Boiled beans",
    77: "Fish cutlet",
    78: "Sweet and Sour Pork",
    79: "Fried laver",
    80: "Radish Nocturne",
    81: "Seasoned Deodeok",
    82: "Gorinamul",
    83: "Chwinamul",
    84: "Bean sprouts",
    85: "Dried dried pollack",
    86: "Seasoned sea snail noodles",
    87: "Japchae",
    88: "Dongchimi",
    89: "Fresh Cabbage",
    90: "White Kimchi",
    91: "Young radish kimchi",
    92: "Cucumber kimchi",
    93: "Pickled Red Pepper",
    94: "Sesame Leaf Pickled",
    95: "Seasoned Cucumber",
    96: "Cucumber Pickle",
    97: "Pickled radish",
    98: "Garaetteok",
    99: "Chalddeok",
}
