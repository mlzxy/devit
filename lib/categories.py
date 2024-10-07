from detectron2.data.datasets.coco_zeroshot_categories import COCO_SEEN_CLS, \
    COCO_UNSEEN_CLS, COCO_OVD_ALL_CLS
from detectron2.data import MetadataCatalog

coco17_all_classes = MetadataCatalog.get('coco_2017_val').thing_classes

contiguous_id_to_thing_dataset_id = {countId: catId for catId, countId in MetadataCatalog.get('coco_2017_val').thing_dataset_id_to_contiguous_id.items()}

# Dataset Names
LVIS = "lvis_v1_train"
COCO_OVD = "coco_2017_ovd_b_train"

COCO_2017_SPLIT_1 = 'coco_2017_train_oneshot_s1'
COCO_2017_SPLIT_2 = 'coco_2017_train_oneshot_s2'
COCO_2017_SPLIT_3 = 'coco_2017_train_oneshot_s3'
COCO_2017_SPLIT_4 = 'coco_2017_train_oneshot_s4'

COCO_2014_FEW_SHOT = 'fs_coco14_base_train'
COCO_2017_FEW_SHOT = 'fs_coco17_base_train' # not used

PASCAL_VOC_SPLIT_1 = 'pascal_voc_train_split_1'
PASCAL_VOC_SPLIT_2 = 'pascal_voc_train_split_2'
PASCAL_VOC_SPLIT_3 = 'pascal_voc_train_split_3'

voc_all_classes_1 = [ "aeroplane", "bicycle", "boat", "bottle", "car", "cat", "chair", "diningtable", "dog", "horse", "person", "pottedplant", "sheep", "train", "tvmonitor", "bird", "bus", "cow", "motorbike", "sofa"]
voc_all_classes_2 = ['bicycle', 'bird', 'boat', 'bus', 'car', 'cat', 'chair', 'diningtable', 'dog', 'motorbike', 'person', 'pottedplant', 'sheep', 'train', 'tvmonitor', 'aeroplane', 'bottle', 'cow', 'horse', 'sofa']
voc_all_classes_3 = ['aeroplane', 'bicycle', 'bird', 'bottle', 'bus', 'car', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'person', 'pottedplant', 'train', 'tvmonitor', 'boat', 'cat', 'motorbike', 'sheep', 'sofa']

voc_split_1_seen_classes = [ "aeroplane", "bicycle", "boat", "bottle", "car", "cat", "chair", "diningtable", "dog", "horse", "person", "pottedplant", "sheep", "train", "tvmonitor"]
voc_split_2_seen_classes = ["bicycle","bird","boat","bus","car","cat","chair","diningtable","dog","motorbike","person","pottedplant","sheep","train","tvmonitor"]
voc_split_3_seen_classes = ["aeroplane","bicycle","bird","bottle","bus","car","chair","cow","diningtable","dog","horse","person","pottedplant","train","tvmonitor"]



fs_coco_2014_seen_classes = ['truck', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'bed', 'toilet', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# from fs_coco_test_all
fs_coco_2014_all_classes = MetadataCatalog.get('coco_2014_val').thing_classes

def get_oneshot_split(split):
    return [coco17_all_classes[cid] for cid in range(80) if contiguous_id_to_thing_dataset_id[cid] % 4 != split]



SEEN_CLS_DICT = {
    COCO_OVD: COCO_SEEN_CLS,
    LVIS: ['aerosol_can', 'air_conditioner', 'airplane', 'alarm_clock', 'alcohol', 'alligator', 'almond', 'ambulance', 'amplifier', 'anklet', 'antenna', 'apple', 'apron', 'aquarium', 'armband', 'armchair', 'artichoke', 'trash_can', 'ashtray', 'asparagus', 'atomizer', 'avocado', 'award', 'awning', 'baby_buggy', 'basketball_backboard', 'backpack', 'handbag', 'suitcase', 'bagel', 'ball', 'balloon', 'bamboo', 'banana', 'Band_Aid', 'bandage', 'bandanna', 'banner', 'barrel', 'barrette', 'barrow', 'baseball_base', 'baseball', 'baseball_bat', 'baseball_cap', 'baseball_glove', 'basket', 'basketball', 'bat_(animal)', 'bath_mat', 'bath_towel', 'bathrobe', 'bathtub', 'battery', 'bead', 'bean_curd', 'beanbag', 'beanie', 'bear', 'bed', 'bedspread', 'cow', 'beef_(food)', 'beer_bottle', 'beer_can', 'bell', 'bell_pepper', 'belt', 'belt_buckle', 'bench', 'beret', 'bib', 'bicycle', 'visor', 'billboard', 'binder', 'binoculars', 'bird', 'birdfeeder', 'birdbath', 'birdcage', 'birdhouse', 'birthday_cake', 'black_sheep', 'blackberry', 'blackboard', 'blanket', 'blazer', 'blender', 'blinker', 'blouse', 'blueberry', 'boat', 'bobbin', 'bobby_pin', 'boiled_egg', 'deadbolt', 'bolt', 'book', 'bookcase', 'booklet', 'boot', 'bottle', 'bottle_opener', 'bouquet', 'bow_(decorative_ribbons)', 'bow-tie', 'bowl', 'bowler_hat', 'box', 'suspenders', 'bracelet', 'brassiere', 'bread-bin', 'bread', 'bridal_gown', 'briefcase', 'broccoli', 'broom', 'brownie', 'brussels_sprouts', 'bucket', 'horned_cow', 'bulldog', 'bullet_train', 'bulletin_board', 'bullhorn', 'bun', 'bunk_bed', 'buoy', 'bus_(vehicle)', 'business_card', 'butter', 'butterfly', 'button', 'cab_(taxi)', 'cabin_car', 'cabinet', 'cake', 'calculator', 'calendar', 'calf', 'camcorder', 'camel', 'camera', 'camera_lens', 'camper_(vehicle)', 'can', 'can_opener', 'candle', 'candle_holder', 'candy_cane', 'walking_cane', 'canister', 'canoe', 'cantaloup', 'cap_(headwear)', 'bottle_cap', 'cape', 'cappuccino', 'car_(automobile)', 'railcar_(part_of_a_train)', 'identity_card', 'card', 'cardigan', 'horse_carriage', 'carrot', 'tote_bag', 'cart', 'carton', 'cash_register', 'cast', 'cat', 'cauliflower', 'cayenne_(spice)', 'CD_player', 'celery', 'cellular_telephone', 'chair', 'chandelier', 'cherry', 'chicken_(animal)', 'chickpea', 'chili_(vegetable)', 'crisp_(potato_chip)', 'chocolate_bar', 'chocolate_cake', 'choker', 'chopping_board', 'chopstick', 'Christmas_tree', 'slide', 'cigarette', 'cigarette_case', 'cistern', 'clasp', 'cleansing_agent', 'clip', 'clipboard', 'clock', 'clock_tower', 'clothes_hamper', 'clothespin', 'coaster', 'coat', 'coat_hanger', 'coatrack', 'cock', 'coconut', 'coffee_maker', 'coffee_table', 'coffeepot', 'coin', 'colander', 'coleslaw', 'pacifier', 'computer_keyboard', 'condiment', 'cone', 'control', 'cookie', 'cooler_(for_food)', 'cork_(bottle_plug)', 'corkscrew', 'edible_corn', 'cornet', 'cornice', 'corset', 'costume', 'cowbell', 'cowboy_hat', 'crab_(animal)', 'cracker', 'crate', 'crayon', 'crescent_roll', 'crib', 'crock_pot', 'crossbar', 'crow', 'crown', 'crucifix', 'cruise_ship', 'police_cruiser', 'crumb', 'crutch', 'cub_(animal)', 'cube', 'cucumber', 'cufflink', 'cup', 'trophy_cup', 'cupboard', 'cupcake', 'curtain', 'cushion', 'dartboard', 'deck_chair', 'deer', 'dental_floss', 'desk', 'diaper', 'dining_table', 'dish', 
            'dish_antenna', 'dishrag', 'dishtowel', 'dishwasher', 'dispenser', 'Dixie_cup', 'dog', 'dog_collar', 'doll', 'dolphin', 'domestic_ass', 'doorknob', 'doormat', 'doughnut', 'drawer', 'underdrawers', 'dress', 'dress_hat', 'dress_suit', 'dresser', 'drill', 'drum_(musical_instrument)', 'duck', 'duckling', 'duct_tape', 'duffel_bag', 'dumpster', 'eagle', 'earphone', 'earring', 'easel', 'egg', 'egg_yolk', 'eggbeater', 'eggplant', 'refrigerator', 'elephant', 'elk', 'envelope', 'eraser', 'fan', 'faucet', 'Ferris_wheel', 'ferry', 'fighter_jet', 'figurine', 'file_cabinet', 'fire_alarm', 'fire_engine', 'fire_extinguisher', 'fire_hose', 'fireplace', 'fireplug', 'fish', 'fish_(food)', 'fishing_rod', 'flag', 'flagpole', 'flamingo', 'flannel', 'flap', 'flashlight', 'flip-flop_(sandal)', 'flipper_(footwear)', 'flower_arrangement', 'flute_glass', 'foal', 'folding_chair', 'food_processor', 'football_(American)', 'footstool', 'fork', 'forklift', 'freight_car', 'French_toast', 'freshener', 'frisbee', 'frog', 'fruit_juice', 'frying_pan', 'garbage_truck', 'garden_hose', 'gargle', 'garlic', 'gazelle', 'gelatin', 'giant_panda', 'gift_wrap', 'ginger', 'giraffe', 'cincture', 'glass_(drink_container)', 'globe', 'glove', 'goat', 'goggles', 'golf_club', 'golfcart', 'goose', 'grape', 'grater', 'gravestone', 'green_bean', 'green_onion', 'grill', 'grizzly', 'grocery_bag', 'guitar', 'gull', 'gun', 'hairbrush', 'hairnet', 'hairpin', 'ham', 'hamburger', 'hammer', 'hammock', 'hamster', 'hair_dryer', 'hand_towel', 'handcart', 'handkerchief', 'handle', 'hat', 'veil', 'headband', 'headboard', 'headlight', 'headscarf', 'headstall_(for_horses)', 'heart', 'heater', 'helicopter', 'helmet', 'highchair', 'hinge', 'hog', 'home_plate_(baseball)', 'honey', 'fume_hood', 'hook', 'horse', 'hose', 'hot_sauce', 'hummingbird', 'polar_bear', 'icecream', 'ice_maker', 'igniter', 'iPod', 'iron_(for_clothing)', 'ironing_board', 'jacket', 'jam', 'jar', 'jean', 'jeep', 'jersey', 'jet_plane', 'jewelry', 'jumpsuit', 'kayak', 'kettle', 'key', 'kilt', 'kimono', 'kitchen_sink', 'kite', 'kitten', 'kiwi_fruit', 'knee_pad', 'knife', 'knob', 'ladder', 'ladle', 'ladybug', 'lamb_(animal)', 'lamp', 'lamppost', 'lampshade', 'lantern', 'lanyard', 'laptop_computer', 'latch', 'legging_(clothing)', 'Lego', 'lemon', 'lettuce', 'license_plate', 'life_buoy', 'life_jacket', 'lightbulb', 'lime', 'lion', 'lip_balm', 'lizard', 'log', 'lollipop', 'speaker_(stero_equipment)', 'loveseat', 'magazine', 'magnet', 'mail_slot', 'mailbox_(at_home)', 'mandarin_orange', 'manger', 'manhole', 'map', 'marker', 'mashed_potato', 'mask', 'mast', 'mat_(gym_equipment)', 'mattress', 'measuring_cup', 'measuring_stick', 'meatball', 'medicine', 'melon', 'microphone', 'microwave_oven', 'milk', 'minivan', 'mirror', 'mitten', 'mixer_(kitchen_tool)', 'money', 'monitor_(computer_equipment) computer_monitor', 'monkey', 'motor', 'motor_scooter', 'motorcycle', 'mound_(baseball)', 'mouse_(computer_equipment)', 'mousepad', 'muffin', 'mug', 'mushroom', 'musical_instrument', 'napkin', 'necklace', 'necktie', 'needle', 'nest', 'newspaper', 'newsstand', 'nightshirt', 'noseband_(for_animals)', 'notebook', 'notepad', 'nut', 'oar', 'oil_lamp', 'olive_oil', 'onion', 'orange_(fruit)', 'orange_juice', 'ostrich', 'ottoman', 'oven', 'overalls_(clothing)', 'owl', 'packet', 'pad', 'paddle', 'padlock', 'paintbrush', 'painting', 'pajamas', 'palette', 'pan_(for_cooking)', 'pancake', 'paper_plate', 'paper_towel', 'parachute', 'parakeet', 'parasail_(sports)', 'parasol', 'parka', 'parking_meter', 'parrot', 'passenger_car_(part_of_a_train)', 'passport', 'pastry', 'pea_(food)', 'peach', 'peanut_butter', 'pear', 'peeler_(tool_for_fruit_and_vegetables)', 'pelican', 'pen', 'pencil', 'penguin', 'pepper', 'pepper_mill', 'perfume', 'person', 'pet', 'pew_(church_bench)', 'phonograph_record', 'piano', 'pickle', 'pickup_truck', 'pie', 'pigeon', 'pillow', 'pineapple', 'pinecone', 'pipe', 'pita_(bread)', 'pitcher_(vessel_for_liquid)', 'pizza', 'place_mat', 'plate', 'platter', 'pliers', 'pocketknife', 'poker_(fire_stirring_tool)', 'pole', 'polo_shirt', 'pony', 'pop_(soda)', 'postbox_(public)', 'postcard', 'poster', 'pot', 'flowerpot', 'potato', 'potholder', 'pottery', 'pouch', 'power_shovel', 'prawn', 'pretzel', 'printer', 'projectile_(weapon)', 'projector', 'propeller', 'pumpkin', 'puppy', 'quilt', 'rabbit', 'racket', 'radiator', 'radio_receiver', 'radish', 'raft', 'raincoat', 'ram_(animal)', 'raspberry', 'razorblade', 'reamer_(juicer)', 'rearview_mirror', 'receipt', 'recliner', 'record_player', 'reflector', 'remote_control', 'rhinoceros', 'rifle', 'ring', 'robe', 'rocking_chair', 'rolling_pin', 'router_(computer_equipment)', 'rubber_band', 'runner_(carpet)', 'plastic_bag', 'saddle_(on_an_animal)', 'saddle_blanket', 'saddlebag', 'sail', 'salad', 'salami', 'salmon_(fish)', 'salsa', 'saltshaker', 'sandal_(type_of_shoe)', 'sandwich', 'saucer', 'sausage', 'scale_(measuring_instrument)', 'scarf', 'school_bus', 'scissors', 'scoreboard', 'screwdriver', 'scrubbing_brush', 'sculpture', 'seabird', 'seahorse', 'seashell', 'sewing_machine', 'shaker', 'shampoo', 'shark', 'shaving_cream', 'sheep', 'shield', 'shirt', 'shoe', 'shopping_bag', 'shopping_cart', 'short_pants', 'shoulder_bag', 'shovel', 'shower_head', 'shower_curtain', 'signboard', 'silo', 'sink', 'skateboard', 'skewer', 'ski', 'ski_boot', 'ski_parka', 'ski_pole', 'skirt', 'sled', 'sleeping_bag', 'slipper_(footwear)', 'snowboard', 'snowman', 'snowmobile', 'soap', 'soccer_ball', 'sock', 'sofa', 'solar_array', 'soup', 'soupspoon', 'sour_cream', 'spatula', 'spectacles', 'spice_rack', 'spider', 'sponge', 'spoon', 'sportswear', 'spotlight', 'squirrel', 'stapler_(stapling_machine)', 'starfish', 'statue_(sculpture)', 'steak_(food)', 'steering_wheel', 'step_stool', 'stereo_(sound_system)', 'stirrup', 'stool', 'stop_sign', 'brake_light', 'stove', 'strainer', 'strap', 'straw_(for_drinking)', 'strawberry', 'street_sign', 'streetlight', 'suit_(clothing)', 'sunflower', 'sunglasses', 'sunhat', 'surfboard', 'sushi', 'mop', 'sweat_pants', 'sweatband', 'sweater', 'sweatshirt', 'sweet_potato', 'swimsuit', 'sword', 'table', 'table_lamp', 'tablecloth', 'tag', 'taillight', 'tank_(storage_vessel)', 'tank_top_(clothing)', 'tape_(sticky_cloth_or_paper)', 'tape_measure', 'tapestry', 'tarp', 'tartan', 'tassel', 'tea_bag', 'teacup', 'teakettle', 'teapot', 'teddy_bear', 'telephone', 'telephone_booth', 'telephone_pole', 'television_camera', 'television_set', 'tennis_ball', 'tennis_racket', 'thermometer', 'thermos_bottle', 'thermostat', 'thread', 'thumbtack', 'tiara', 'tiger', 'tights_(clothing)', 'timer', 'tinfoil', 'tinsel', 'tissue_paper', 'toast_(food)', 'toaster', 'toaster_oven', 'toilet', 'toilet_tissue', 'tomato', 'tongs', 'toolbox', 'toothbrush', 'toothpaste', 'toothpick', 'cover', 'tortilla', 'tow_truck', 'towel', 'towel_rack', 'toy', 'tractor_(farm_equipment)', 'traffic_light', 'dirt_bike', 'trailer_truck', 'train_(railroad_vehicle)', 'tray', 'tricycle', 'tripod', 'trousers', 'truck', 'trunk', 'turban', 'turkey_(food)', 'turtle', 'turtleneck_(clothing)', 'typewriter', 'umbrella', 'underwear', 'urinal', 'urn', 'vacuum_cleaner', 'vase', 'vending_machine', 'vent', 'vest', 'videotape', 'volleyball', 'waffle', 'wagon', 'wagon_wheel', 'walking_stick', 'wall_clock', 'wall_socket', 'wallet', 'automatic_washer', 'watch', 'water_bottle', 'water_cooler', 'water_faucet', 'water_jug', 'water_scooter', 'water_ski', 'water_tower', 'watering_can', 'watermelon', 'weathervane', 'webcam', 'wedding_cake', 'wedding_ring', 'wet_suit', 'wheel', 'wheelchair', 'whipped_cream', 'whistle', 'wig', 'wind_chime', 'windmill', 'window_box_(for_plants)', 'windshield_wiper', 'windsock', 'wine_bottle', 'wine_bucket', 'wineglass', 'blinder_(for_horses)', 'wok', 'wooden_spoon', 'wreath', 'wrench', 'wristband', 'wristlet', 'yacht', 'yogurt', 'yoke_(animal_equipment)', 'zebra', 'zucchini'],

    COCO_2017_SPLIT_1: get_oneshot_split(1),
    COCO_2017_SPLIT_2: get_oneshot_split(2),
    COCO_2017_SPLIT_3: get_oneshot_split(3),
    COCO_2017_SPLIT_4: get_oneshot_split(0),

    COCO_2014_FEW_SHOT: fs_coco_2014_seen_classes,
    COCO_2017_FEW_SHOT: fs_coco_2014_seen_classes,
    
    PASCAL_VOC_SPLIT_1: voc_split_1_seen_classes,
    PASCAL_VOC_SPLIT_2: voc_split_2_seen_classes,
    PASCAL_VOC_SPLIT_3: voc_split_3_seen_classes,
}


ALL_CLS_DICT = {
    COCO_OVD: COCO_OVD_ALL_CLS,
    LVIS: MetadataCatalog.get('lvis_v1_train').thing_classes,

    COCO_2017_SPLIT_1: coco17_all_classes,
    COCO_2017_SPLIT_2: coco17_all_classes,
    COCO_2017_SPLIT_3: coco17_all_classes,
    COCO_2017_SPLIT_4: coco17_all_classes,
    
    COCO_2014_FEW_SHOT: fs_coco_2014_all_classes,
    COCO_2017_FEW_SHOT: fs_coco_2014_all_classes,
    
    PASCAL_VOC_SPLIT_1: voc_all_classes_1,
    PASCAL_VOC_SPLIT_2: voc_all_classes_2,
    PASCAL_VOC_SPLIT_3: voc_all_classes_3
}



if __name__ == "__main__":
    print("COCO_2017_SPLIT_1 (seen)", SEEN_CLS_DICT[COCO_2017_SPLIT_1])
    print("COCO_2017_SPLIT_1 (unseen)", [c for c in ALL_CLS_DICT[COCO_2017_SPLIT_1] if c not in SEEN_CLS_DICT[COCO_2017_SPLIT_1]])