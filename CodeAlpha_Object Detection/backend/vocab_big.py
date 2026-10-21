"""
The big everyday-object vocabulary (a few thousand names) used by Auto mode and the target menus.

Each entry of EXTRA is (menu key, menu title, comma separated names). Names are lower case, singular, plain English.
`vocab.py` merges these into the nine original menus (same keys) and adds the new menus, then removes duplicates.
Vehicles are NOT listed here in full - `vehicles.py` holds the detailed car / truck / tempo / bicycle / motorcycle catalogue.
"""
from __future__ import annotations

_RAW = {
"people_vehicles": ("People & Vehicles", """
man, woman, boy, girl, toddler, teenager, elderly person, pedestrian, passenger, driver, rider, motorcyclist, skateboarder, jogger, tourist,
crowd, student, teacher, doctor, nurse, farmer, chef, waiter, security guard, firefighter, construction worker, vendor, street vendor, delivery person,
mechanic, athlete, referee, musician, dancer, pilot, sailor, baby carriage, pram, mobility scooter, electric scooter, kick scooter, hoverboard, unicycle,
tricycle, rickshaw, cycle rickshaw, e-rickshaw, tempo, mini truck, tuk tuk, bullock cart, horse cart, hand cart, pushcart, tanker truck, dump truck,
garbage truck, cement mixer, tow truck, flatbed truck, box truck, semi truck, trailer, caravan, camper van, minibus, school bus, double decker bus,
coach bus, city bus, limousine, convertible, hatchback, sedan, suv, station wagon, sports car, race car, go kart, golf cart, forklift, excavator,
backhoe, road roller, snowplow, street sweeper, bike, mountain bike, road bike, bmx, electric bike, cargo bike, tandem bicycle, moped, scooter motorcycle,
quad bike, atv, snowmobile, sled, sailboat, speedboat, yacht, ferry, cruise ship, cargo ship, fishing boat, rowboat, lifeboat, canoe, raft, paddleboard,
submarine, hot air balloon, glider, jet plane, propeller plane, fighter jet, drone aircraft, rocket, space shuttle, monorail, metro train, locomotive,
freight train, train carriage, cable car, ski lift, gondola, elevator
"""),
"animals": ("Animals", """
sparrow, chickadee, tit bird, wren, warbler, thrush, bluebird, wagtail, nuthatch, red squirrel, grey squirrel, flying squirrel, ground squirrel, dragonfly, damselfly, ladybug, bumblebee, honeybee, mayfly, stag beetle, ground beetle, weevil, cicada, aphid, stick insect, leafhopper, flamingo, swan, duckling, goose, heron bird, mouse, vole, dormouse, marmot, prairie dog, hedgehog, red panda, fox cub, deer fawn, 
puppy, kitten, bull, ox, calf, lamb, ram, ewe, hen, chick, turkey, mule, pony, foal, llama, alpaca, yak, bison, moose, elk, reindeer, antelope, gazelle,
impala, wildebeest, warthog, boar, jaguar, puma, cougar, lynx, bobcat, hyena, jackal, coyote, dingo, meerkat, mongoose, otter, beaver, badger, raccoon,
skunk, weasel, ferret, mink, seal, sea lion, walrus, polar bear, grizzly bear, black bear, koala, wombat, platypus, sloth, armadillo, anteater, bat,
lemur, baboon, orangutan, mandrill, gibbon, tapir, okapi, chipmunk, hamster, guinea pig, mouse animal, gerbil, porcupine, mole, hare, vulture, hawk,
falcon, kite bird, buzzard, osprey, condor, heron, crane bird, stork, pelican, cormorant, ibis, egret, kingfisher, woodpecker, hummingbird, robin,
blue jay, cardinal, finch, canary, swallow, starling, magpie, raven, cockatoo, macaw, parakeet, toucan, hornbill, ostrich, emu, kiwi bird, quail,
pheasant, partridge, dove, mynah, bulbul, sunbird, weaver bird, oriole, lark, nightingale, albatross, puffin, gull, tern, sandpiper, plover, goldfish,
koi, salmon, trout, tuna, catfish, eel, stingray, manta ray, seahorse, clownfish, pufferfish, octopus, squid, starfish, sea urchin, lobster, shrimp,
prawn, oyster, clam, snail, slug, worm, earthworm, centipede, scorpion, tarantula, mosquito, fly, housefly, wasp, hornet, moth, caterpillar, beetle,
cockroach, grasshopper, cricket insect, locust, praying mantis, firefly, termite, tick, flea, tortoise, gecko, chameleon, iguana, komodo dragon,
monitor lizard, python, cobra, viper, rattlesnake, alligator, toad, salamander, newt, bullfrog, cattle, herd, flock of birds, bee hive, spider web
"""),
"carried": ("Carried Items", """
bag, tote bag, duffel bag, messenger bag, laptop bag, sling bag, fanny pack, briefcase, clutch, rucksack, gym bag, trolley bag, carry-on bag,
paper bag, plastic bag, grocery bag, lunch box, tiffin box, thermos, flask, tumbler, mug, sippy cup, juice box, umbrella folded, walking stick, cane,
crutch, selfie stick, tripod, mobile phone, smartphone, feature phone, earphones, earbuds, airpods, headset, microphone, walkie talkie, passport,
ticket, boarding pass, credit card, banknote, coin, coin purse, keychain, key fob, lanyard, badge, name tag, sunglasses case, glasses case, newspaper,
magazine, comic book, notebook carried, diary, planner, clipboard, document folder, file folder, cardboard box, parcel, package, gift box, gift bag,
flower bouquet, pizza box, takeaway container, food delivery bag, baby bottle, pacifier, diaper bag, stuffed toy, kite carried, guitar case, violin case,
sports bag, yoga mat, towel carried, water jug, jerrycan, bucket carried, basket carried, wicker basket, sack, gunny sack, rice sack, milk can, tool bag
"""),
"clothing": ("Clothing & Wear", """
t-shirt, polo shirt, blouse, sweater, hoodie, cardigan, vest, waistcoat, blazer, suit, tuxedo, raincoat, windbreaker, parka, poncho, kurta, sari,
saree, salwar, lehenga, dupatta, sherwani, dhoti, lungi, burqa, abaya, kimono, kilt, skirt, leggings, trousers, pants, cargo pants, track pants,
swimsuit, bikini, swim trunks, pajamas, nightgown, bathrobe, apron, lab coat, overalls, boiler suit, school uniform, police uniform, army uniform,
hi-vis vest, life jacket, bulletproof vest, hard hat, safety helmet, bicycle helmet, motorcycle helmet, baseball cap, beanie, bucket hat, sun hat,
cowboy hat, straw hat, top hat, fedora, beret, headband, bandana, turban cloth, veil, shawl, stole, muffler, bow tie, necktie, socks, stockings,
high heels, flip flops, slippers, loafers, running shoes, football boots, hiking boots, rain boots, gumboots, clogs, mittens, wristwatch, bangle,
anklet, pendant, brooch, cufflinks, hairband, hair clip, scrunchie, wig, backpack straps, face shield, goggles, swimming goggles, ski goggles, mask
"""),
"street": ("Street & Outdoor", """
road, highway, street, sidewalk, pavement, crosswalk, zebra crossing, speed limit sign, one way sign, no entry sign, warning sign, street name sign,
traffic signal, traffic barrier, road divider, guard rail, toll booth, parking lot, parking space, garage, gas station, fuel pump, charging station,
electric pole, power line, transformer, telephone pole, cell tower, satellite dish, antenna, cctv camera, security camera, street light, lamp post,
flyover, overpass, underpass, tunnel, railway station, platform, railway crossing, level crossing barrier, bus shelter, taxi stand, ticket machine,
atm, vending machine, phone booth, newspaper stand, kiosk, food cart, food stall, market stall, shop, storefront, shop sign, awning, shutter,
garbage bin, dustbin, recycling bin, dumpster, drain cover, water tap, drinking fountain, public toilet, bench park, picnic table, playground,
swing set, monkey bars, sandpit, skate park, football field, cricket pitch, basketball court, tennis court, swimming pool, stadium, grandstand,
scoreboard, castle, temple, mosque, church, cathedral, pagoda, minaret, dome, monument, obelisk, lighthouse, windmill, wind turbine, solar panel,
water tank, silo, barn, farmhouse, greenhouse, shed, hut, cottage, apartment building, skyscraper, office building, factory, warehouse, chimney,
construction site, scaffolding, crane tower, brick wall, stone wall, concrete wall, graffiti, mural, poster, banner, hoarding, flagpole, bunting,
rope, chain, wire fence, barbed wire, hedge, lawn, grass, meadow, hill, mountain, valley, cliff, cave, waterfall, river, stream, lake, sea, beach,
sand, pebbles, boulder, log, tree trunk, branch, leaf, palm tree, pine tree, oak tree, banyan tree, bamboo, cactus, sunflower, rose, tulip, lotus,
daisy, orchid, lavender, marigold, jasmine, fern, moss, mushroom wild, haystack, firewood, campfire, bonfire, fireworks, cloud, sun, moon, rainbow,
puddle, snow, snowman, ice, icicle, pothole, speed breaker, road marking, lane marking, stop line, arrow marking
"""),
"sports_toys": ("Sports & Toys", """
rugby ball, tennis ball, ping pong ball, golf ball, bowling ball, bowling pin, cricket stump, cricket helmet, cricket pad, baseball, softball,
hockey puck, ice skates, roller skates, inline skates, skateboard deck, snowboard boots, ski poles, ski helmet, surf fin, paddle, oar, life ring,
swim fins, snorkel, diving mask, scuba tank, fishing rod, fishing net, tackle box, bow and arrow, archery target, dartboard, dart, pool table,
billiard cue, chess board, chess piece, checkers, dominoes, playing cards, dice, board game, jigsaw puzzle, rubik's cube, lego, building blocks,
toy train, toy truck, toy robot, action figure, barbie doll, toy house, toy kitchen, rocking horse, hobby horse, yo-yo, spinning top, marbles,
bubble wand, kite string, paper plane, paper boat, balloon animal, piñata, party hat, confetti, birthday cake, candles birthday, sparkler,
punching bag, boxing ring, kettlebell, barbell, weight plate, treadmill, exercise bike, rowing machine, yoga ball, resistance band, jump rope,
hula hoop, pull-up bar, gymnastics rings, balance beam, pommel horse, hurdle, javelin, discus, shot put, starting block, finish line, whistle,
trophy, medal, podium, scoreboard sports, basketball hoop, football goal, tennis net, volleyball net, badminton shuttlecock, cricket wicket,
golf bag, golf cart sport, climbing rope, carabiner, climbing wall, tent camping, sleeping bag, camping stove, hiking pole, binoculars,
telescope, compass, trampoline sport, bouncy castle, merry go round, ferris wheel, roller coaster, carousel, slide playground
"""),
"food": ("Food & Kitchen", """
bread loaf, baguette, croissant, bagel, toast, pancake, waffle, muffin, cupcake, pie, pastry, brownie, biscuit, cracker, chips, popcorn, pretzel,
french fries, fried chicken, roast chicken, steak, sausage, bacon, ham, meatball, kebab, shawarma, taco, burrito, sushi, dumpling, spring roll,
samosa, pakora, dosa, idli, vada, paratha, chapati, naan, roti, puri, biryani, pulao, curry, dal, paneer, chaat, pav bhaji, vada pav, jalebi,
gulab jamun, ladoo, barfi, halwa, kheer, lassi, chai, pasta, spaghetti, lasagna, ramen, pho, fried rice, paella, risotto, omelette, boiled egg,
fried egg, cheese, butter, yogurt, cream, ice cream cone, popsicle, candy, lollipop, gum, jelly, pudding, honey, jam, peanut butter, nutella,
ketchup, mustard, mayonnaise, soy sauce, chili sauce, salt shaker, pepper shaker, sugar bowl, spice jar, cooking oil, vinegar, flour, wheat,
corn, maize, cob, peas, beans, lentils, chickpeas, peanut, almond, cashew, walnut, pistachio, coconut, date fruit, fig, raisin, pear, peach,
plum, cherry, apricot, pomegranate, papaya, guava, pineapple, kiwi fruit, melon, cantaloupe, lychee, jackfruit, dragon fruit, passion fruit,
blueberry, raspberry, blackberry, cranberry, lime, grapefruit, tangerine, avocado, cabbage, cauliflower, spinach, lettuce, kale, celery,
asparagus, eggplant, brinjal, okra, pumpkin, zucchini, bell pepper, chili pepper, garlic, ginger, radish, beetroot, turnip, sweet potato,
yam, leek, corn on the cob, coffee bean, coffee mug, espresso cup, teapot, tea bag, wine bottle, beer bottle, beer mug, cocktail glass,
champagne flute, soda can, soda bottle, cola, water glass, juice glass, milk carton, milk bottle, frying pan, saucepan, wok, pressure cooker,
rice cooker, blender, mixer grinder, food processor, toaster oven, air fryer, kettle electric, induction cooktop, gas stove, gas cylinder,
chopping board, rolling pin, whisk, ladle, spatula, tongs, peeler, grater, colander, strainer, measuring cup, measuring spoon, mortar and pestle,
can opener, bottle opener, corkscrew, lunch tray, serving dish, casserole, baking tray, cake stand, cookie jar, butter knife, steak knife,
cleaver, fruit basket, bread basket, picnic basket, thali, banana leaf, tiffin carrier, water jug kitchen, pitcher, carafe, teacup, saucer
"""),
"home": ("Home & Furniture", """
sofa, armchair, recliner, stool, bar stool, ottoman, bean bag, bunk bed, crib, cradle, hammock, coffee table, side table, nightstand, dresser,
chest of drawers, tv stand, bookcase, display cabinet, kitchen cabinet, pantry, counter, kitchen island, headboard, futon, rocking chair,
high chair, office chair, folding chair, plastic chair, wooden chair, bench indoor, doorbell, door handle, door knob, lock, padlock, key,
hinge, doorstep, doormat, curtain rod, blinds, window sill, balcony, railing, banister, staircase, ladder, step stool, skylight, ceiling fan,
chandelier, table lamp, floor lamp, wall lamp, night light, flashlight, lantern, light switch, power socket, extension cord, plug, battery,
smoke detector, thermostat, radiator, heater, water heater, geyser, air cooler, humidifier, air purifier, iron, ironing board, clothes line,
clothes basket, laundry basket, washing basket, drying rack, broom, mop, dustpan, duster, sponge, scrub brush, spray bottle, detergent bottle,
soap, soap dispenser, shampoo bottle, toothpaste, razor, comb, hairbrush, hair dryer, mirror bathroom, shower, shower head, faucet, bathtub tap,
toilet paper, tissue box, trash can indoor, laundry machine, dishwasher, freezer, range hood, wall clock, alarm clock, grandfather clock, hourglass,
photo frame, painting, wall art, poster wall, rug, doormat indoor, cushion, bedsheet, comforter, quilt, bunk, tapestry, vase flowers, flower pot,
plant pot, watering can, garden hose, lawn mower, rake, shovel, spade, hoe, wheelbarrow, pruning shears, garden gnome, bird house, bird feeder,
doghouse, pet bed, pet bowl, leash, collar, cat tree, litter box, fish tank, birdcage, hamster cage, piggy bank, safe box, shoe rack, coat rack,
umbrella stand, hat stand, bookends, candle holder, incense stick, idol, statue small, crucifix, prayer mat, rosary, bell, wind chime, dreamcatcher
"""),
"electronics": ("Electronics & Office", """
desktop computer, computer tower, monitor, laptop computer, notebook computer, chromebook, tablet computer, ipad, e-reader, kindle, smartphone screen,
mobile charger, usb cable, usb drive, hard drive, memory card, sd card, webcam, microphone desk, headphones over ear, gaming headset, speaker bluetooth,
soundbar, home theater, dvd player, cd, vinyl record, record player, radio, walkman, mp3 player, camera dslr, mirrorless camera, film camera, polaroid,
camcorder, gopro, action camera, camera lens, camera flash, tripod stand, ring light, studio light, drone quadcopter, vr headset, game console,
playstation, xbox, nintendo switch, joystick, gamepad, arcade machine, television, smart tv, projector screen, set top box, modem, wifi router,
network switch, server rack, cable bundle, power strip, ups, inverter, generator, solar charger, power bank portable, battery pack, led strip,
tube light, cfl bulb, led bulb, traffic camera, barcode scanner, cash register, pos terminal, credit card machine, receipt, label printer,
3d printer, scanner, photocopier, fax machine, shredder, laminator, paper tray, paper clip, binder clip, rubber band, push pin, thumbtack,
tape dispenser, scotch tape, glue stick, glue bottle, scissors office, paper cutter, hole punch, sticky note, post-it, highlighter, ballpoint pen,
fountain pen, pencil case, pencil sharpener, crayon, colored pencil, paintbrush, paint palette, watercolor, canvas, easel, sketchbook, drawing board,
t-square, protractor, compass drawing, set square, abacus, slide rule, calendar, desk calendar, planner book, whiteboard marker, chalk, chalkboard,
duster board, projector slide, lecture podium, school desk, school bench, school bus stop, locker, bulletin board, notice board, map wall, globe desk,
microscope, test tube, beaker, flask lab, bunsen burner, stethoscope, thermometer, syringe, pill bottle, first aid kit, bandage, wheelchair medical,
hospital bed, iv stand, x-ray, mri machine, defibrillator, oxygen cylinder, fire extinguisher, fire alarm, exit sign, emergency light, siren
"""),
"tools": ("Tools & Hardware", """
hammer, mallet, sledgehammer, screwdriver, wrench, spanner, pliers, wire cutter, bolt cutter, crowbar, chisel, saw, hand saw, chainsaw, hacksaw,
drill, power drill, drill bit, angle grinder, sander, jigsaw tool, circular saw, welding machine, welding mask, soldering iron, multimeter,
tape measure, spirit level, plumb bob, square tool, clamp, vise, anvil, workbench, toolbox, tool belt, nail, screw, bolt, nut, washer, rivet,
staple, hook, bracket, hinge hardware, pipe, pvc pipe, hose, valve, pipe wrench, plunger, ladder extension, scaffold plank, paint roller,
paint can, paint tray, putty knife, trowel, cement bag, brick, concrete block, tile, plywood, plank, wooden beam, steel beam, rebar, wire mesh,
rope coil, cable drum, pulley, winch, hook crane, jack, car jack, tire, tyre, wheel, rim, spare tire, car battery, jumper cable, engine,
exhaust pipe, headlight, tail light, side mirror, license plate, number plate, windshield, wiper, steering wheel, gear lever, dashboard,
seat belt, airbag, car seat, child seat, roof rack, bike rack, bike lock, bike pump, bike bell, bike basket, bike saddle, handlebar, pedal,
bike chain, bike wheel, kickstand, mudguard, fuel can, oil can, grease gun, funnel, tarpaulin, tarp, plastic sheet, sandbag, barrel, drum,
crate, pallet, cardboard carton, shipping container, trolley cart, hand truck, sack truck, conveyor belt, shelf rack, steel rack, gas cylinder industrial
"""),
"music_art": ("Music, Art & Culture", """
guitar, electric guitar, bass guitar, acoustic guitar, ukulele, banjo, violin, viola, cello, double bass, harp, sitar, tabla, harmonium, sarod,
veena, flute, bansuri, clarinet, oboe, saxophone, trumpet, trombone, tuba, french horn, harmonica, accordion, piano, grand piano, keyboard piano,
synthesizer, organ, drum, drum kit, snare drum, bass drum, dhol, dholak, mridangam, tambourine, maracas, xylophone, cymbal, triangle instrument,
conga, bongo, djembe, bagpipes, music stand, sheet music stand, microphone stand, amplifier, mixing console, dj turntable, stage, curtain stage,
spotlight, orchestra, choir, puppet, marionette, mask theatre, costume, mannequin, sculpture, bust, figurine, pottery, clay pot, earthen pot,
ceramic bowl, porcelain, tapestry art, mosaic, stained glass, rangoli, mandala, calligraphy, book open, bookshelf library, library, museum exhibit,
cinema screen, movie camera, clapperboard, film reel, popcorn bucket, ticket booth, theatre seat, balcony seat, flag national, coat of arms, crown,
sword, shield, helmet armor, armor, throne, scepter, chess king, trophy cup, candle altar, diya, lamp oil, incense burner, church bell, cross,
star of david, om symbol, buddha statue, ganesha idol, prayer flag, prayer wheel, torii gate, pagoda roof, lantern festival, kite festival
"""),
"nature": ("Plants & Nature", """
toadstool, fly agaric mushroom, bolete mushroom, chanterelle mushroom, tree stump, abutilon flower, chinese lantern flower, red flower, yellow flower, pink flower, white flower, purple flower, berry, acorn, pine cone, seed pod, dewdrop, spider on web, 
tree, oak, maple, birch, willow, cedar, redwood, eucalyptus, mango tree, neem tree, coconut palm, date palm, banana plant, papaya tree, apple tree,
orange tree, lemon tree, olive tree, cherry blossom, magnolia, hibiscus, bougainvillea, chrysanthemum, dahlia, lily, water lily, carnation, peony,
poppy, lilac, jasmine flower, frangipani, gerbera, petunia, pansy, geranium, begonia, succulent, aloe vera, snake plant, monstera, money plant,
tulsi plant, ivy, vine, creeper, seaweed, coral, reef, kelp, grass blade, reed, wheat field, rice paddy, corn field, tea plantation, vineyard,
orchard, farm field, tractor field, hay bale, scarecrow, irrigation channel, well, pond water, marsh, swamp, delta, island, peninsula, volcano,
glacier, iceberg, desert, dune, canyon, plateau, forest, jungle, rainforest, savanna, tundra, meadow flowers, garden, park, botanical garden,
sky, sunset, sunrise, star, galaxy, planet, comet, lightning, tornado, storm cloud, fog, smoke, steam, dust, flame, spark, ember, wave, foam,
shell, seashell, conch, pearl, driftwood, feather, nest, egg bird, cocoon, honeycomb, anthill, termite mound, spider egg sac, pawprint, footprint
"""),
}

EXTRA: list[tuple[str, str, list[str]]] = []
for _key, (_title, _blob) in _RAW.items():
    _names = []
    for _n in _blob.replace("\n", " ").split(","):
        _n = " ".join(_n.strip().lower().split())
        if _n and _n not in _names:
            _names.append(_n)
    EXTRA.append((_key, _title, _names))
