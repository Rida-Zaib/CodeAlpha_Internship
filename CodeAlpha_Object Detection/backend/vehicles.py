"""
Vehicle types and models - a free-format second opinion on every car, truck, bus, bicycle and motorcycle.

YOLO only knows "car / truck / bus / motorcycle / bicycle" (and gets car vs truck wrong now and then). For every
confirmed vehicle track we take its best 1-4 crops and ask CLIP (zero-shot, no training) which of ~1,000 type and
model names fits best:

  1. TYPE   - hatchback, SUV, pickup, tempo, auto rickshaw, dump truck, road bicycle, cruiser motorcycle ...
  2. MODEL  - "Maruti Suzuki Swift", "Tata Ace", "Royal Enfield Classic 350", "Trek Marlin" ... (only if confident)
  3. CLASS  - if CLIP strongly disagrees with YOLO on the broad class (a car labelled truck), the class is corrected.

Free format: any names typed into "Find anything by name" are added to the candidate list for this run, so
"mahindra bolero pickup, hero splendor" can be searched for even if it is not in the built-in catalogue.

Honesty note: CLIP is a general image-text model, not a trained car-model recogniser. Types are usually right;
exact models are an educated guess that is only shown when the match is clear, and they work best on a close,
sharp, side or three-quarter view. Small or blurry vehicles fall back to the type (or no label).
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

# broad YOLO class -> list of types
TYPES: dict[str, list[str]] = {
    "car": ["hatchback", "sedan", "suv", "compact suv", "crossover", "minivan", "mpv", "station wagon", "coupe",
            "convertible", "sports car", "supercar", "luxury car", "limousine", "electric car", "vintage car", "hot rod",
            "microcar", "jeep", "taxi", "police car", "ambulance", "pickup truck", "kei car", "muscle car", "off-road suv"],
    "truck": ["mini truck", "tempo", "pickup tempo", "delivery van", "panel van", "cargo van", "box truck", "flatbed truck",
              "dump truck", "tipper truck", "tanker truck", "tow truck", "garbage truck", "cement mixer", "container truck",
              "trailer truck", "semi truck", "fire truck", "car carrier truck", "logging truck", "refrigerated truck",
              "tractor", "tractor trolley", "light commercial vehicle", "heavy commercial vehicle", "water tanker", "lorry",
              "pickup truck", "auto rickshaw", "cargo auto rickshaw", "e-rickshaw", "tuk tuk", "three wheeler cargo"],
    "bus": ["city bus", "school bus", "double decker bus", "coach bus", "minibus", "tempo traveller", "electric bus",
            "articulated bus", "open top bus", "sleeper bus", "tram", "trolleybus"],
    "motorcycle": ["scooter", "moped", "cruiser motorcycle", "sport bike", "naked bike", "touring motorcycle",
                   "adventure motorcycle", "cafe racer", "dirt bike", "motocross bike", "electric scooter", "electric motorcycle",
                   "commuter motorcycle", "chopper", "classic motorcycle", "quad bike", "sidecar motorcycle", "trike",
                   "auto rickshaw", "e-rickshaw", "tuk tuk"],
    "bicycle": ["road bicycle", "mountain bike", "bmx bike", "hybrid bicycle", "city bicycle", "cargo bike", "folding bicycle",
                "tandem bicycle", "electric bicycle", "kids bicycle", "fat bike", "gravel bike", "touring bicycle",
                "fixed gear bicycle", "recumbent bicycle", "tricycle", "cycle rickshaw", "lowrider bicycle", "unicycle"],
    "train": ["metro train", "steam locomotive", "diesel locomotive", "electric locomotive", "bullet train", "freight train",
              "passenger train", "local train", "monorail", "tram car", "toy train"],
    "boat": ["sailboat", "speedboat", "yacht", "ferry", "cruise ship", "cargo ship", "fishing boat", "rowboat", "canoe",
             "kayak", "houseboat", "tugboat", "jet ski", "dinghy", "catamaran", "gondola", "lifeboat", "container ship"],
    "airplane": ["passenger jet", "propeller plane", "fighter jet", "cargo plane", "private jet", "helicopter", "glider",
                 "seaplane", "biplane", "drone aircraft", "hot air balloon", "airship"],
}

# which fine-model list belongs to which coarse type (a type only matches models of its own group)
TYPE_GROUP = {t: g for g, ts in TYPES.items() for t in ts}
for _t in ("pickup truck", "auto rickshaw", "e-rickshaw", "tuk tuk"):
    TYPE_GROUP[_t] = "truck" if _t != "pickup truck" else "car"
TYPE_GROUP["auto rickshaw"] = TYPE_GROUP["e-rickshaw"] = TYPE_GROUP["tuk tuk"] = "truck"  # 3-wheelers: own list below

_MAKES = {
    "car": """
Maruti Suzuki: Alto 800, Alto K10, S-Presso, Wagon R, Celerio, Swift, Dzire, Baleno, Ignis, Ertiga, XL6, Brezza, Fronx, Grand Vitara, Jimny, Invicto, Eeco, Ciaz, Omni, Maruti 800, Zen, Esteem, Gypsy;
Hyundai: Santro, Grand i10, i10, i20, Aura, Verna, Elantra, Creta, Venue, Exter, Alcazar, Tucson, Ioniq 5, Kona Electric, Accent, Eon, Xcent, Sonata;
Tata: Tiago, Tigor, Altroz, Punch, Nexon, Harrier, Safari, Curvv, Nano, Indica, Indigo, Sumo, Tata Nexon EV, Tata Tiago EV;
Mahindra: Thar, Scorpio, Scorpio N, Bolero, Bolero Neo, XUV300, XUV400, XUV500, XUV700, XUV3XO, Marazzo, KUV100, TUV300, Xylo, Verito, BE 6, XEV 9e;
Honda: City, Amaze, Jazz, WR-V, Elevate, Civic, Accord, CR-V, Brio, Fit, Odyssey, HR-V, Pilot, Passport, Ridgeline;
Toyota: Innova, Innova Crysta, Innova Hycross, Fortuner, Camry, Corolla, Corolla Altis, Etios, Glanza, Urban Cruiser Hyryder, Hilux, Land Cruiser, Prius, RAV4, Yaris, Vellfire, Highlander, Tacoma, Tundra, Sienna, Supra, GR86;
Kia: Seltos, Sonet, Carens, Carnival, EV6, Sportage, Sorento, Picanto, Rio, Telluride, Stinger;
Renault: Kwid, Triber, Kiger, Duster, Captur, Clio, Megane, Scenic, Twingo, Zoe;
Nissan: Magnite, Micra, Sunny, Terrano, Kicks, X-Trail, Qashqai, Leaf, Juke, Altima, Rogue, GT-R, Patrol, Navara, Skyline;
Volkswagen: Polo, Vento, Virtus, Taigun, Tiguan, Passat, Golf, Jetta, Beetle, Touareg, ID.4, Arteon, Up, T-Roc, Transporter;
Skoda: Rapid, Slavia, Kushaq, Kodiaq, Superb, Octavia, Fabia, Karoq, Kamiq, Yeti;
Ford: EcoSport, Figo, Aspire, Endeavour, Fiesta, Focus, Mustang, F-150, Explorer, Escape, Bronco, Ranger, Fusion, Edge, Expedition, Maverick;
Chevrolet: Beat, Spark, Sail, Cruze, Aveo, Enjoy, Tavera, Captiva, Camaro, Corvette, Silverado, Malibu, Tahoe, Suburban, Impala, Equinox, Bolt;
MG: Hector, Astor, ZS EV, Gloster, Comet EV, Windsor EV;
Jeep: Compass, Meridian, Wrangler, Grand Cherokee, Renegade, Gladiator, Cherokee;
BMW: 1 Series, 2 Series, 3 Series, 5 Series, 7 Series, X1, X3, X5, X7, Z4, M3, M4, M5, i4, iX, Mini Cooper;
Mercedes-Benz: A-Class, C-Class, E-Class, S-Class, GLA, GLC, GLE, GLS, G-Class, CLA, AMG GT, EQS, Maybach;
Audi: A3, A4, A6, A8, Q3, Q5, Q7, Q8, TT, R8, e-tron;
Volvo: XC40, XC60, XC90, S60, S90, V40;
Lexus: ES, RX, NX, LX, LS, UX;
Porsche: 911, Cayenne, Macan, Panamera, Taycan, Boxster;
Land Rover: Defender, Discovery, Range Rover, Range Rover Sport, Evoque, Velar;
Jaguar: XE, XF, F-Pace, F-Type;
Tesla: Model S, Model 3, Model X, Model Y, Cybertruck;
Ferrari: 488, F8 Tributo, Roma, Portofino, SF90, 812;
Lamborghini: Huracan, Aventador, Urus;
Bentley: Continental GT, Bentayga;
Rolls-Royce: Ghost, Phantom, Cullinan;
Peugeot: 208, 308, 3008, 5008;
Citroen: C3, C5 Aircross, 2CV, Berlingo;
Fiat: 500, Punto, Linea, Panda, Uno, Premier Padmini;
Hindustan Motors: Ambassador;
Datsun: Go, Redi-Go;
Suzuki: Vitara, Swift Sport, Jimny Sierra, Samurai;
Mazda: Mazda3, Mazda6, CX-5, MX-5 Miata, CX-30;
Subaru: Impreza, Outback, Forester, WRX, Crosstrek, BRZ;
Mitsubishi: Pajero, Pajero Sport, Outlander, Lancer, Montero, Eclipse Cross;
Isuzu: D-Max, MU-X, V-Cross;
BYD: Atto 3, Seal, e6, Dolphin;
Dacia: Duster, Sandero, Logan;
Opel: Corsa, Astra;
Seat: Ibiza, Leon;
Alfa Romeo: Giulia, Stelvio;
Maserati: Ghibli, Levante;
Aston Martin: DB11, Vantage;
McLaren: 720S;
Bugatti: Chiron;
Genesis: G70, G80, GV70""",
    "truck": """
Tata Motors: Ace, Ace Gold, Intra, Yodha, Xenon, 407, 709, Prima, Signa, Ultra, LPT 1613, Magic, Winger;
Ashok Leyland: Dost, Bada Dost, Partner, Boss, Ecomet, Captain, U-Truck, Viking, Stile;
Mahindra: Bolero Pickup, Bolero Maxx, Jeeto, Supro, Furio, Blazo, Imperio, Loadking, Pik-Up, Scorpio Getaway, Treo Zor;
Eicher: Pro 2049, Pro 3015, Pro 6025, Pro 8031, Terra 16;
BharatBenz: 1215, 1617, 2823, 3523, 5528;
Force Motors: Traveller, Trax, Gurkha, Shaktiman;
Isuzu: D-Max V-Cross, NPR, ELF, Giga;
Maruti Suzuki: Super Carry, Eeco Cargo, Omni Cargo;
Ford: F-150, F-250, F-350, Ranger, Transit, Super Duty, Maverick;
Chevrolet: Silverado, Colorado, Express, Avalanche;
Ram: 1500, 2500, 3500, ProMaster;
GMC: Sierra, Canyon, Savana, Hummer EV;
Toyota: Hilux, Tacoma, Tundra, HiAce, Land Cruiser 70, Dyna;
Nissan: Navara, Frontier, Titan, NV200, Urvan, Cabstar;
Mitsubishi: L200, Triton, Canter, Fuso Fighter;
Mercedes-Benz: Sprinter, Actros, Atego, Arocs, Unimog, Vito, Econic;
Volvo Trucks: FH, FM, FMX, FE, VNL;
Scania: R series, S series, G series, P series;
MAN: TGX, TGS, TGL, TGE;
DAF: XF, CF, LF, XG;
Iveco: Daily, Stralis, S-Way, Eurocargo;
Renault Trucks: T, C, D, Master;
Freightliner: Cascadia, M2, Columbia;
Kenworth: T680, T800, W900;
Peterbilt: 579, 389, 567;
Mack: Anthem, Granite, Pinnacle;
International: LT, HX, ProStar;
Volkswagen: Transporter, Crafter, Amarok, Caddy;
Hyundai: Porter, H-100, Mighty, Xcient;
Kia: Bongo, K2500;
Hino: 300, 500, 700;
Kamaz: 5320, 6520;
Tatra: T815;
Piaggio: Ape Xtra, Ape City, Ape Auto DX, Porter 700;
Bajaj: Maxima C, Maxima Cargo, Qute;
TVS: King, King Deluxe;
Atul: Gem, Shakti, Smart;
Mahindra Treo: Treo, Treo Yaari, Alfa;
Piaggio Ape: E-City, E-Xtra, Ape Auto""",
    "bus": """
Tata: Starbus, Marcopolo, Ultra, Winger, Cityride, LP 1512, Magna;
Ashok Leyland: Viking, Lynx, Sunshine, Oyster, Cheetah, Circuit;
Eicher: Skyline, Starline, Pro 3000;
Volvo: 9400, 9700, B7R, B9R, 8400, 7900 Electric;
Mercedes-Benz: Citaro, Tourismo, Sprinter bus, Conecto, Travego;
Scania: Touring, Irizar i6, Citywide;
MAN: Lion's City, Lion's Coach;
Leyland: Titan, National, Olympian, Atlantean;
AEC: Routemaster, Regent;
Alexander Dennis: Enviro400, Enviro200;
BYD: K9, K7, K8, eBus;
Olectra: eBuzz K9;
Switch: EiV 12, EiV 22;
Force: Traveller 26, Traveller 3350;
Mahindra: Tourister, Cruzio;
Blue Bird: Vision, All American;
Thomas Built: Saf-T-Liner;
New Flyer: Xcelsior;
Gillig: Low Floor;
Setra: S 515, S 531;
Neoplan: Cityliner, Skyliner;
Irizar: i8;
Van Hool: TX;
Toyota: Coaster, HiAce;
Nissan: Civilian""",
    "motorcycle": """
Royal Enfield: Classic 350, Bullet 350, Hunter 350, Meteor 350, Thunderbird, Himalayan, Scram 411, Interceptor 650, Continental GT 650, Super Meteor 650, Shotgun 650, Guerrilla 450, Electra, Standard;
Hero: Splendor, Splendor Plus, HF Deluxe, Passion Pro, Glamour, Xtreme 160R, Xpulse 200, Karizma, Maestro Edge, Pleasure, Destini, Super Splendor, CD 100, Hunk, Achiever, Impulse;
Honda: Activa, Dio, Shine, Unicorn, Hornet 2.0, CB350, CB300R, SP125, Livo, CBR650R, Africa Twin, Gold Wing, CB500X, Rebel, Navi, Grazia, Aviator, Cliq, Transalp, Monkey, Super Cub, CBR1000RR;
Bajaj: Pulsar 150, Pulsar 220F, Pulsar NS200, Pulsar RS200, Pulsar N250, Platina, CT100, Avenger, Dominar 400, Chetak, Discover, V15, XCD, Boxer, KB125;
TVS: Apache RTR 160, Apache RTR 200 4V, Apache RR 310, Jupiter, Ntorq, Star City, Sport, Radeon, Raider 125, XL100, Scooty Pep, iQube, Victor, Phoenix, Ronin;
Yamaha: R15, R3, MT-15, MT-03, MT-07, MT-09, FZ-S, FZ-X, Fascino, Ray ZR, Aerox 155, RX100, Crux, YZF-R1, YZF-R6, XSR900, Tenere 700, Tracer 9;
Suzuki: Access 125, Burgman, Gixxer, Gixxer SF, Intruder, V-Strom, Hayabusa, GSX-R1000, GSX-S750, Avenis, Samurai, Katana;
KTM: Duke 125, Duke 200, Duke 250, Duke 390, RC 200, RC 390, Adventure 390, 890 Adventure, 1290 Super Duke R, 790 Duke, EXC, SX-F;
Kawasaki: Ninja 300, Ninja 400, Ninja ZX-10R, Ninja H2, Z650, Z900, Versys 650, W175, Vulcan S, KLX 230, Z H2;
Harley-Davidson: Street 750, Iron 883, Fat Boy, Fat Bob, Road King, Street Glide, Sportster S, Nightster, Pan America, Softail Standard, Heritage Classic, Low Rider, LiveWire;
Ducati: Panigale V4, Monster, Scrambler, Multistrada, Diavel, Streetfighter V4, Hypermotard, DesertX;
Triumph: Bonneville T120, Street Triple, Speed Triple, Tiger 900, Scrambler 400 X, Speed 400, Rocket 3, Thruxton;
BMW Motorrad: G 310 R, G 310 GS, S 1000 RR, R 1250 GS, R nineT, F 850 GS, K 1600;
Aprilia: RS 457, RSV4, Tuono, SR 160, SXR 160;
Vespa: Primavera, GTS, Sprint, Elettrica, LX 125;
Mahindra: Mojo, Centuro, Gusto, Duro, Pantero, Rodeo;
Ola: S1 Pro, S1 Air;
Ather: 450X, 450S, Rizta;
Okinawa: i-Praise, Praise Pro;
Revolt: RV400;
Jawa: 42, Perak, Yezdi Roadster, Jawa 350;
LML: Vespa, Freedom, Star;
Bajaj Chetak: Chetak Classic;
Lambretta: LI 150;
Piaggio: Ape;
Indian: Scout, Chief, FTR;
Husqvarna: Svartpilen 401, Vitpilen 250;
BSA: Gold Star;
Moto Guzzi: V7, V85 TT;
MV Agusta: Brutale, F3;
CFMoto: 650NK, 300SR;
Benelli: TRK 502, Imperiale 400, Leoncino;
Keeway: K300 N, SR 250;
Zero: SR/F, DS;
Energica: Ego""",
    "bicycle": """
Hero Cycles: Sprint, Lectro, Octane, Hero Ranger, Hero Jet;
Atlas: Atlas Golden Line, Atlas Gold, Atlas Saathi, Atlas Roadster;
Hercules: Hercules Roadeo, Hercules Street Cat, Hercules Ryder, Hercules ThunderBolt, Hercules MTB;
Avon: Avon Ranger, Avon Wheelie;
BSA: BSA Photon, BSA Ladybird, BSA Champ, BSA Mach City, BSA Hercules;
Firefox: Bad Attitude, Rapide, Typhoon;
Trek: Marlin 5, Marlin 7, FX 2, Domane, Madone, Emonda, Fuel EX, Dual Sport, Verve;
Giant: Escape 3, Talon, Contend, TCR, Defy, Trance, Revolt, Roam;
Specialized: Rockhopper, Sirrus, Allez, Tarmac, Roubaix, Stumpjumper, Turbo Vado, Diverge;
Cannondale: Quick, Synapse, CAAD13, Trail, Topstone, SuperSix Evo;
Scott: Scale, Speedster, Addict, Aspect, Spark;
Cube: Aim, Reaction, Attention, Nuroad;
Merida: Big Nine, Scultura, Reacto, Crossway;
Bianchi: Oltre, Specialissima, Impulso;
Pinarello: Dogma;
Cervelo: S5, R5, Caledonia;
Canyon: Endurace, Aeroad, Ultimate, Grizl;
GT: Aggressor, Avalanche;
Schwinn: Sting-Ray, Traveler, Phocus;
Raleigh: Chopper, Grifter, Burner;
Brompton: M6L, P Line, C Line;
Dahon: Vitesse, Mu;
Tern: Verge, GSD, HSD;
Rad Power: RadWagon, RadCity, RadRover;
Decathlon: Rockrider ST 100, Riverside 500, Triban RC 500, Btwin Elops, Btwin My Bike;
Btwin: Rockrider 520, Elops 120, Original 100;
Montra: Helicon, Blues, Rock;
Kross: Level, Trans;
Mongoose: Legion, Switchback;
Huffy: Cranbrook, Stone Mountain;
Dutch: Gazelle Chamonix, Batavus, Omafiets;
Mobike: Mobike share;
Lime: Lime e-bike;
Yulu: Miracle, Move;
Bounce: Bounce share""",
}

THREE_WHEELERS = ["Bajaj RE", "Bajaj Maxima", "Bajaj Compact", "Bajaj GoGo", "Piaggio Ape", "Piaggio Ape City", "Piaggio Ape Xtra LDX",
                  "Mahindra Treo", "Mahindra Alfa", "Mahindra Jeeto", "TVS King", "Atul Auto", "Atul Gem", "Force Minidor",
                  "Lohia Humsafar", "Mini Metro e-rickshaw", "Saarthi e-rickshaw", "YC Electric", "Kinetic Safar", "Scooter Vikram",
                  "Tata Magic Iris", "Tata Ace Zip", "Euler HiLoad", "Greaves Eltra", "Altigreen neEV", "Bajaj Qute", "Piaggio Porter 700"]


def _expand(blob: str) -> list[str]:
    out: list[str] = []
    for part in blob.replace("\n", " ").split(";"):
        if ":" not in part:
            continue
        make, models = part.split(":", 1)
        make = make.strip()
        for m in models.split(","):
            m = m.strip()
            if not m:
                continue
            label = m if m.lower().startswith(make.lower().split()[0]) else f"{make} {m}"
            out.append(" ".join(label.split()))
    return list(dict.fromkeys(out))


MODELS: dict[str, list[str]] = {k: _expand(v) for k, v in _MAKES.items()}
MODELS["truck"] = list(dict.fromkeys(MODELS["truck"] + THREE_WHEELERS))
MODELS["train"] = MODELS["boat"] = MODELS["airplane"] = []
FAMILY_OF_TYPE_GROUP = {"car", "truck", "bus", "motorcycle", "bicycle", "train", "boat", "airplane"}

# names added to the website's People & Vehicles menu / Auto vocabulary (types + three-wheelers; models are matched by CLIP, not by YOLO)
VEHICLE_MENU_NAMES: list[str] = list(dict.fromkeys(t for ts in TYPES.values() for t in ts))

BASE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle", "train", "boat", "airplane"}
CORRECTABLE = {"car", "truck", "bus", "motorcycle", "bicycle"}


def class_of_type(t: str) -> Optional[str]:
    """Broad YOLO class that a type belongs to (None when it is ambiguous, e.g. a 3-wheeler)."""
    if t in ("auto rickshaw", "cargo auto rickshaw", "e-rickshaw", "tuk tuk", "three wheeler cargo", "pickup truck", "tricycle", "trike", "quad bike", "tram", "tram car"):
        return None
    for g, ts in TYPES.items():
        if t in ts:
            return g
    return None


class VehicleClassifier:
    """CLIP zero-shot. `analyzer` must offer embed_images(list[BGR]) and embed_texts(list[str]) -> L2-normalised ndarray."""

    ACCEPT_MODEL = 0.12
    ACCEPT_TYPE = 0.12
    OVERRIDE_CLASS = 0.60

    def __init__(self, analyzer):
        self.a = analyzer
        self._text: dict[str, np.ndarray] = {}

    def _emb(self, labels: list[str], prefix: str) -> np.ndarray:
        miss = [l for l in labels if prefix + l not in self._text]
        if miss:
            v = self.a.embed_texts([f"a photo of a {l}" if prefix == "t:" else f"a photo of a {l}, a vehicle" for l in miss])
            for l, x in zip(miss, v):
                self._text[prefix + l] = x
        return np.stack([self._text[prefix + l] for l in labels])

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def classify(self, crops: dict[int, list], base: dict[int, str], extra: Optional[list[str]] = None) -> dict[int, dict]:
        """crops: track id -> list of BGR crops; base: track id -> YOLO class. Returns per-track dict."""
        extra = [e for e in (extra or []) if e]
        ids = [i for i in crops if crops[i]]
        if not ids:
            return {}
        flat = [c for i in ids for c in crops[i]]
        emb = self.a.embed_images(flat)
        out: dict[int, dict] = {}
        pos = 0
        all_types = list(dict.fromkeys(t for ts in TYPES.values() for t in ts))
        T = self._emb(all_types, "t:")
        feats = []
        for i in ids:
            n = len(crops[i])
            f = emb[pos:pos + n].mean(axis=0)
            pos += n
            feats.append(f / (np.linalg.norm(f) + 1e-9))
        # CLIP likes some labels (e.g. "compact suv") for everything; remove half of each label's average pull across this run
        bias = 0.5 * (T @ np.stack(feats).T).mean(axis=1) if len(feats) >= 4 else np.zeros(len(all_types))
        for i, f in zip(ids, feats):
            p = self._softmax(100.0 * (T @ f - bias))
            # restrict to types of the YOLO family (plus 3-wheelers) unless CLIP is clearly sure of another family
            k = int(np.argmax(p))
            ctype, pc = all_types[k], float(p[k])
            if base.get(i) in TYPES:   # keep the guess inside the detector's family unless CLIP is very sure it is another one
                fam = [t for t in all_types if base[i] == class_of_type(t) or class_of_type(t) is None]
                if class_of_type(ctype) not in (None, base[i]) and pc < self.OVERRIDE_CLASS and fam:
                    sub = [all_types.index(t) for t in fam]
                    k = sub[int(np.argmax(p[sub]))]
                    ctype, pc = all_types[k], float(p[k])
            group = class_of_type(ctype) or base.get(i, "car")
            if group not in MODELS:
                group = base.get(i, "car")
            cands = MODELS.get(group, [])
            if ctype in ("auto rickshaw", "cargo auto rickshaw", "e-rickshaw", "tuk tuk", "three wheeler cargo"):
                cands = THREE_WHEELERS
            pool = list(dict.fromkeys(cands + extra))
            model, pm = None, 0.0
            if pool:
                M = self._emb(pool, "m:")
                q = self._softmax(100.0 * (M @ f))
                j = int(np.argmax(q))
                thr = self.ACCEPT_MODEL if pool[j] not in extra else 0.15
                if q[j] >= thr:
                    model, pm = pool[j], float(q[j])
            override = None
            want = class_of_type(ctype)
            if want and want != base.get(i) and base.get(i) in CORRECTABLE and want in CORRECTABLE and pc >= self.OVERRIDE_CLASS:
                override = want
            label = None
            if model:
                label = f"{model} ({ctype})"
            elif pc >= self.ACCEPT_TYPE:
                label = ctype
            out[i] = {"type": ctype if pc >= self.ACCEPT_TYPE else None, "type_score": round(pc, 2), "model": model,
                      "model_score": round(pm, 2), "label": label, "class_override": override}
        return out


def best_crop(frame: np.ndarray, box, pad: float = 0.04, side: int = 224) -> Optional[np.ndarray]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    bw, bh = x2 - x1, y2 - y1
    if bw < 24 or bh < 24:
        return None
    x1, x2 = int(max(0, x1 - pad * bw)), int(min(w, x2 + pad * bw))
    y1, y2 = int(max(0, y1 - pad * bh)), int(min(h, y2 + pad * bh))
    c = frame[y1:y2, x1:x2]
    if c.size == 0:
        return None
    s = side / max(c.shape[:2])
    import cv2
    return cv2.resize(c, (max(1, int(c.shape[1] * s)), max(1, int(c.shape[0] * s))), interpolation=cv2.INTER_AREA) if s < 1 else c.copy()


_CLASSIFIER = None


def get_classifier():
    """Shares the CLIP model that 'Analyze people' uses (one download). Returns None when it is not loaded yet."""
    global _CLASSIFIER
    from attributes import get_analyzer
    a = get_analyzer()
    if not a.is_ready():
        a.load_async()
        return None
    if _CLASSIFIER is None or _CLASSIFIER.a is not a:
        _CLASSIFIER = VehicleClassifier(a)
    return _CLASSIFIER
