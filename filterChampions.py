import os
import json

FIELDS_TO_KEEP = ["name", "stats", "roles", "abilities"]
SPELL_FIELDS_TO_KEEP = ["DataValues", "mSpellCalculations", "cooldownTime", "castRange", "mCoefficient", "castRangeDisplayOverride", "mSpellTags", "mTargetingTypeData"]
STATS_TO_KEEP = ["health", "healthRegen", "armor", "magicResistance", "attackDamage", "attackSpeed"]


os.makedirs("champions_clean", exist_ok=True)

for filename in os.listdir("champions"):
    input_path = os.path.join("champions", filename)
    output_path = os.path.join("champions_clean", filename)

    with open(input_path, encoding="utf-8") as f:
        raw = json.load(f)

    filtered = {}
    for field, data in raw.items():
        if field == "name":
            filtered["name"] = data
        elif field == "stats":
            for stat, values in data.items():
                if stat in STATS_TO_KEEP:
                    filtered[stat] = values["flat"]
                    filtered[stat + "PerLevel"] = values["perLevel"]
                if stat == "attackRange":
                    filtered["attackRange"] = values["flat"]
        elif field == "roles":
            filtered["roles"] = data
        elif field == "abilities":
            for spell, listOfEffect in data.items():
                effects = []
                for elem in listOfEffect:
                    stats = {}
                    for key, value in elem.items():
                        if key == "name":
                            stats[key] = value
                        elif key == "effects":
                            listEffect = []
                            for e in value:
                                for k, v in e.items():
                                    if k == "leveling" and len(v) != 0:
                                        listEffect.append(v)
                            stats[key] = listEffect
                        #elif key == "cooldown":
                        #    stats[key] = value
                        elif key == "affects":
                            stats[key] = value
                    effects.append(stats)
                filtered[spell] = effects

    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)
