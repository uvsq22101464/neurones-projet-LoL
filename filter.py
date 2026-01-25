import os
import json
import regex

INPUT_DIR = "out"
OUTPUT_DIR = "out_clean"
SPELL_FIELDS_TO_KEEP = ["DataValues", "mSpellCalculations", "cooldownTime", "castRange", "mCoefficient", "castRangeDisplayOverride", "mSpellTags", "mTargetingTypeData"]
STATS_TO_KEEP = ["mCharacterName", "baseHP", "hpPerLevel", "baseStaticHPRegen", "hpRegenPerLevel", 
                 "primaryAbilityResource", "baseDamage", "damagePerLevel", "baseArmor", "armorPerLevel", 
                 "baseSpellBlock", "spellBlockPerLevel", "baseMoveSpeed", "attackRange", "attackSpeed", 
                 "attackSpeedRatio", "attackSpeedPerLevel", "acquisitionRange"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

for filename in os.listdir(INPUT_DIR):
    if not filename.endswith(".bin.json"):
        continue

    champ_name = filename.split(".")[0].lower()
    input_path = os.path.join(INPUT_DIR, filename)
    output_path = os.path.join(OUTPUT_DIR, filename)

    with open(input_path, encoding="utf-8") as f:
        raw = json.load(f)

    filtered = {}
    spellsNames = []

    reg1 = rf"^characters/{champ_name}/spells/[^/]+/{champ_name}[^/]*$"
    reg2 = rf"^characters/{champ_name}/spells/{champ_name}[a-z]+/{champ_name}[a-z]$"

    # Character root
    for key, value in raw.items():
        key_l = key.lower()
        if f"characters/{champ_name}/characterrecords/root" in key_l:
            val = {}
            for k, v in value.items():
                if k == "mAbilities":
                    for spell in v:
                        if "{" in spell:
                            continue
                        spellsNames.append(f"{spell}/{spell.split('/')[3][:-7]}")
                if k in (STATS_TO_KEEP):
                    if type(v) == float:
                        val[k] = round(v, 3)
                    else:
                        val[k] = v
                if k == "characterToolData":
                    for tag, role in v.items():
                        if tag == "searchTags" or tag == "searchTagsSecondary" or tag == "roles":
                            val[tag] = role
            filtered[key] = val
            break
    
    with open(input_path, encoding="utf-8") as f:
        raw = json.load(f)

    for key, value in raw.items():
        key_l = key.lower()
        #print(key +"         "+ spellsNames[0])
        if key in spellsNames:
            for k, v in value.items():
                if k == "mSpell":
                    val = {}
                    for data, vals in v.items():
                        if data in SPELL_FIELDS_TO_KEEP:
                            val[data] = vals

                    filtered[key] = val



            
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)
    print(f"✔ {filename} → {len(filtered)} entrées conservées")
