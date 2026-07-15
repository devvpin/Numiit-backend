import re

with open(r'd:\Coin Project\app\NumiIT\backend\app\services\ml_inference.py', 'r', encoding='utf-8') as f:
    content = f.read()

new_era = '''ERA_CONTEXT: dict[str, dict] = {
    # Rudrasena variants
    "rudrasena_i": {
        "era": "Rudrasena I",
        "dynasty": "Western Kshatrapas (c. 200–222 CE)",
        "script": "Brahmi",
        "transliteration": "Rājadhirāja Rudrasena",
        "translation": "Great King Rudrasena",
        "description": "Silver dramma of Rudrasena I, featuring royal bust obverse and Brahmi legend reverse.",
        "father": "Rudrasimha I",
        "legend": "rajnomahaksatrapasarudrasihaputrasarajnomahaksatrapasarudrasenasa",
        "rules": [
            "Identified by the phrase \\"Rudrasihaputrasa\\" (son of Rudrasimha I).",
            "The king's name appears as \\"Rudrasenasa\\" with title \\"Rajno Mahaksatrapasa\\".",
            "Sequence contains unique consonant conjuncts for \\"dra\\" (𑀤𑁆𑀭) and \\"sa\\" (𑀲)."
        ]
    },
    "rudrasena_ii": {
        "era": "Rudrasena II",
        "dynasty": "Western Kshatrapas (c. 255–278 CE)",
        "script": "Brahmi",
        "transliteration": "Mahākṣatrapa Rudrasena",
        "translation": "Great Satrap Rudrasena II",
        "description": "Silver dramma of Rudrasena II with Kshatrapa titles in Brahmi script.",
        "father": "Viradaman",
        "legend": "rajnoksatrapasaviradamaputrasarajnomahaksatrapasarudrasenasa",
        "rules": [
            "Identified by the phrase \\"Viradamaputrasa\\" (son of Viradaman).",
            "The king's name appears as \\"Rudrasenasa\\" with title \\"Rajno Mahaksatrapasa\\".",
            "Matches the sequence sequence with Levenshtein similarity above 50%."
        ]
    },
    "rudrasena_iii": {
        "era": "Rudrasena III",
        "dynasty": "Western Kshatrapas (c. 348–380 CE)",
        "script": "Brahmi",
        "transliteration": "Svāmi Mahākṣatrapa Rudrasena",
        "translation": "Lord Great Satrap Rudrasena III",
        "description": "Late Western Kshatrapa coinage with elaborate Brahmi inscriptions.",
        "father": "Rudradaman II",
        "legend": "rajnomahaksatrapasasvamirudradamaputrasarajnomahaksatrapasasvamirudrasenasa",
        "rules": [
            "Identified by the phrase \\"Svamirudradamaputrasa\\" (son of Lord Rudradaman II).",
            "Uses the prestigious title \\"Svami\\" (Lord) in front of the ruler's name.",
            "Matches sequence with \\"svami\\" (𑀲𑁆𑀯𑀸𑀫𑀺) character patterns."
        ]
    },
    "rudrasena_iv": {
        "era": "Rudrasena IV",
        "dynasty": "Western Kshatrapas (c. 382–388 CE)",
        "script": "Brahmi",
        "transliteration": "Mahākṣatrapa Rudrasena",
        "translation": "Great Satrap Rudrasena IV",
        "description": "Last phase Western Kshatrapa coin before Gupta conquest.",
        "father": "Simhasena",
        "legend": "rajnomahaksatrapasasvamisihasenaputrasarajnomahaksatrapasasvamirudrasenasa",
        "rules": [
            "Identified by the phrase \\"Svamisihasenaputrasa\\" (son of Lord Simhasena).",
            "Late dynasty coinage showing slightly stylized Brahmi script characters."
        ]
    },
    # Chastana lineage
    "chastana": {
        "era": "Chastana",
        "dynasty": "Western Kshatrapas (c. 78–130 CE)",
        "script": "Brahmi & Kharoshthi",
        "transliteration": "Mahākṣatrapa Chaṣṭana",
        "translation": "Great Satrap Chastana",
        "description": "Founder of the Kardamaka dynasty; bilingual silver coinage.",
        "father": "Ysamotika",
        "legend": "rajnoksatrapasaysamotikaputrasachastanasa",
        "rules": [
            "Identified by the father \\"Ysamotika\\" (ysamotikaputrasa).",
            "Bilingual reverse/obverse legend using Brahmi and Kharoshthi/Greek scripts."
        ]
    },
    "jayadaman": {
        "era": "Jayadaman",
        "dynasty": "Western Kshatrapas (c. 170–178 CE)",
        "script": "Brahmi",
        "transliteration": "Kṣatrapa Jayadāman",
        "translation": "Satrap Jayadaman",
        "description": "Short-reign Kshatrapa whose coins are comparatively rare.",
        "father": "Chastana",
        "legend": "rajnoksatrapasasvamichastanaputrasajayadamasa",
        "rules": [
            "Identified by the father \\"Chastana\\" (chastanaputrasa).",
            "Ruler holds the title \\"Ksatrapa\\" (Satrap) instead of \\"Mahaksatrapa\\"."
        ]
    },
    "damajadasri_i": {
        "era": "Damajadasri I",
        "dynasty": "Western Kshatrapas (c. 178–197 CE)",
        "script": "Brahmi",
        "transliteration": "Mahākṣatrapa Dāmājadasri",
        "translation": "Great Satrap Damajadasri I",
        "description": "Prolific reign; standardised the dramma coinage format.",
        "father": "Rudradaman I",
        "legend": "rajnomahaksatrapasarudradamaputrasarajnomahaksatrapasadamajadasriyasa",
        "rules": [
            "Identified by the father \\"Rudradaman I\\" (rudradamaputrasa).",
            "Contains character sequence \\"damajadasriyasa\\" at the end of the legend."
        ]
    },
    "damajadasri_ii": {
        "era": "Damajadasri II",
        "dynasty": "Western Kshatrapas (c. 232–235 CE)",
        "script": "Brahmi",
        "transliteration": "Mahākṣatrapa Dāmājadasri",
        "translation": "Great Satrap Damajadasri II",
        "description": "Brief interlude ruler between Rudrasena lines.",
        "father": "Rudrasena I",
        "legend": "rajnomahaksatrapasarudrasenaputrasarajnoksatrapasadamajadasriyasa",
        "rules": [
            "Identified by the father \\"Rudrasena I\\" (rudrasenaputrasa).",
            "Ruler's title on legend is \\"Ksatrapa\\" while the father's is \\"Mahaksatrapa\\"."
        ]
    },
    "viradaman": {
        "era": "Viradaman",
        "dynasty": "Western Kshatrapas (c. 234–238 CE)",
        "script": "Brahmi",
        "transliteration": "Mahākṣatrapa Vīradāman",
        "translation": "Great Satrap Viradaman",
        "description": "Coins show degraded portrait style typical of mid-3rd century.",
        "father": "Damasena",
        "legend": "rajnomahaksatrapasadamasenaputrasarajnoksatrapasaviradamasa",
        "rules": [
            "Identified by the father \\"Damasena\\" (damasenaputrasa).",
            "Ruler's name is \\"Viradamasa\\" with title \\"Ksatrapasa\\"."
        ]
    },
    "visvavarman": {
        "era": "Visvavarman",
        "dynasty": "Western Kshatrapas (c. 238–250 CE)",
        "script": "Brahmi",
        "transliteration": "Mahākṣatrapa Viśvavarman",
        "translation": "Great Satrap Visvavarman",
        "description": "Transitional ruler; coins bridge two Rudrasena lines.",
        "father": "Visvasena",
        "legend": "rajnomahaksatrapasavisvasenaputrasarajnomahaksatrapasavisvavarmasa",
        "rules": [
            "Identified by the father \\"Visvasena\\" (visvasenaputrasa)."
        ]
    },
    "yasodaman_i": {
        "era": "Yasodaman I",
        "dynasty": "Western Kshatrapas (c. 222–225 CE)",
        "script": "Brahmi",
        "transliteration": "Mahākṣatrapa Yaśodāman",
        "translation": "Great Satrap Yasodaman I",
        "description": "Short-reign; notable for high silver purity coinage.",
        "father": "Damasena",
        "legend": "rajnomahaksatrapasadamasenaputrasarajnomahaksatrapasayasodamasa",
        "rules": [
            "Identified by the father \\"Damasena\\" (damasenaputrasa).",
            "Ruler's name is \\"Yasodamasa\\" with title \\"Mahaksatrapasa\\"."
        ]
    },
    "yasodaman_ii": {
        "era": "Yasodaman II",
        "dynasty": "Western Kshatrapas (c. 278–295 CE)",
        "script": "Brahmi",
        "transliteration": "Svāmi Yaśodāman",
        "translation": "Lord Yasodaman II",
        "description": "Coins show Brahmi legend with royal epithets.",
        "father": "Rudrasimha II",
        "legend": "rajnoksatrapasasvamirudrasihaputrasarajnoksatrapasasvamiyasodamasa",
        "rules": [
            "Identified by the father \\"Rudrasimha II\\" (svamirudrasihaputrasa).",
            "Uses title \\"Svami\\" (Lord) for both father and son."
        ]
    },
    "abhyadaman": {
        "era": "Abhyadaman",
        "dynasty": "Western Kshatrapas (c. 295–300 CE)",
        "script": "Brahmi",
        "transliteration": "Mahākṣatrapa Abhyadāman",
        "translation": "Great Satrap Abhyadaman",
        "description": "Rare coins from a transitional period.",
        "father": "Unknown",
        "legend": "rajnomahaksatrapasaabhyadamasa",
        "rules": [
            "Shorter legend sequence matching \\"abhyadamasa\\" directly."
        ]
    },
    "isvaradatta": {
        "era": "Isvaradatta",
        "dynasty": "Abhira / anti-Kshatrapa (c. 235 CE)",
        "script": "Brahmi",
        "transliteration": "Mahākṣatrapa Īśvaradatta",
        "translation": "Great Satrap Isvaradatta",
        "description": "Usurper who briefly controlled western India; very rare coins.",
        "father": "Unknown",
        "legend": "rajnomahaksatrapasaisvaradattasa",
        "rules": [
            "Unique non-dynastic legend without the standard \\"putrasa\\" (son of) formula.",
            "Explicitly names \\"Isvaradattasa\\" with regnal year on the obverse."
        ]
    },
    "nahapana": {
        "era": "Nahapana",
        "dynasty": "Western Kshatrapas (c. 105–125 CE)",
        "script": "Brahmi & Kharoshthi",
        "transliteration": "Kṣaharāta Nahapāna",
        "translation": "Kshaharata Nahapana",
        "description": "Early bilingual silver coinage; predecessor dynasty to Kardamakas.",
        "father": "Bhumaka",
        "legend": "rajnoksaharatasanahapanasa",
        "rules": [
            "Uses the distinct dynasty title \\"Ksaharatasa\\" (Kshaharata).",
            "Obverse features Greek legend; reverse features bilingual Brahmi and Kharoshthi script."
        ]
    },
    "rudradaman_i": {
        "era": "Rudradaman I",
        "dynasty": "Western Kshatrapas (c. 130–150 CE)",
        "script": "Brahmi",
        "transliteration": "Mahākṣatrapa Rudradāman",
        "translation": "Great Satrap Rudradaman I",
        "description": "Most celebrated Kshatrapa ruler; issued famous Junagadh inscription.",
        "father": "Jayadaman",
        "legend": "rajnoksatrapasajayadamaputrasarajnomahaksatrapasarudradamasa",
        "rules": [
            "Identified by the father \\"Jayadaman\\" (jayadamaputrasa).",
            "Ruler's name is \\"Rudradamasa\\" with title \\"Mahaksatrapasa\\"."
        ]
    },
    "brahmi_kshatrap": {
        "era": "Brahmi Kshatrap",
        "dynasty": "Western Kshatrapas (generic)",
        "script": "Brahmi",
        "transliteration": "Mahākṣatrapa",
        "translation": "Great Satrap",
        "description": "Western Kshatrapa silver dramma with standard Brahmi legend.",
        "father": "Various",
        "legend": "rajnomahaksatrapasa...",
        "rules": [
            "A generic Western Kshatrapa designation when titles are detected but names are missing."
        ]
    },
    "unknown": {
        "era": "Unknown / Other Era",
        "dynasty": "Unknown Dynasty",
        "script": "Unknown",
        "transliteration": "???",
        "translation": "Unknown meaning",
        "description": "The inscription did not match any known templates with high enough confidence.",
        "father": "Unknown",
        "legend": "Unknown",
        "rules": []
    },
}'''

# Fallback string replace
start_str = 'ERA_CONTEXT: dict[str, dict[str, str]] = {'
end_str = '    },\n}'

start_idx = content.find(start_str)
end_idx = content.find(end_str, start_idx) + len(end_str)

if start_idx != -1:
    new_content = content[:start_idx] + new_era + content[end_idx:]
    with open(r'd:\Coin Project\app\NumiIT\backend\app\services\ml_inference.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("Replaced ERA_CONTEXT")
else:
    print("Could not find ERA_CONTEXT block")
