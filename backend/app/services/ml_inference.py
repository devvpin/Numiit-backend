"""
ml_inference.py – NumiIT YOLO model inference service.

Loads the two trained segmentation models at startup (singleton pattern)
and exposes two public functions:
  • infer_coin(image_path)       → coin era recognition
  • infer_character(image_path)  → Brahmi character recognition

Both return structured dicts compatible with the API schemas.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any
import cv2

try:
    import Levenshtein
except ImportError:
    Levenshtein = None

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Absolute paths to the two trained model weights
# ─────────────────────────────────────────────────────────────────────────────

# Try to find the models inside the backend folder first (new location), then fall back to the repo root (old location)
_BACKEND_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[3]

if (_BACKEND_ROOT / "runs_coins_final").exists():
    _MODEL_ROOT = _BACKEND_ROOT
else:
    _MODEL_ROOT = _REPO_ROOT

COIN_MODEL_PATH = (
    _MODEL_ROOT
    / "runs_coins_final"
    / "segment"
    / "brahmi_coins_identification"
    / "initial_19_coins"
    / "weights"
    / "best.pt"
)

CHAR_MODEL_PATH = (
    _MODEL_ROOT
    / "runs_characters"
    / "segment"
    / "brahmi_character_handwritten_identification"
    / "handwritten_character"
    / "weights"
    / "best.pt"
)

COIN_CLASSIFIER_PATH = (
    _MODEL_ROOT
    / "coins_classification"
    / "best_model.pth"
)

# Method 1 Two-Sided (Paired) Model Directory (stored in coins_classification)
if (_BACKEND_ROOT / "coins_classification" / "final_summary.json").exists():
    PAIRED_COIN_DIR = _BACKEND_ROOT / "coins_classification"
elif (_MODEL_ROOT / "coins_classification" / "final_summary.json").exists():
    PAIRED_COIN_DIR = _MODEL_ROOT / "coins_classification"
elif (_MODEL_ROOT / "coins_classification" / "joint_siamese_224.pth").exists():
    PAIRED_COIN_DIR = _MODEL_ROOT / "coins_classification"
elif (_BACKEND_ROOT.parent / "final").exists():
    PAIRED_COIN_DIR = _BACKEND_ROOT.parent / "final"
else:
    PAIRED_COIN_DIR = _REPO_ROOT / "final"


# ─────────────────────────────────────────────────────────────────────────────
# Dynasty / era context lookup for coin classes
#
# Keys are YOLO class names (as stored in the model's .names dict).
# Add / rename entries here when you retrain with new classes.
# ─────────────────────────────────────────────────────────────────────────────

ERA_CONTEXT: dict[str, dict] = {
    # ── Table 1 Chronological Order of Western Kshatrapa Kings & Fathers ──
    "chastana": {
        "era": "Chastana",
        "dynasty": "Western Kshatrapas (c. 78–130 CE)",
        "script": "Brahmi & Kharoshthi",
        "transliteration": "Rājño Kṣatrapasa Chastaṇa",
        "translation": "Satrap Chastana",
        "description": "Founder of the Kardamaka dynasty; silver dramma with bilingual Brahmi and Kharoshthi legends.",
        "father": "Zamotika",
        "legend": "rajnoksatrapasazamotikaputrasachastanasa",
        "rules": [
            "Identified by father Zamotika (zamotikaputrasa).",
            "Bilingual legend using Brahmi on reverse and Greek/Kharoshthi elements."
        ]
    },
    "damazada_i": {
        "era": "Damazada I",
        "dynasty": "Western Kshatrapas (c. 130–140 CE)",
        "script": "Brahmi",
        "transliteration": "Rājño Kṣatrapasa Dāmazada",
        "translation": "Satrap Damazada I",
        "description": "Early Kardamaka ruler; brother or contemporary of Chastana lineage.",
        "father": "Zamotika",
        "legend": "rajnoksatrapasazamotikaputrasadamazadasa",
        "rules": [
            "Identified by father Zamotika (zamotikaputrasa).",
            "Holds the title Ksatrapa (rajnoksatrapasa)."
        ]
    },
    "jayadaman": {
        "era": "Jayadaman",
        "dynasty": "Western Kshatrapas (c. 130–140 CE)",
        "script": "Brahmi",
        "transliteration": "Rājño Kṣatrapasa Jayadāman",
        "translation": "Satrap Jayadaman",
        "description": "Satrap whose father is recorded as Unknown in Table 1; coins are comparatively rare.",
        "father": "Unknown",
        "legend": "rajnoksatrapasajayadamanasa",
        "rules": [
            "Father listed as Unknown in royal table sequence.",
            "Ruler holds the title Ksatrapa (Satrap) instead of Mahaksatrapa."
        ]
    },
    "rudradaman_i": {
        "era": "Rudradaman I",
        "dynasty": "Western Kshatrapas (c. 130–150 CE)",
        "script": "Brahmi",
        "transliteration": "Rājño Mahākṣatrapasa Rudradāman",
        "translation": "Great Satrap Rudradaman I",
        "description": "Celebrated ruler who defeated Satavahanas and authored the historic Junagadh rock inscription.",
        "father": "Jayadaman",
        "legend": "rajnoksatrapasajayadamaputrasarajnomahaksatrapasarudradamasa",
        "rules": [
            "Identified by father Jayadaman (jayadamaputrasa).",
            "Ruler's name is Rudradamasa with title Mahaksatrapasa."
        ]
    },
    "rudradaman": {
        "era": "Rudradaman I",
        "dynasty": "Western Kshatrapas (c. 130–150 CE)",
        "script": "Brahmi",
        "transliteration": "Rājño Mahākṣatrapasa Rudradāman",
        "translation": "Great Satrap Rudradaman I",
        "description": "Celebrated ruler who defeated Satavahanas and authored the historic Junagadh rock inscription.",
        "father": "Jayadaman",
        "legend": "rajnoksatrapasajayadamaputrasarajnomahaksatrapasarudradamasa",
        "rules": [
            "Identified by father Jayadaman (jayadamaputrasa)."
        ]
    },
    "damazada_ii": {
        "era": "Damazada II",
        "dynasty": "Western Kshatrapas (c. 178–197 CE)",
        "script": "Brahmi",
        "transliteration": "Rājño Mahākṣatrapasa Dāmazada",
        "translation": "Great Satrap Damazada II",
        "description": "Son of Rudradaman I, ruling as Ksatrapa and Mahaksatrapa.",
        "father": "Rudradaman",
        "legend": "rajnomahaksatrapasarudradamaputrasarajnoksatrapasadamazadasa",
        "rules": [
            "Identified by father Rudradaman (rudradamaputrasa).",
            "Coins show both Ksatrapa and Mahaksatrapa titles."
        ]
    },
    "damajadasri_i": {
        "era": "Damajadasri I (Damazada II)",
        "dynasty": "Western Kshatrapas (c. 178–197 CE)",
        "script": "Brahmi",
        "transliteration": "Rājño Mahākṣatrapasa Dāmajadaśrī",
        "translation": "Great Satrap Damajadasri I",
        "description": "Standardised the silver dramma coinage format with Saka era dates.",
        "father": "Rudradaman",
        "legend": "rajnomahaksatrapasarudradamaputrasarajnomahaksatrapasadamajadasriyasa",
        "rules": [
            "Identified by father Rudradaman (rudradamaputrasa)."
        ]
    },
    "rudrasimha_i": {
        "era": "Rudrasimha I",
        "dynasty": "Western Kshatrapas (c. 178–197 CE)",
        "script": "Brahmi",
        "transliteration": "Rājño Mahākṣatrapasa Rudrasiṁha",
        "translation": "Great Satrap Rudrasimha I",
        "description": "Prolific coinage establishing dated Saka era numerals on obverse behind the royal bust.",
        "father": "Rudradaman",
        "legend": "rajnomahaksatrapasarudradamaputrasarajnomahaksatrapasarudrasihasa",
        "rules": [
            "Identified by father Rudradaman (rudradamaputrasa).",
            "Name appears as Rudrasihasa with title Mahaksatrapasa."
        ]
    },
    "jivadaman": {
        "era": "Jivadaman",
        "dynasty": "Western Kshatrapas (c. 178–181, 197–198 CE)",
        "script": "Brahmi",
        "transliteration": "Rājño Mahākṣatrapasa Jīvadāman",
        "translation": "Great Satrap Jivadaman",
        "description": "Pioneered dated Saka coinage; held two distinct reigns separated by Rudrasimha I.",
        "father": "Damazada II",
        "legend": "rajnomahaksatrapasadamazadasaputrasarajnomahaksatrapasajivadamasa",
        "rules": [
            "Identified by father Damazada II (damazadaputrasa).",
            "Title Mahaksatrapasa with name Jivadamasa."
        ]
    },
    "satyadaman": {
        "era": "Satyadaman",
        "dynasty": "Western Kshatrapas (c. 198–199 CE)",
        "script": "Brahmi",
        "transliteration": "Rājño Kṣatrapasa Satyadāman",
        "translation": "Satrap Satyadaman",
        "description": "Brief reign as Ksatrapa; coins are among the rarest in Western Kshatrapa numismatics.",
        "father": "Damazada II",
        "legend": "rajnomahaksatrapasadamazadasaputrasarajnoksatrapasasatyadamasa",
        "rules": [
            "Identified by father Damazada II (damazadaputrasa).",
            "Holds the title Ksatrapa with name Satyadamasa."
        ]
    },
    "rudrasena_i": {
        "era": "Rudrasena I",
        "dynasty": "Western Kshatrapas (c. 200–222 CE)",
        "script": "Brahmi",
        "transliteration": "Rājño Mahākṣatrapasa Rudrasena",
        "translation": "Great Satrap Rudrasena I",
        "description": "Silver dramma of Rudrasena I, featuring royal bust obverse and Brahmi legend reverse.",
        "father": "Rudrasimha I",
        "legend": "rajnomahaksatrapasarudrasihaputrasarajnomahaksatrapasarudrasenasa",
        "rules": [
            "Identified by the phrase Rudrasihaputrasa (son of Rudrasimha I).",
            "The king's name appears as Rudrasenasa with title Rajno Mahaksatrapasa."
        ]
    },
    "isvaradatta": {
        "era": "Isvaradatta / Isvaradeva",
        "dynasty": "Western Kshatrapas (c. 220–222 CE)",
        "script": "Brahmi",
        "transliteration": "Rājño Kṣatrapasa Īśvaradeva",
        "translation": "Satrap Isvaradeva / Isvaradatta",
        "description": "Documented in royal tables as son of Rudrasimha I; dated regnal issues.",
        "father": "Rudrasimha I",
        "legend": "rajnomahaksatrapasarudrasihaputrasarajnoksatrapasaisvaradevassa",
        "rules": [
            "Table 1 lists father as Rudrasimha I.",
            "Coin legend features distinctive Isvaradevassa / Isvaradattasa."
        ]
    },
    "isvaradeva": {
        "era": "Isvaradeva",
        "dynasty": "Western Kshatrapas (c. 220–222 CE)",
        "script": "Brahmi",
        "transliteration": "Rājño Kṣatrapasa Īśvaradeva",
        "translation": "Satrap Isvaradeva",
        "description": "Recorded in chronological Table 1 as son of Rudrasimha I.",
        "father": "Rudrasimha I",
        "legend": "rajnomahaksatrapasarudrasihaputrasarajnoksatrapasaisvaradevassa",
        "rules": [
            "Table 1 lists father as Rudrasimha I."
        ]
    },
    "prithvisena": {
        "era": "Prithvisena",
        "dynasty": "Western Kshatrapas (c. 222 CE)",
        "script": "Brahmi",
        "transliteration": "Rājño Kṣatrapasa Pṛthvīsena",
        "translation": "Satrap Prithvisena",
        "description": "Short reign as Ksatrapa before Damasena assumed the Mahaksatrapa title.",
        "father": "Rudrasena I",
        "legend": "rajnomahaksatrapasarudrasenaputrasarajnoksatrapasaprithvisenasa",
        "rules": [
            "Identified by father Rudrasena I (rudrasenaputrasa).",
            "Name Prithvisenasa with title Ksatrapasa."
        ]
    },
    "damasena": {
        "era": "Damasena",
        "dynasty": "Western Kshatrapas (c. 222–236 CE)",
        "script": "Brahmi",
        "transliteration": "Rājño Mahākṣatrapasa Dāmasena",
        "translation": "Great Satrap Damasena",
        "description": "Patriarch of the mid-dynasty; four of his sons subsequently succeeded him as rulers.",
        "father": "Rudrasimha I",
        "legend": "rajnomahaksatrapasarudrasihaputrasarajnomahaksatrapasadamasenasa",
        "rules": [
            "Identified by father Rudrasimha I (rudrasihaputrasa).",
            "Name Damasenasa with title Mahaksatrapasa."
        ]
    },
    "samghadaman": {
        "era": "Samghadaman",
        "dynasty": "Western Kshatrapas (c. 222–223 CE)",
        "script": "Brahmi",
        "transliteration": "Rājño Mahākṣatrapasa Saṁghadāman",
        "translation": "Great Satrap Samghadaman",
        "description": "Brief reign as Mahaksatrapa during succession transitions following Rudrasena I.",
        "father": "Rudrasimha I",
        "legend": "rajnomahaksatrapasarudrasihaputrasarajnomahaksatrapasasamghadamasa",
        "rules": [
            "Identified by father Rudrasimha I (rudrasihaputrasa).",
            "Name Samghadamasa with title Mahaksatrapasa."
        ]
    },
    "damajadasri_ii": {
        "era": "Damajadasri II",
        "dynasty": "Western Kshatrapas (c. 232–235 CE)",
        "script": "Brahmi",
        "transliteration": "Rājño Kṣatrapasa Dāmajadaśrī",
        "translation": "Satrap Damajadasri II",
        "description": "Son of Rudrasena I who ruled as Ksatrapa while uncle Damasena held Mahaksatrapa rank.",
        "father": "Rudrasena I",
        "legend": "rajnomahaksatrapasarudrasenaputrasarajnoksatrapasadamajadasriyasa",
        "rules": [
            "Identified by father Rudrasena I (rudrasenaputrasa).",
            "Title Ksatrapasa with name Damajadasriyasa."
        ]
    },
    "viradaman": {
        "era": "Viradaman",
        "dynasty": "Western Kshatrapas (c. 234–238 CE)",
        "script": "Brahmi",
        "transliteration": "Rājño Kṣatrapasa Vīradāman",
        "translation": "Satrap Viradaman",
        "description": "Son of Damasena; ruled as Ksatrapa and fathered future Mahaksatrapa Rudrasena II.",
        "father": "Damasena",
        "legend": "rajnomahaksatrapasadamasenaputrasarajnoksatrapasaviradamasa",
        "rules": [
            "Identified by father Damasena (damasenaputrasa).",
            "Name Viradamasa with title Ksatrapasa."
        ]
    },
    "yasodaman_i": {
        "era": "Yasodaman I",
        "dynasty": "Western Kshatrapas (c. 238–239 CE)",
        "script": "Brahmi",
        "transliteration": "Rājño Mahākṣatrapasa Yaśodāman",
        "translation": "Great Satrap Yasodaman I",
        "description": "Son of Damasena; briefly advanced from Ksatrapa to Mahaksatrapa in Saka 160–161.",
        "father": "Damasena",
        "legend": "rajnomahaksatrapasadamasenaputrasarajnomahaksatrapasayasodamasa",
        "rules": [
            "Identified by father Damasena (damasenaputrasa).",
            "Name Yasodamasa with title Mahaksatrapasa."
        ]
    },
    "vijayasena": {
        "era": "Vijayasena",
        "dynasty": "Western Kshatrapas (c. 238–250 CE)",
        "script": "Brahmi",
        "transliteration": "Rājño Mahākṣatrapasa Vijayasena",
        "translation": "Great Satrap Vijayasena",
        "description": "Son of Damasena; celebrated 12-year reign marked by plentiful high-grade silver drammas.",
        "father": "Damasena",
        "legend": "rajnomahaksatrapasadamasenaputrasarajnomahaksatrapasavijayasenasa",
        "rules": [
            "Identified by father Damasena (damasenaputrasa).",
            "Name Vijayasenasa with title Mahaksatrapasa."
        ]
    },
    "damajadasri_iii": {
        "era": "Damajadasri III",
        "dynasty": "Western Kshatrapas (c. 250–255 CE)",
        "script": "Brahmi",
        "transliteration": "Rājño Mahākṣatrapasa Dāmajadaśrī",
        "translation": "Great Satrap Damajadasri III",
        "description": "Fourth and final son of Damasena to rule as Mahaksatrapa.",
        "father": "Damasena",
        "legend": "rajnomahaksatrapasadamasenaputrasarajnomahaksatrapasadamajadasriyasa",
        "rules": [
            "Identified by father Damasena (damasenaputrasa).",
            "Title Mahaksatrapasa."
        ]
    },
    "rudrasena_ii": {
        "era": "Rudrasena II",
        "dynasty": "Western Kshatrapas (c. 255–278 CE)",
        "script": "Brahmi",
        "transliteration": "Rājño Mahākṣatrapasa Rudrasena",
        "translation": "Great Satrap Rudrasena II",
        "description": "Son of Viradaman; long and prosperous reign, father of three subsequent rulers.",
        "father": "Viradaman",
        "legend": "rajnoksatrapasaviradamaputrasarajnomahaksatrapasarudrasenasa",
        "rules": [
            "Identified by father Viradaman (viradamaputrasa).",
            "Name Rudrasenasa with title Mahaksatrapasa."
        ]
    },
    "visvasimha": {
        "era": "Visvasimha",
        "dynasty": "Western Kshatrapas (c. 278–282 CE)",
        "script": "Brahmi",
        "transliteration": "Rājño Kṣatrapasa Viśvasiṁha",
        "translation": "Satrap Visvasimha",
        "description": "Eldest son of Rudrasena II; ruled as Ksatrapa and briefly as Mahaksatrapa.",
        "father": "Rudrasena II",
        "legend": "rajnomahaksatrapasarudrasenaputrasarajnoksatrapasavisvasihasa",
        "rules": [
            "Identified by father Rudrasena II (rudrasenaputrasa).",
            "Name Visvasihasa with title Ksatrapasa."
        ]
    },
    "bhartrdaman": {
        "era": "Bhartrdaman",
        "dynasty": "Western Kshatrapas (c. 278–295 CE)",
        "script": "Brahmi",
        "transliteration": "Rājño Mahākṣatrapasa Bhartṛdāman",
        "translation": "Great Satrap Bhartrdaman",
        "description": "Second son of Rudrasena II; prolonged stable reign with prolific silver drammas.",
        "father": "Rudrasena II",
        "legend": "rajnomahaksatrapasarudrasenaputrasarajnomahaksatrapasabhartrdamasa",
        "rules": [
            "Identified by father Rudrasena II (rudrasenaputrasa).",
            "Name Bhartrdamasa with titles Ksatrapasa and Mahaksatrapasa."
        ]
    },
    "visvasena": {
        "era": "Visvasena",
        "dynasty": "Western Kshatrapas (c. 293–304 CE)",
        "script": "Brahmi",
        "transliteration": "Rājño Kṣatrapasa Viśvasena",
        "translation": "Satrap Visvasena",
        "description": "Third son of Rudrasena II; ruled strictly as Ksatrapa until Saka 226.",
        "father": "Rudrasena II",
        "legend": "rajnomahaksatrapasarudrasenaputrasarajnoksatrapasavisvasenasa",
        "rules": [
            "Identified by father Rudrasena II (rudrasenaputrasa).",
            "Name Visvasenasa with title Ksatrapasa."
        ]
    },
    "rudrasimha_ii": {
        "era": "Rudrasimha II",
        "dynasty": "Western Kshatrapas (c. 305–317 CE)",
        "script": "Brahmi",
        "transliteration": "Rājño Kṣatrapasa Svāmi Rudrasiṁha",
        "translation": "Satrap Rudrasimha II",
        "description": "Began the collateral line; father held title Swami Jivadaman without royal titles.",
        "father": "Jivadaman",
        "legend": "svamijivadamaputrasarajnoksatrapasarudrasihasa",
        "rules": [
            "Identified by father Swami Jivadaman (svamijivadamaputrasa).",
            "Coins introduce the honorific Svami (Lord) prefix."
        ]
    },
    "yasodaman_ii": {
        "era": "Yasodaman II",
        "dynasty": "Western Kshatrapas (c. 317–332 CE)",
        "script": "Brahmi",
        "transliteration": "Rājño Kṣatrapasa Svāmi Yaśodāman",
        "translation": "Satrap Yasodaman II",
        "description": "Son of Rudrasimha II; silver drammas show both father and son holding Svami honorifics.",
        "father": "Rudrasimha II",
        "legend": "rajnoksatrapasasvamirudrasihaputrasarajnoksatrapasasvamiyasodamasa",
        "rules": [
            "Identified by father Rudrasimha II (svamirudrasihaputrasa).",
            "Name Yasodamasa with title Ksatrapasa."
        ]
    },
    "rudrasena_iii": {
        "era": "Rudrasena III",
        "dynasty": "Western Kshatrapas (c. 348–380 CE)",
        "script": "Brahmi",
        "transliteration": "Rājño Mahākṣatrapasa Svāmi Rudrasena",
        "translation": "Lord Great Satrap Rudrasena III",
        "description": "Restored Mahaksatrapa authority with abundant silver drammas and lead square issues.",
        "father": "Swami Rudradaman",
        "legend": "rajnomahaksatrapasasvamirudradamaputrasarajnomahaksatrapasasvamirudrasenasa",
        "rules": [
            "Identified by father Swami Rudradaman (svamirudradamaputrasa).",
            "Uses the prestigious title Svami (Lord) on both ruler and father."
        ]
    },
    "simhasena": {
        "era": "Swami Simhasena",
        "dynasty": "Western Kshatrapas (c. 382 CE)",
        "script": "Brahmi",
        "transliteration": "Rājño Mahākṣatrapasa Svāmi Siṁhasena",
        "translation": "Lord Great Satrap Simhasena",
        "description": "Succeeded Swami Rudrasena III; coins record his lineage as sister's son / son.",
        "father": "Swami Rudrasena",
        "legend": "rajnomahaksatrapasasvamirudrasenaputrasarajnomahaksatrapasasvamisihasenasa",
        "rules": [
            "Identified by father Swami Rudrasena (svamirudrasenaputrasa).",
            "Title Svami Mahaksatrapa."
        ]
    },
    "rudrasena_iv": {
        "era": "Rudrasena IV",
        "dynasty": "Western Kshatrapas (c. 382–388 CE)",
        "script": "Brahmi",
        "transliteration": "Rājño Mahākṣatrapasa Svāmi Rudrasena",
        "translation": "Lord Great Satrap Rudrasena IV",
        "description": "Son of Swami Simhasena; late dynasty coinage with stylized Brahmi characters.",
        "father": "Swami Simhasena",
        "legend": "rajnomahaksatrapasasvamisihasenaputrasarajnomahaksatrapasasvamirudrasenasa",
        "rules": [
            "Identified by father Swami Simhasena (svamisihasenaputrasa).",
            "Late dynasty coinage showing slightly stylized Brahmi script characters."
        ]
    },
    "rudrasimha_iii": {
        "era": "Rudrasimha III",
        "dynasty": "Western Kshatrapas (c. 388–395 CE)",
        "script": "Brahmi",
        "transliteration": "Rājño Mahākṣatrapasa Svāmi Rudrasiṁha",
        "translation": "Lord Great Satrap Rudrasimha III",
        "description": "Final recorded Mahaksatrapa ruler before Chandragupta II's Gupta conquest of Malwa/Gujarat.",
        "father": "Swami Satyasimha",
        "legend": "rajnomahaksatrapasasvamisatyasihaputrasarajnomahaksatrapasasvamirudrasihasa",
        "rules": [
            "Identified by father Swami Satyasimha (svamisatyasihaputrasa).",
            "Final Kshatrapa coinage before absorption into the Gupta empire."
        ]
    },
    "horumurnata": {
        "era": "Swami Horumurnata",
        "dynasty": "Western Kshatrapas / Regional (c. 4th century CE)",
        "script": "Brahmi",
        "transliteration": "Rājño Kṣatrapasa Svāmi Horumūrṇata",
        "translation": "Lord Satrap Horumurnata",
        "description": "Regional Kshatrapa ruler in Saurashtra/Sindh issuing rare potin and lead coins.",
        "father": "Unknown",
        "legend": "svamihorumurnatasa",
        "rules": [
            "Father Unknown.",
            "Distinctive non-Sanskritic name Horumurnata with Svami title."
        ]
    },
    # ── Additional Reference & Legacy Classes ──
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
            "Uses the distinct dynasty title Ksaharatasa (Kshaharata).",
            "Obverse features Greek legend; reverse features bilingual Brahmi and Kharoshthi script."
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
            "Identified by father Visvasena (visvasenaputrasa)."
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
            "Shorter legend sequence matching abhyadamasa directly."
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
}

# ─────────────────────────────────────────────────────────────────────────────
# Brahmi character transliteration table
# ─────────────────────────────────────────────────────────────────────────────

BRAHMI_CHAR_MAP: dict[str, dict[str, str]] = {
    "da": {"unicode": "𑀤", "iast": "da", "meaning": "Consonant 'da'", "fontChar": "A"},
    "dra": {"unicode": "𑀤𑁆𑀭", "iast": "dra", "meaning": "Conjunct 'dra'", "fontChar": "B"},
    "ha": {"unicode": "𑀳", "iast": "ha", "meaning": "Consonant 'ha'", "fontChar": "C"},
    "jnah": {"unicode": "𑀚𑁆𑀜𑀂", "iast": "jnah", "meaning": "Conjunct 'jnah'", "fontChar": "D"},
    "jno": {"unicode": "𑀚𑁆𑀜𑁄", "iast": "jno", "meaning": "Conjunct 'jno'", "fontChar": "E"},
    "ksha": {"unicode": "𑀓𑁆𑀱", "iast": "ksa", "meaning": "Conjunct 'kṣa'", "fontChar": "F"},
    "ma": {"unicode": "𑀫", "iast": "ma", "meaning": "Consonant 'ma'", "fontChar": "G"},
    "na": {"unicode": "𑀦", "iast": "na", "meaning": "Consonant 'na'", "fontChar": "H"},
    "pa": {"unicode": "𑀧", "iast": "pa", "meaning": "Consonant 'pa'", "fontChar": "I"},
    "pu": {"unicode": "𑀧𑀼", "iast": "pu", "meaning": "Consonant 'pu'", "fontChar": "J"},
    "ra": {"unicode": "𑀭", "iast": "ra", "meaning": "Consonant 'ra'", "fontChar": "K"},
    "ru": {"unicode": "𑀭𑀼", "iast": "ru", "meaning": "Consonant 'ru'", "fontChar": "L"},
    "sa": {"unicode": "𑀲", "iast": "sa", "meaning": "Consonant 'sa'", "fontChar": "M"},
    "se": {"unicode": "𑀲𑁂", "iast": "se", "meaning": "Consonant 'se'", "fontChar": "M"},
    "tra": {"unicode": "𑀢𑁆𑀭", "iast": "tra", "meaning": "Conjunct 'tra'", "fontChar": "N"},
    "vi": {"unicode": "𑀯𑀺", "iast": "vi", "meaning": "Consonant 'vi'", "fontChar": "O"},
}


# ─────────────────────────────────────────────────────────────────────────────
# Known Coin Legends for Template Matching
# ─────────────────────────────────────────────────────────────────────────────

KNOWN_LEGENDS: dict[str, dict[str, str]] = {
    # ── Rulers and Fathers in Chronological Order (Table 1) ──
    "chastana": {
        "legend": "rajnoksatrapasazamotikaputrasachastanasa",
        "father": "Zamotika",
        "king": "Chastana"
    },
    "damazada_i": {
        "legend": "rajnoksatrapasazamotikaputrasadamazadasa",
        "father": "Zamotika",
        "king": "Damazada I"
    },
    "jayadaman": {
        "legend": "rajnoksatrapasajayadamanasa",
        "father": "Unknown",
        "king": "Jayadaman"
    },
    "rudradaman_i": {
        "legend": "rajnoksatrapasajayadamaputrasarajnomahaksatrapasarudradamasa",
        "father": "Jayadaman",
        "king": "Rudradaman I"
    },
    "rudradaman": {
        "legend": "rajnoksatrapasajayadamaputrasarajnomahaksatrapasarudradamasa",
        "father": "Jayadaman",
        "king": "Rudradaman I"
    },
    "damazada_ii": {
        "legend": "rajnomahaksatrapasarudradamaputrasarajnoksatrapasadamazadasa",
        "father": "Rudradaman",
        "king": "Damazada II"
    },
    "damajadasri_i": {
        "legend": "rajnomahaksatrapasarudradamaputrasarajnomahaksatrapasadamajadasriyasa",
        "father": "Rudradaman",
        "king": "Damajadasri I"
    },
    "rudrasimha_i": {
        "legend": "rajnomahaksatrapasarudradamaputrasarajnomahaksatrapasarudrasihasa",
        "father": "Rudradaman",
        "king": "Rudrasimha I"
    },
    "jivadaman": {
        "legend": "rajnomahaksatrapasadamazadasaputrasarajnomahaksatrapasajivadamasa",
        "father": "Damazada II",
        "king": "Jivadaman"
    },
    "satyadaman": {
        "legend": "rajnomahaksatrapasadamazadasaputrasarajnoksatrapasasatyadamasa",
        "father": "Damazada II",
        "king": "Satyadaman"
    },
    "rudrasena_i": {
        "legend": "rajnomahaksatrapasarudrasihaputrasarajnomahaksatrapasarudrasenasa",
        "father": "Rudrasimha I",
        "king": "Rudrasena I"
    },
    "isvaradatta": {
        "legend": "rajnomahaksatrapasarudrasihaputrasarajnoksatrapasaisvaradevassa",
        "father": "Rudrasimha I",
        "king": "Isvaradatta / Isvaradeva"
    },
    "isvaradeva": {
        "legend": "rajnomahaksatrapasarudrasihaputrasarajnoksatrapasaisvaradevassa",
        "father": "Rudrasimha I",
        "king": "Isvaradeva"
    },
    "prithvisena": {
        "legend": "rajnomahaksatrapasarudrasenaputrasarajnoksatrapasaprithvisenasa",
        "father": "Rudrasena I",
        "king": "Prithvisena"
    },
    "damasena": {
        "legend": "rajnomahaksatrapasarudrasihaputrasarajnomahaksatrapasadamasenasa",
        "father": "Rudrasimha I",
        "king": "Damasena"
    },
    "samghadaman": {
        "legend": "rajnomahaksatrapasarudrasihaputrasarajnomahaksatrapasasamghadamasa",
        "father": "Rudrasimha I",
        "king": "Samghadaman"
    },
    "damajadasri_ii": {
        "legend": "rajnomahaksatrapasarudrasenaputrasarajnoksatrapasadamajadasriyasa",
        "father": "Rudrasena I",
        "king": "Damajadasri II"
    },
    "viradaman": {
        "legend": "rajnomahaksatrapasadamasenaputrasarajnoksatrapasaviradamasa",
        "father": "Damasena",
        "king": "Viradaman"
    },
    "yasodaman_i": {
        "legend": "rajnomahaksatrapasadamasenaputrasarajnomahaksatrapasayasodamasa",
        "father": "Damasena",
        "king": "Yasodaman I"
    },
    "vijayasena": {
        "legend": "rajnomahaksatrapasadamasenaputrasarajnomahaksatrapasavijayasenasa",
        "father": "Damasena",
        "king": "Vijayasena"
    },
    "damajadasri_iii": {
        "legend": "rajnomahaksatrapasadamasenaputrasarajnomahaksatrapasadamajadasriyasa",
        "father": "Damasena",
        "king": "Damajadasri III"
    },
    "rudrasena_ii": {
        "legend": "rajnoksatrapasaviradamaputrasarajnomahaksatrapasarudrasenasa",
        "father": "Viradaman",
        "king": "Rudrasena II"
    },
    "visvasimha": {
        "legend": "rajnomahaksatrapasarudrasenaputrasarajnoksatrapasavisvasihasa",
        "father": "Rudrasena II",
        "king": "Visvasimha"
    },
    "bhartrdaman": {
        "legend": "rajnomahaksatrapasarudrasenaputrasarajnomahaksatrapasabhartrdamasa",
        "father": "Rudrasena II",
        "king": "Bhartrdaman"
    },
    "visvasena": {
        "legend": "rajnomahaksatrapasarudrasenaputrasarajnoksatrapasavisvasenasa",
        "father": "Rudrasena II",
        "king": "Visvasena"
    },
    "rudrasimha_ii": {
        "legend": "svamijivadamaputrasarajnoksatrapasarudrasihasa",
        "father": "Jivadaman",
        "king": "Rudrasimha II"
    },
    "yasodaman_ii": {
        "legend": "rajnoksatrapasasvamirudrasihaputrasarajnoksatrapasasvamiyasodamasa",
        "father": "Rudrasimha II",
        "king": "Yasodaman II"
    },
    "rudrasena_iii": {
        "legend": "rajnomahaksatrapasasvamirudradamaputrasarajnomahaksatrapasasvamirudrasenasa",
        "father": "Swami Rudradaman",
        "king": "Rudrasena III"
    },
    "simhasena": {
        "legend": "rajnomahaksatrapasasvamirudrasenaputrasarajnomahaksatrapasasvamisihasenasa",
        "father": "Swami Rudrasena",
        "king": "Swami Simhasena"
    },
    "rudrasena_iv": {
        "legend": "rajnomahaksatrapasasvamisihasenaputrasarajnomahaksatrapasasvamirudrasenasa",
        "father": "Swami Simhasena",
        "king": "Rudrasena IV"
    },
    "rudrasimha_iii": {
        "legend": "rajnomahaksatrapasasvamisatyasihaputrasarajnomahaksatrapasasvamirudrasihasa",
        "father": "Swami Satyasimha",
        "king": "Rudrasimha III"
    },
    "horumurnata": {
        "legend": "svamihorumurnatasa",
        "father": "Unknown",
        "king": "Swami Horumurnata"
    },
    # ── Additional Reference Classes ──
    "nahapana": {
        "legend": "rajnoksaharatasanahapanasa",
        "father": "Bhumaka",
        "king": "Nahapana"
    },
    "visvavarman": {
        "legend": "rajnomahaksatrapasavisvasenaputrasarajnomahaksatrapasavisvavarmasa",
        "father": "Visvasena",
        "king": "Visvavarman"
    },
    "abhyadaman": {
        "legend": "rajnomahaksatrapasaabhyadamasa",
        "father": "Unknown",
        "king": "Abhyadaman"
    }
}


# ─────────────────────────────────────────────────────────────────────────────
# Singleton model holder
# ─────────────────────────────────────────────────────────────────────────────

class _ModelHolder:
    """Lazy-loaded singleton holding both YOLO models."""

    def __init__(self) -> None:
        self._coin_model: Any | None = None
        self._char_model: Any | None = None
        self._classifier_model: Any | None = None
        self._paired_models: Any | None = None
        self._coin_names: dict[int, str] = {}
        self._char_names: dict[int, str] = {}
        self._classifier_classes: list[str] = []
        self._paired_classes: list[str] = []
        self._paired_cache: Any | None = None
        self._paired_device: Any | None = None

    def _load_coin_model(self) -> Any:
        if self._coin_model is None:
            try:
                from ultralytics import YOLO  # type: ignore[import-untyped]
                logger.info("Loading coin recognition model from %s", COIN_MODEL_PATH)
                self._coin_model = YOLO(str(COIN_MODEL_PATH))
                self._coin_names = self._coin_model.names  # {0: 'class_name', ...}
                logger.info("Coin model loaded. Classes: %s", self._coin_names)
            except Exception as exc:
                logger.error("Failed to load coin model: %s", exc)
                raise
        return self._coin_model

    def _load_char_model(self) -> Any:
        if self._char_model is None:
            try:
                from ultralytics import YOLO  # type: ignore[import-untyped]
                logger.info("Loading character recognition model from %s", CHAR_MODEL_PATH)
                self._char_model = YOLO(str(CHAR_MODEL_PATH))
                self._char_names = self._char_model.names
                logger.info("Character model loaded. Classes: %s", self._char_names)
            except Exception as exc:
                logger.error("Failed to load character model: %s", exc)
                raise
        return self._char_model

    def _load_classifier_model(self) -> Any:
        if self._classifier_model is None:
            try:
                import torch
                from app.services.coin_classifier import ImprovCoinNet
                logger.info("Loading ArcFace coin classifier from %s", COIN_CLASSIFIER_PATH)
                if not COIN_CLASSIFIER_PATH.exists():
                    raise FileNotFoundError(f"Classification model weights not found at: {COIN_CLASSIFIER_PATH}")

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                checkpoint = torch.load(COIN_CLASSIFIER_PATH, map_location=device, weights_only=False)
                self._classifier_classes = checkpoint.get("class_names", [])
                arch_cfg = checkpoint.get("arch_config", {})

                model = ImprovCoinNet(
                    num_classes=len(self._classifier_classes),
                    backbone=arch_cfg.get("backbone", "resnet18"),
                    embedding_dim=arch_cfg.get("embedding_dim", 512),
                    s=arch_cfg.get("scale", 30.0),
                    m=arch_cfg.get("margin", 0.50),
                    pretrained=False,
                ).to(device)

                model.load_state_dict(checkpoint["model_state_dict"])
                model.eval()
                self._classifier_model = model
                self._classifier_device = device
                logger.info("Coin classifier loaded. Total classes: %d (Device: %s)", len(self._classifier_classes), device)
            except Exception as exc:
                logger.error("Failed to load classifier model: %s", exc)
                raise
        return self._classifier_model

    def _load_paired_models(self) -> Any:
        if self._paired_models is None:
            try:
                import json
                import torch
                from app.services.coin_classifier_paired import load_paired_models
                logger.info("Loading Method 1 Paired models from %s", PAIRED_COIN_DIR)

                summary_file = PAIRED_COIN_DIR / "final_summary.json"
                if not summary_file.exists():
                    summary_file = PAIRED_COIN_DIR / "results" / "final_summary.json"
                if not summary_file.exists():
                    raise FileNotFoundError(f"final_summary.json not found in {PAIRED_COIN_DIR}")

                with open(summary_file, "r", encoding="utf-8") as f:
                    summary = json.load(f)
                self._paired_classes = [x["king"] for x in summary.get("per_king_stats", [])]

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                if (PAIRED_COIN_DIR / "joint_siamese_224.pth").exists() or (PAIRED_COIN_DIR / "joint_siamese_224.pth.part1").exists() or (PAIRED_COIN_DIR / "legend_model.pth").exists():
                    ckpt_dir = PAIRED_COIN_DIR
                else:
                    ckpt_dir = PAIRED_COIN_DIR / "checkpoints"
                self._paired_models = load_paired_models(ckpt_dir, len(self._paired_classes), device)

                cache_file = PAIRED_COIN_DIR / "cached_embeddings.pt"
                if not cache_file.exists():
                    cache_file = PAIRED_COIN_DIR / "results" / "cached_embeddings.pt"

                if cache_file.exists():
                    loaded_cache = torch.load(cache_file, map_location="cpu", weights_only=False)
                    if "train_embs_fused" in loaded_cache and isinstance(loaded_cache["train_embs_fused"], torch.Tensor):
                        loaded_cache["train_embs_fused"] = loaded_cache["train_embs_fused"].to(device)
                    self._paired_cache = loaded_cache
                else:
                    self._paired_cache = None

                self._paired_device = device
                logger.info("Method 1 Paired models loaded successfully. Total classes: %d (Device: %s)", len(self._paired_classes), device)
            except Exception as exc:
                logger.error("Failed to load Method 1 paired models: %s", exc)
                raise
        return self._paired_models

    @property
    def coin_model(self) -> Any:
        return self._load_coin_model()

    @property
    def char_model(self) -> Any:
        return self._load_char_model()

    @property
    def classifier_model(self) -> Any:
        return self._load_classifier_model()

    @property
    def classifier_classes(self) -> list[str]:
        self._load_classifier_model()
        return self._classifier_classes

    @property
    def classifier_device(self) -> Any:
        self._load_classifier_model()
        return self._classifier_device

    @property
    def paired_models(self) -> Any:
        return self._load_paired_models()

    @property
    def paired_classes(self) -> list[str]:
        self._load_paired_models()
        return self._paired_classes

    @property
    def paired_cache(self) -> Any:
        self._load_paired_models()
        return self._paired_cache

    @property
    def paired_device(self) -> Any:
        self._load_paired_models()
        return self._paired_device

    @property
    def coin_names(self) -> dict[int, str]:
        self._load_coin_model()
        return self._coin_names

    @property
    def char_names(self) -> dict[int, str]:
        self._load_char_model()
        return self._char_names


_holder = _ModelHolder()


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def _levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate the Levenshtein distance between two strings."""
    if Levenshtein is not None:
        return Levenshtein.distance(s1, s2)
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def _get_circular_order(characters: list[dict], regions: list[dict]) -> tuple[list[dict], list[dict]]:
    """Sort characters and regions clockwise around their collective centroid."""
    if not characters:
        return characters, regions
        
    # Calculate centroid
    cx = sum(c["boundingBox"]["left"] + c["boundingBox"]["width"]/2 for c in characters) / len(characters)
    cy = sum(c["boundingBox"]["top"] + c["boundingBox"]["height"]/2 for c in characters) / len(characters)
    
    def get_angle(bbox: dict) -> float:
        bx = bbox["left"] + bbox["width"]/2
        by = bbox["top"] + bbox["height"]/2
        # Use atan2(y, x). In image space, y increases downwards.
        # We want clockwise starting from 12 o'clock (top, which is -y).
        # atan2(y, x) starts from 3 o'clock and goes clockwise (positive y) or counterclockwise (negative y).
        # We'll just use atan2(y - cy, x - cx) and sort by it.
        return math.atan2(by - cy, bx - cx)
        
    # Zip, sort, and unzip
    combined = list(zip(characters, regions))
    combined.sort(key=lambda x: get_angle(x[0]["boundingBox"]))
    
    sorted_chars = [c for c, r in combined]
    sorted_regions = [r for c, r in combined]
    
    return sorted_chars, sorted_regions


def _align_and_match_sequence(characters: list[dict], regions: list[dict]) -> tuple[list[dict], list[dict], str | None, dict | None]:
    """Find the best circular shift that matches a known legend template."""
    if not characters:
        return characters, regions, None, None
        
    # The standard Kshatrapa coin legend is mostly titles (rajnoksatrapasa...putrasa...).
    # If the model only detects title characters, it's impossible to uniquely identify the king.
    # The following characters only appear in the specific names of the kings/fathers in our dataset:
    UNIQUE_NAME_CHARS = {"da", "dra", "na", "ru", "se", "vi"}
    
    detected_iast_set = {c.get("iast", "").lower().replace(" ", "") for c in characters}
    if not detected_iast_set.intersection(UNIQUE_NAME_CHARS):
        # No unique king/father characters detected; cannot confidently identify the king.
        return characters, regions, None, None
        
    n = len(characters)
    best_score = -1.0
    best_shift = 0
    best_legend_keys = []
    
    # Generate all candidate strings (transliterations)
    for shift in range(n):
        shifted_chars = characters[shift:] + characters[:shift]
        candidate_str = "".join(c.get("iast", "").lower().replace(" ", "") for c in shifted_chars)
        window_size = len(candidate_str)
        
        if window_size == 0:
            continue
            
        # Compare against all known legends using a sliding window to eliminate length bias
        for key, info in KNOWN_LEGENDS.items():
            legend = info["legend"]
            legend_full = legend + legend # for circular wrap-around
            
            if window_size >= len(legend):
                dist = _levenshtein_distance(candidate_str, legend)
            else:
                dist = float('inf')
                for i in range(len(legend)):
                    sub_legend = legend_full[i:i+window_size]
                    d = _levenshtein_distance(candidate_str, sub_legend)
                    if d < dist:
                        dist = d
            
            # Accuracy score (0.0 to 1.0). 1.0 means perfect substring match.
            score = 1.0 - (dist / window_size)
            
            if score > best_score:
                best_score = score
                best_shift = shift
                best_legend_keys = [key]
            elif score == best_score and key not in best_legend_keys:
                best_legend_keys.append(key)
                
    # If the score is too low, we cannot identify the king.
    if best_score < 0.5:
        return characters, regions, None, None

    ambiguous_base_name = None
    if len(best_legend_keys) > 1:
        base_names = set()
        for key in best_legend_keys:
            king_name = KNOWN_LEGENDS[key]["king"]
            # Remove " I", " II", " III", " IV" to get base name
            base = king_name.split(" I")[0].strip()
            base_names.add(base)
            
        if len(base_names) == 1:
            ambiguous_base_name = base_names.pop()
        else:
            # Ambiguous across DIFFERENT kings, truly unknown
            return characters, regions, None, None
                
    # Realign lists based on best shift
    aligned_chars = characters[best_shift:] + characters[:best_shift]
    aligned_regions = regions[best_shift:] + regions[:best_shift]
    
    best_key = best_legend_keys[0]
    match_info = KNOWN_LEGENDS[best_key].copy()
    
    if ambiguous_base_name:
        match_info["king"] = ambiguous_base_name
        match_info["father"] = "Unknown"
        
    return aligned_chars, aligned_regions, best_key, match_info

def _normalize_class_name(raw: str) -> str:
    """Lowercase, strip spaces, replace spaces/dashes with underscores."""
    return raw.strip().lower().replace(" ", "_").replace("-", "_")


def _get_era_context(class_name: str) -> dict[str, str]:
    """Return era context dict; falls back to unknown if key not found."""
    key = _normalize_class_name(class_name)
    return ERA_CONTEXT.get(key, ERA_CONTEXT["unknown"])


def _build_era_scores_from_predictions(
    raw_class_confs: dict[int, float],
    all_class_names: dict[int, str],
) -> list[dict]:
    """
    Build the full era_scores list including classes with 0% confidence.

    raw_class_confs: {class_id: max_confidence_seen}  (only detected classes)
    all_class_names: {class_id: class_name}  (all model classes)

    Returns list sorted by confidence descending.
    """
    scores: list[dict] = []
    for class_id, class_name in all_class_names.items():
        conf = raw_class_confs.get(class_id, 0.0)
        ctx = _get_era_context(class_name)
        scores.append({
            "era": ctx["era"],
            "class_name": class_name,
            "confidence": round(conf, 4),
            "is_primary": False,  # set after sorting
        })

    # Sort descending by confidence
    scores.sort(key=lambda x: x["confidence"], reverse=True)

    # Mark highest-confidence as primary (only if > 0)
    if scores and scores[0]["confidence"] > 0:
        scores[0]["is_primary"] = True

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def infer_coin(image_path: str) -> dict:
    """
    Run coin-era recognition on the given image.

    Returns:
    {
      "scan_mode": "coin",
      "era_scores": [{"era": str, "confidence": float, "is_primary": bool}, ...],
      "primary_era": str,
      "primary_confidence": float,
      "dynasty_context": str,
      "script": str,
      "transliteration": str,
      "translation": str,
      "regions": [DetectedRegion-compatible dict, ...]
    }
    """
    try:
        model = _holder.coin_model
        all_names = _holder.coin_names

        results = model.predict(
            source=image_path,
            conf=0.40,   # very low threshold — we want ALL classes for negative display
            iou=0.42,
            verbose=False,
            save=False,
        )

        # Overwrite the original image with YOLO's plotted image
        for result in results:
            if result.boxes is not None:
                im_bgr = result.plot()
                cv2.imwrite(image_path, im_bgr)

        characters: list[dict] = []
        regions: list[dict] = []
        region_idx = 0

        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                cls_name = all_names.get(cls_id, f"char_{cls_id}")
                norm_name = _normalize_class_name(cls_name)
                
                # Drop 'unknown' or dash bounding boxes instead of returning them
                if "unknown" in norm_name or norm_name == "_" or cls_name in ("-", "—"):
                    continue

                char_info = BRAHMI_CHAR_MAP.get(norm_name, {})

                xyxyn = boxes.xyxyn[i].tolist()
                x1, y1, x2, y2 = xyxyn
                bbox = {
                    "left": round(x1, 4),
                    "top": round(y1, 4),
                    "width": round(x2 - x1, 4),
                    "height": round(y2 - y1, 4),
                }

                unicode_char = char_info.get("unicode", cls_name)
                iast = char_info.get("iast", cls_name)
                meaning = char_info.get("meaning", f"Brahmi character '{cls_name}'")
                font_char = char_info.get("fontChar", "")

                characters.append({
                    "char": cls_name,
                    "unicode": unicode_char,
                    "iast": iast,
                    "confidence": round(conf, 4),
                    "boundingBox": bbox,
                })

                regions.append({
                    "regionIndex": region_idx,
                    "boundingBox": bbox,
                    "scriptName": "Brahmi",
                    "originalText": unicode_char,
                    "fontChar": font_char,
                    "transliteration": iast,
                    "translation": meaning,
                    "dynastyContext": "Brahmi Script – Handwritten / Distorted Character Recognition",
                    "confidence": round(conf, 4),
                    "glyphCount": 1,
                })
                region_idx += 1

        # Sort characters radially around the centroid
        characters, regions = _get_circular_order(characters, regions)
        
        # Align the starting point and match against known legends
        characters, regions, best_legend_key, match_info = _align_and_match_sequence(characters, regions)

        # Re-index regions after sort
        for i, r in enumerate(regions):
            r["regionIndex"] = i

        # Now determine the primary era based on the OCR template match
        primary_class = "unknown"
        if best_legend_key:
            primary_class = best_legend_key
        elif characters:
            # Characters were detected, but no king could be identified (e.g. only titles found)
            primary_class = "brahmi_kshatrap"

        ctx = _get_era_context(primary_class)
        
        # Update the regions with the historically accurate dynasty context
        for r in regions:
            r["dynastyContext"] = f"{ctx.get('dynasty', 'Unknown Dynasty')} - {ctx.get('description', '')}".strip(" -")

        # Build era_scores based on the matched template (Negative Calling)
        era_scores = []
        avg_char_conf = sum(c["confidence"] for c in characters) / len(characters) if characters else 0.0
        candidate_str = "".join(c.get("iast", "").lower().replace(" ", "") for c in characters) if characters else ""

        # For each class in ERA_CONTEXT, calculate its confidence
        for class_key, class_ctx in ERA_CONTEXT.items():
            if class_key == "unknown":
                continue
            
            is_primary = (class_key == primary_class)
            
            if not characters:
                conf = 0.0
            elif class_key == "brahmi_kshatrap":
                if primary_class != "brahmi_kshatrap" and primary_class != "unknown":
                    # A specific king was matched, so generic kshatrap is low (0-10%)
                    conf = 0.05
                else:
                    conf = avg_char_conf
            elif class_key in KNOWN_LEGENDS:
                # Calculate similarity score to this specific legend
                legend = KNOWN_LEGENDS[class_key]["legend"]
                legend_full = legend + legend
                window_size = len(candidate_str)
                
                if window_size >= len(legend):
                    dist = _levenshtein_distance(candidate_str, legend)
                else:
                    dist = float('inf')
                    for idx in range(len(legend)):
                        sub_legend = legend_full[idx:idx+window_size]
                        d = _levenshtein_distance(candidate_str, sub_legend)
                        if d < dist:
                            dist = d
                sim_score = max(0.0, 1.0 - (dist / window_size))
                
                # Combine similarity score with average character detection confidence
                conf = sim_score * avg_char_conf
                
                # If this is the identified primary class, ensure a solid confidence
                if is_primary:
                    conf = max(conf, avg_char_conf)
            else:
                conf = 0.0

            # Round to 4 decimal places
            conf = round(conf, 4)
            
            # Format display era name correctly
            display_era = class_ctx["era"]
            if class_key == primary_class and match_info:
                display_era = match_info["king"]

            era_scores.append({
                "era": display_era,
                "class_name": class_key,
                "confidence": conf,
                "is_primary": is_primary,
                "dynasty": class_ctx.get("dynasty", ""),
                "transliteration": class_ctx.get("transliteration", ""),
                "translation": class_ctx.get("translation", ""),
                "father": class_ctx.get("father", ""),
                "legend": class_ctx.get("legend", ""),
                "rules": class_ctx.get("rules", []),
            })

        # Sort the era_scores descending by confidence
        era_scores.sort(key=lambda x: x["confidence"], reverse=True)

        # If primary class was unknown, or if no characters were detected,
        # ensure there is an 'unknown' score at the very top (or as primary)
        if primary_class == "unknown":
            max_sim_score = max([x["confidence"] for x in era_scores]) if era_scores else 0.0
            unknown_ctx = ERA_CONTEXT["unknown"]
            era_scores.insert(0, {
                "era": unknown_ctx["era"],
                "class_name": "unknown",
                "confidence": 1.0 if not characters else round(max(0.0, 1.0 - max_sim_score), 4),
                "is_primary": True,
                "dynasty": unknown_ctx.get("dynasty", ""),
                "transliteration": unknown_ctx.get("transliteration", ""),
                "translation": unknown_ctx.get("translation", ""),
                "father": unknown_ctx.get("father", ""),
                "legend": unknown_ctx.get("legend", ""),
                "rules": unknown_ctx.get("rules", []),
            })
        else:
            unknown_ctx = ERA_CONTEXT["unknown"]
            era_scores.append({
                "era": unknown_ctx["era"],
                "class_name": "unknown",
                "confidence": 0.0,
                "is_primary": False,
                "dynasty": unknown_ctx.get("dynasty", ""),
                "transliteration": unknown_ctx.get("transliteration", ""),
                "translation": unknown_ctx.get("translation", ""),
                "father": unknown_ctx.get("father", ""),
                "legend": unknown_ctx.get("legend", ""),
                "rules": unknown_ctx.get("rules", []),
            })

        if not regions:
            # No characters detected, so it's unknown
            regions = [
                {
                    "regionIndex": 0,
                    "boundingBox": {"left": 0.05, "top": 0.05, "width": 0.9, "height": 0.9},
                    "scriptName": ctx["script"],
                    "originalText": "",
                    "transliteration": "—",
                    "translation": "No Brahmi characters detected",
                    "dynastyContext": f"{ctx['era']} – {ctx['dynasty']} | {ctx['description']}",
                    "confidence": 0.0,
                    "glyphCount": 0,
                }
            ]

        primary_conf = 0.0
        for score_item in era_scores:
            if score_item["is_primary"]:
                primary_conf = score_item["confidence"]
                break

        return {
            "scan_mode": "coin",
            "era_scores": era_scores,
            "primary_era": ctx["era"],
            "primary_confidence": primary_conf,
            "dynasty_context": ctx["dynasty"],
            "script": ctx["script"],
            "transliteration": ctx["transliteration"],
            "translation": ctx["translation"],
            "regions": regions,
        }

    except Exception as exc:
        logger.exception("Coin inference failed: %s", exc)
        # Return graceful fallback so the API doesn't 500
        return {
            "scan_mode": "coin",
            "era_scores": [
                {"era": "Unknown / Other Era", "class_name": "unknown",
                 "confidence": 0.0, "is_primary": True}
            ],
            "primary_era": "Unknown / Other Era",
            "primary_confidence": 0.0,
            "dynasty_context": "Unidentified",
            "script": "Unknown",
            "transliteration": "—",
            "translation": "Model inference failed",
            "regions": [
                {
                    "regionIndex": 0,
                    "boundingBox": {"left": 0.05, "top": 0.05, "width": 0.9, "height": 0.9},
                    "scriptName": "Unknown",
                    "originalText": "",
                    "transliteration": "—",
                    "translation": "Model inference failed",
                    "dynastyContext": str(exc),
                    "confidence": 0.0,
                    "glyphCount": 0,
                }
            ],
        }


def infer_character(image_path: str) -> dict:
    """
    Run Brahmi character recognition on the given image.

    Returns:
    {
      "scan_mode": "character",
      "characters": [{"char": str, "unicode": str, "iast": str, "confidence": float,
                       "boundingBox": {...}}, ...],
      "regions": [DetectedRegion-compatible dict, ...]
    }
    """
    try:
        model = _holder.char_model
        all_names = _holder.char_names

        results = model.predict(
            source=image_path,
            conf=0.40,
            iou=0.42,
            verbose=False,
            save=False,
        )

        # Overwrite the original image with YOLO's plotted image
        for result in results:
            if result.boxes is not None:
                im_bgr = result.plot()
                cv2.imwrite(image_path, im_bgr)

        characters: list[dict] = []
        regions: list[dict] = []
        region_idx = 0

        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                cls_name = all_names.get(cls_id, f"char_{cls_id}")
                norm_name = _normalize_class_name(cls_name)
                
                # Drop 'unknown' or dash bounding boxes instead of returning them
                if "unknown" in norm_name or norm_name == "_" or cls_name in ("-", "—"):
                    continue

                char_info = BRAHMI_CHAR_MAP.get(norm_name, {})

                xyxyn = boxes.xyxyn[i].tolist()
                x1, y1, x2, y2 = xyxyn
                bbox = {
                    "left": round(x1, 4),
                    "top": round(y1, 4),
                    "width": round(x2 - x1, 4),
                    "height": round(y2 - y1, 4),
                }

                unicode_char = char_info.get("unicode", cls_name)
                iast = char_info.get("iast", cls_name)
                meaning = char_info.get("meaning", f"Brahmi character '{cls_name}'")
                font_char = char_info.get("fontChar", "")

                characters.append({
                    "char": cls_name,
                    "unicode": unicode_char,
                    "iast": iast,
                    "confidence": round(conf, 4),
                    "boundingBox": bbox,
                })

                regions.append({
                    "regionIndex": region_idx,
                    "boundingBox": bbox,
                    "scriptName": "Brahmi",
                    "originalText": unicode_char,
                    "fontChar": font_char,
                    "transliteration": iast,
                    "translation": meaning,
                    "dynastyContext": "Brahmi Script – Handwritten / Distorted Character Recognition",
                    "confidence": round(conf, 4),
                    "glyphCount": 1,
                })
                region_idx += 1

        # Sort characters radially around the centroid
        characters, regions = _get_circular_order(characters, regions)
        
        # Align the starting point and match against known legends
        characters, regions, best_legend_key, match_info = _align_and_match_sequence(characters, regions)

        # Re-index regions after sort
        for i, r in enumerate(regions):
            r["regionIndex"] = i

        predicted_king = "Unknown"
        predicted_father = "Unknown"
        predicted_legend = ""
        
        if match_info:
            predicted_king = match_info.get("king", "Unknown")
            predicted_father = match_info.get("father", "Unknown")
            predicted_legend = match_info.get("legend", "")

        if not regions:
            # No characters detected
            regions = [
                {
                    "regionIndex": 0,
                    "boundingBox": {"left": 0.05, "top": 0.05, "width": 0.9, "height": 0.9},
                    "scriptName": "Brahmi",
                    "originalText": "",
                    "transliteration": "—",
                    "translation": "No Brahmi characters detected",
                    "dynastyContext": "Brahmi Script – Handwritten / Distorted Character Recognition",
                    "confidence": 0.0,
                    "glyphCount": 0,
                }
            ]

        return {
            "scan_mode": "character",
            "characters": characters,
            "regions": regions,
            "predicted_king_from_text": predicted_king,
            "predicted_father_from_text": predicted_father,
            "predicted_legend": predicted_legend,
        }

    except Exception as exc:
        logger.exception("Character inference failed: %s", exc)
        return {
            "scan_mode": "character",
            "characters": [],
            "regions": [
                {
                    "regionIndex": 0,
                    "boundingBox": {"left": 0.05, "top": 0.05, "width": 0.9, "height": 0.9},
                    "scriptName": "Brahmi",
                    "originalText": "",
                    "transliteration": "—",
                    "translation": "Character model inference failed",
                    "dynastyContext": str(exc),
                    "confidence": 0.0,
                    "glyphCount": 0,
                }
            ],
            "predicted_king_from_text": "Unknown",
            "predicted_father_from_text": "Unknown",
            "predicted_legend": ""
        }


def infer_coin_classification(image_path: str, use_tta: bool = True) -> dict:
    """
    Run ArcFace deep metric learning classification on a Western Kshatrapa coin.
    Identifies the specific ruler among 29 kings using ResNet-18 + ArcFace
    with 8-angle Test-Time Augmentation (TTA).

    Returns:
    {
      "scan_mode": "classification",
      "predicted_king": str,
      "confidence": float,
      "top_predictions": [{"king": str, "confidence": float, "rank": int}, ...],
      "dynasty_context": str,
      "script": str,
      "transliteration": str,
      "translation": str,
      "father": str
    }
    """
    try:
        import torch
        import torch.nn.functional as F
        import numpy as np
        from app.services.coin_classifier import enhance_coin_image

        model = _holder.classifier_model
        class_names = _holder.classifier_classes
        device = _holder.classifier_device

        bgr = cv2.imread(image_path)
        if bgr is None:
            raise ValueError(f"Failed to load image from: {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # 1. High-fidelity numismatic preprocessing (LAB CLAHE + unsharp masking)
        enhanced = enhance_coin_image(rgb, target_size=224)

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        if use_tta:
            # 8-angle rotational TTA: (0, 45, 90, 135, 180, 225, 270, 315)
            # Averages L2-normalized embeddings to cancel out local shadows and flan orientation bias
            angles = [0, 45, 90, 135, 180, 225, 270, 315]
            h, w = enhanced.shape[:2]
            center = (w // 2, h // 2)
            views = []
            for ang in angles:
                if ang == 0:
                    rot = enhanced
                else:
                    M = cv2.getRotationMatrix2D(center, ang, 1.0)
                    rot = cv2.warpAffine(enhanced, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
                norm_v = (rot.astype(np.float32) / 255.0 - mean) / std
                views.append(torch.tensor(norm_v, dtype=torch.float32).permute(2, 0, 1))

            batch_tensor = torch.stack(views).to(device)
            with torch.no_grad():
                features = model.extract_features(batch_tensor)
                avg_feat = F.normalize(features.mean(dim=0, keepdim=True), p=2, dim=1)
                logits = model.arcface_head(avg_feat, label=None)
                probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            method_name = "arcface_resnet18_8angle_tta"
        else:
            norm_img = (enhanced.astype(np.float32) / 255.0 - mean) / std
            tensor = torch.tensor(norm_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(tensor, label=None)
                probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            method_name = "arcface_resnet18_single_view"

        sorted_indices = np.argsort(probs)[::-1]
        best_idx = sorted_indices[0]
        predicted_king = class_names[best_idx]
        best_conf = float(probs[best_idx])

        top5_preds = [
            {
                "king": class_names[idx],
                "confidence": round(float(probs[idx]), 4),
                "rank": r + 1,
            }
            for r, idx in enumerate(sorted_indices[:5])
        ]

        ctx = _get_era_context(predicted_king)

        return {
            "scan_mode": "classification",
            "predicted_king": predicted_king,
            "confidence": round(best_conf, 4),
            "top_predictions": top5_preds,
            "dynasty_context": ctx.get("dynasty", "Western Kshatrapas"),
            "script": ctx.get("script", "Brahmi"),
            "transliteration": ctx.get("transliteration", "—"),
            "translation": ctx.get("translation", "—"),
            "father": ctx.get("father", "Unknown"),
        }

    except Exception as exc:
        logger.exception("Coin classification inference failed: %s", exc)
        return {
            "scan_mode": "classification",
            "predicted_king": "Unknown",
            "confidence": 0.0,
            "top_predictions": [],
            "dynasty_context": "Unidentified",
            "script": "Brahmi",
            "transliteration": "—",
            "translation": "Coin classification inference failed",
            "father": "Unknown",
            "error": str(exc),
        }


def infer_paired_coins(obverse_path: str, reverse_path: str, use_genealogy: bool = True) -> dict:
    """
    Run Method 1 Two-Sided Classification on a Western Kshatrapa coin.
    Takes both Obverse (Face/Portrait) and Reverse (Legend/Inscription) images.
    Uses Multi-Scale Concentric Ring TTA + Joint Siamese 224+288 + Soft k-NN +
    Genealogical Disambiguation Engine.
    """
    try:
        from app.services.coin_classifier_paired import predict_paired_coins

        models = _holder.paired_models
        class_names = _holder.paired_classes
        train_cache = _holder.paired_cache
        device = _holder.paired_device

        res = predict_paired_coins(
            obverse_img_or_path=obverse_path,
            reverse_img_or_path=reverse_path,
            models_tuple=models,
            class_names=class_names,
            train_cache=train_cache,
            device=device,
            use_genealogy=use_genealogy,
        )

        predicted_king = res["predicted_king"]
        ctx = _get_era_context(predicted_king)

        return {
            "scan_mode": "paired_classification",
            "predicted_king": predicted_king,
            "confidence": res["confidence"],
            "top_predictions": res["top_predictions"],
            "disambiguation_applied": res["disambiguation_applied"],
            "disambiguation_note": res["disambiguation_note"],
            "dynasty_context": ctx.get("dynasty", "Western Kshatrapas"),
            "script": ctx.get("script", "Brahmi"),
            "transliteration": ctx.get("transliteration", "—"),
            "translation": ctx.get("translation", "—"),
            "father": ctx.get("father", "Unknown"),
            "description": ctx.get("description", ""),
            "rules": ctx.get("rules", []),
        }

    except Exception as exc:
        logger.exception("Paired coin inference failed: %s", exc)
        return {
            "scan_mode": "paired_classification",
            "predicted_king": "Unknown",
            "confidence": 0.0,
            "top_predictions": [],
            "disambiguation_applied": False,
            "disambiguation_note": "Inference error",
            "dynasty_context": "Unidentified",
            "script": "Brahmi",
            "transliteration": "—",
            "translation": "Paired coin inference failed",
            "father": "Unknown",
            "error": str(exc),
        }



