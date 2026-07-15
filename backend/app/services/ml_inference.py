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

# ─────────────────────────────────────────────────────────────────────────────
# Dynasty / era context lookup for coin classes
#
# Keys are YOLO class names (as stored in the model's .names dict).
# Add / rename entries here when you retrain with new classes.
# ─────────────────────────────────────────────────────────────────────────────

ERA_CONTEXT: dict[str, dict] = {
    # Rudrasena variants
    "rudrasena_i": {
        "era": "Rudrasena I",
        "dynasty": "Western Kshatrapas (c. 200–222 CE)",
        "script": "Brahmi",
        "transliteration": "Rājño Mahākṣatrapasa Rudrasena",
        "translation": "Great King Rudrasena",
        "description": "Silver dramma of Rudrasena I, featuring royal bust obverse and Brahmi legend reverse.",
        "father": "Rudrasimha I",
        "legend": "rajnomahaksatrapasarudrasihaputrasarajnomahaksatrapasarudrasenasa",
        "rules": [
            "Identified by the phrase \"Rudrasihaputrasa\" (son of Rudrasimha I).",
            "The king's name appears as \"Rudrasenasa\" with title \"Rajno Mahaksatrapasa\".",
            "Sequence contains unique consonant conjuncts for \"dra\" (𑀤𑁆𑀭) and \"sa\" (𑀲)."
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
            "Identified by the phrase \"Viradamaputrasa\" (son of Viradaman).",
            "The king's name appears as \"Rudrasenasa\" with title \"Rajno Mahaksatrapasa\".",
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
            "Identified by the phrase \"Svamirudradamaputrasa\" (son of Lord Rudradaman II).",
            "Uses the prestigious title \"Svami\" (Lord) in front of the ruler's name.",
            "Matches sequence with \"svami\" (𑀲𑁆𑀯𑀸𑀫𑀺) character patterns."
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
            "Identified by the phrase \"Svamisihasenaputrasa\" (son of Lord Simhasena).",
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
            "Identified by the father \"Ysamotika\" (ysamotikaputrasa).",
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
            "Identified by the father \"Chastana\" (chastanaputrasa).",
            "Ruler holds the title \"Ksatrapa\" (Satrap) instead of \"Mahaksatrapa\"."
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
            "Identified by the father \"Rudradaman I\" (rudradamaputrasa).",
            "Contains character sequence \"damajadasriyasa\" at the end of the legend."
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
            "Identified by the father \"Rudrasena I\" (rudrasenaputrasa).",
            "Ruler's title on legend is \"Ksatrapa\" while the father's is \"Mahaksatrapa\"."
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
            "Identified by the father \"Damasena\" (damasenaputrasa).",
            "Ruler's name is \"Viradamasa\" with title \"Ksatrapasa\"."
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
            "Identified by the father \"Visvasena\" (visvasenaputrasa)."
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
            "Identified by the father \"Damasena\" (damasenaputrasa).",
            "Ruler's name is \"Yasodamasa\" with title \"Mahaksatrapasa\"."
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
            "Identified by the father \"Rudrasimha II\" (svamirudrasihaputrasa).",
            "Uses title \"Svami\" (Lord) for both father and son."
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
            "Shorter legend sequence matching \"abhyadamasa\" directly."
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
            "Unique non-dynastic legend without the standard \"putrasa\" (son of) formula.",
            "Explicitly names \"Isvaradattasa\" with regnal year on the obverse."
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
            "Uses the distinct dynasty title \"Ksaharatasa\" (Kshaharata).",
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
            "Identified by the father \"Jayadaman\" (jayadamaputrasa).",
            "Ruler's name is \"Rudradamasa\" with title \"Mahaksatrapasa\"."
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
    "rudrasena_i": {
        "legend": "rajnomahaksatrapasarudrasihaputrasarajnomahaksatrapasarudrasenasa",
        "father": "Rudrasimha I",
        "king": "Rudrasena I"
    },
    "rudrasena_ii": {
        "legend": "rajnoksatrapasaviradamaputrasarajnomahaksatrapasarudrasenasa",
        "father": "Viradaman",
        "king": "Rudrasena II"
    },
    "rudrasena_iii": {
        "legend": "rajnomahaksatrapasasvamirudradamaputrasarajnomahaksatrapasasvamirudrasenasa",
        "father": "Rudradaman II",
        "king": "Rudrasena III"
    },
    "rudrasena_iv": {
        "legend": "rajnomahaksatrapasasvamisihasenaputrasarajnomahaksatrapasasvamirudrasenasa",
        "father": "Simhasena",
        "king": "Rudrasena IV"
    },
    "rudradaman_i": {
        "legend": "rajnoksatrapasajayadamaputrasarajnomahaksatrapasarudradamasa",
        "father": "Jayadaman",
        "king": "Rudradaman I"
    },
    "damajadasri_i": {
        "legend": "rajnomahaksatrapasarudradamaputrasarajnomahaksatrapasadamajadasriyasa",
        "father": "Rudradaman I",
        "king": "Damajadasri I"
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
    "yasodaman_ii": {
        "legend": "rajnoksatrapasasvamirudrasihaputrasarajnoksatrapasasvamiyasodamasa",
        "father": "Rudrasimha II",
        "king": "Yasodaman II"
    },
    "jayadaman": {
        "legend": "rajnoksatrapasasvamichastanaputrasajayadamasa",
        "father": "Chastana",
        "king": "Jayadaman"
    },
    "chastana": {
        "legend": "rajnoksatrapasaysamotikaputrasachastanasa",
        "father": "Ysamotika",
        "king": "Chastana"
    },
    "nahapana": {
        "legend": "rajnoksaharatasanahapanasa",
        "father": "Bhumaka",
        "king": "Nahapana"
    },
    "isvaradatta": {
        "legend": "rajnomahaksatrapasaisvaradattasa",
        "father": "Unknown",
        "king": "Isvaradatta"
    },
    "visvavarman": {
        "legend": "rajnomahaksatrapasavisvasenaputrasarajnomahaksatrapasavisvavarmasa", # Approximation for Visvavarman
        "father": "Visvasena",
        "king": "Visvavarman"
    },
    "abhyadaman": {
        "legend": "rajnomahaksatrapasaabhyadamasa", # Approximation
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
        self._coin_names: dict[int, str] = {}
        self._char_names: dict[int, str] = {}

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

    @property
    def coin_model(self) -> Any:
        return self._load_coin_model()

    @property
    def char_model(self) -> Any:
        return self._load_char_model()

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
