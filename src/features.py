"""
Feature metadata for the psychological scales used in the project.

This module defines:
- the item names for each questionnaire (DASS, TIPI, VCL),
- short labels for each item.

It does not perform any computation or data cleaning.
"""

from __future__ import annotations

from typing import Dict, List

# DASS-42 items and labels

DASS_ITEMS: List[str] = [f"Q{i}A" for i in range(1, 43)]

DASS_LABELS: Dict[str, str] = {
    "Q1A": "Upset by trivial things",
    "Q2A": "Dryness of mouth",
    "Q3A": "No positive feelings",
    "Q4A": "Breathing difficulty",
    "Q5A": "Lack of motivation",
    "Q6A": "Over-reacting",
    "Q7A": "Shakiness",
    "Q8A": "Difficulty relaxing",
    "Q9A": "Anxious in situations",
    "Q10A": "Nothing to look forward to",
    "Q11A": "Easily upset",
    "Q12A": "Using nervous energy",
    "Q13A": "Sad and depressed",
    "Q14A": "Impatient when delayed",
    "Q15A": "Faintness",
    "Q16A": "Lost interest in everything",
    "Q17A": "Low self-worth",
    "Q18A": "Touchy",
    "Q19A": "Unexplained sweating",
    "Q20A": "Scared without reason",
    "Q21A": "Life not worthwhile",
    "Q22A": "Hard to wind down",
    "Q23A": "Difficulty swallowing",
    "Q24A": "No enjoyment",
    "Q25A": "Heart awareness",
    "Q26A": "Down-hearted",
    "Q27A": "Very irritable",
    "Q28A": "Close to panic",
    "Q29A": "Hard to calm down",
    "Q30A": "Fear trivial tasks",
    "Q31A": "No enthusiasm",
    "Q32A": "Irritated by interruptions",
    "Q33A": "Nervous tension",
    "Q34A": "Worthless",
    "Q35A": "Intolerant of delays",
    "Q36A": "Terrified",
    "Q37A": "Hopeless future",
    "Q38A": "Life meaningless",
    "Q39A": "Agitated",
    "Q40A": "Fear panic situations",
    "Q41A": "Trembling",
    "Q42A": "No initiative",
}

# TIPI items and labels

TIPI_ITEMS: List[str] = [f"TIPI{i}" for i in range(1, 11)]

TIPI_LABELS: Dict[str, str] = {
    "TIPI1": "Extraverted",
    "TIPI2": "Critical",
    "TIPI3": "Dependable",
    "TIPI4": "Anxious",
    "TIPI5": "Open to new experiences",
    "TIPI6": "Reserved",
    "TIPI7": "Sympathic",
    "TIPI8": "Disorganized",
    "TIPI9": "Calm",
    "TIPI10": "Conventional",
}

# VCL items and labels

VCL_ITEMS: List[str] = [f"VCL{i}" for i in range(1, 17)]

VCL_LABELS: Dict[str, str] = {
    "VCL1": "boat",
    "VCL2": "incoherent",
    "VCL3": "pallid",
    "VCL4": "robot",
    "VCL5": "audible",
    "VCL6": "cuivocal",
    "VCL7": "paucity",
    "VCL8": "epistemology",
    "VCL9": "florted",
    "VCL10": "decide",
    "VCL11": "pastiche",
    "VCL12": "verdid",
    "VCL13": "abysmal",
    "VCL14": "lucid",
    "VCL15": "betray",
    "VCL16": "funny",
}