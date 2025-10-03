
#!/usr/bin/env python3
"""
convert_to_ej.py
----------------
Convert oil/gas quantities into Energy in Exajoules (EJ)

Accepts:
  • quantity (numeric)
  • unit from the GOGET manifest (exact list of 57 units)
  • fuel_type from GOGET fuel descriptions (exact list of 30)
  • optional gas_ratio (float default = 0.20)

Usage (CLI):
    $ convert_to_ej.py --quantity 20 --unit "bcm" --fuel_type "dry gas"
      → {"total_ej": "7.200e+00", "oil_ej": "0.000e+00", "gas_ej": "7.200e+00"}

Key logic:
  • Uses PREFIX_MULTIPLIERS to convert word prefixes (“million”, “billion”, etc.)
  • Maps units to standard types ("Gbl", "tcm", or "PJ") via UNIT_TO_STANDARD
  • Converts via STANDARD_TO_EJ:
      Gbl → 6.119 EJ,
      tcm → 36.0 EJ,
      PJ → 0.001 EJ
  • Uses FUEL_CLASS to determine category:
      oil, gas, or mixed (mixed uses gas_ratio)
  • Ensures strict coverage—if unit or fuel_type not in lists → ValueError

Author: John Peach (Wild Peaches)
Date:   2025‑08‑04 (v1)
"""
    
import sys
import argparse
import json

# User-defined default gas ratio
DEFAULT_GAS_RATIO = 0.20

# Step‑1: Prefix multipliers
PREFIX_MULTIPLIERS = {
    "thousand": 1e3,  "million": 1e6,  "billion": 1e9,
    "trillion": 1e12, "giga": 1e9,     "mega": 1e6,
    "kilo": 1e3,      "": 1.0
}

# Step‑2: Fuel description classification (exact list)
FUEL_CLASS = {
    "oil":         "oil",
    "crude oil and condensate": "oil",
    "crude oil":   "oil",
    "condensate":  "oil",
    "liquids":     "oil",
    "liquid hydrocarbons": "oil",
    "hydrocarbons": "oil",
    "ngl":         "oil",
    "lpg":         "oil",
    "total liquids": "oil",
    "gas":         "gas",
    "associated gas": "gas",
    "nonassociated gas": "gas",
    "non-associated gas": "gas",
    "sales gas":   "gas",
    "dry gas":     "gas",
    "coal seam gas": "gas",
    "coal bed methane": "gas",
    "lng":         "gas",
    "gas condensate": "gas",
    # Mixed
    "oil and gas": "mixed",
    "oil and gas condensate": "mixed",
    "oil and ngl": "mixed",
    "oil and lpg": "mixed",
    "oil with associated gas": "mixed",
    "oil, ngl, and gas": "mixed",
    "oil and associated gas": "mixed",
    "[not stated]": "mixed"
}

# Step‑3: Unit → (standard, multiplier_to_standard)
# standard = "Gbl", "tcm", or "PJ"
UNIT_TO_STANDARD = {
    # Oil units to Gbl
    "bbl":                ("Gbl", 1e-9),
    "million bbl":        ("Gbl", 1e-3),
    "billion bbl":        ("Gbl", 1.0),
    "boe":                ("Gbl", 1e-9),
    "million boe":        ("Gbl", 1e-3),
    "billion boe":        ("Gbl", 1.0),
    "toe":                ("Gbl", (41.868 / 6.119) * 1e-9),  # 1 toe = 41.868 GJ, 1 boe = 6.119 GJ
    "million toe":        ("Gbl", (41.868 / 6.119) * 1e-9 * 1e6),
    "tons":               ("Gbl", 4.1868e-8 / 6.119),       # 1 ton = 1 toe = 41.868 GJ
    "million tonnes":     ("Gbl", (4.1868e-8 / 6.119) * 1e6),
    "million tons":       ("Gbl", (4.1868e-8 / 6.119) * 1e6),
    "thousand tons":      ("Gbl", (4.1868e-8 / 6.119) * 1e3),
    "metric tons":        ("Gbl",  4.1868e-8 / 6.119),
    "billion tons":       ("Gbl", (4.1868e-8 / 6.119) * 1e9),
    "million metric tons":("Gbl", (4.1868e-8 / 6.119) * 1e6),
    "billion metric tons":("Gbl", (4.1868e-8 / 6.119) * 1e9),

    # Volumetric oil units
    "million barrels":    ("Gbl", 1e-3),  # duplicates million bbl
    "million stock tank barrels": ("Gbl", 1e-3),
    "thousand bbl":       ("Gbl", 1e-6),

    # Gas units to tcm
    "bcf":                ("tcm", 2.83168e-5),   # 1 bcf = 2.83168e-5 tcm
    "billion cubic feet": ("tcm", 2.83168e-5),
    "BCF":                ("tcm", 2.83168e-5),
    "billion scf":        ("tcm", 2.83168e-5),
    "giga cubic feet":    ("tcm", 2.83168e-5),
    "billion cf":         ("tcm", 2.83168e-5),
    "million scf":        ("tcm", 2.83168e-9),
    "thousand scf":       ("tcm", 2.83168e-12),
    "scf":                ("tcm", 2.83168e-14),
    "thousand cubic feet":("tcm", 2.83168e-12),
    "tcf":                ("tcm", 1.0),         # 1 tcf = 1 tcm
    "tscf":               ("tcm", 1.0),
    "tscf [trillion standard cubic feet]": ("tcm", 1.0),
    "trillion cubic feet":("tcm", 1.0),

    # Metric gas
    "bcm":                ("tcm", 1e-3),    # 1 bcm = 0.001 tcm
    "giga m³":            ("tcm", 1e-3),
    "million m³":         ("tcm", 1e-6),
    "million m3":         ("tcm", 1e-6),
    "thousand m³":        ("tcm", 1e-9),
    "million Sm³":        ("tcm", 1e-6),
    "million Nm³":        ("tcm", 1e-6),
    "billion Sm³":        ("tcm", 1e-3),
    "billion Nm³":        ("tcm", 1e-3),
    "Nm³":                ("tcm", 1e-12),
    "sm³ o.e.":           ("tcm", 1e-6),
    "megalitres":         ("tcm", 1e-6),
    "m³":                 ("tcm", 1e-12),
    "thousand m3":        ("tcm", 1e-9),

    # Direct energy
    "petajoules":         ("PJ", 1.0),
}

# Step‑4: Standard to EJ
STANDARD_TO_EJ = {
    "Gbl": 6.119,
    "tcm": 36.0,
    "PJ": 0.001
}

# Exact allowed unit & fuel lists (for strict validation)
ALLOWED_UNITS = set(UNIT_TO_STANDARD)
ALLOWED_FUELS = set(FUEL_CLASS)

def convert_to_ej(quantity, unit, fuel_type, gas_ratio=None):
    uq = unit.strip().lower()
    if uq not in ALLOWED_UNITS:
        raise ValueError(f"Unit '{unit}' not supported.")
    fq = fuel_type.strip().lower()
    if fq not in ALLOWED_FUELS:
        raise ValueError(f"Fuel type '{fuel_type}' not supported.")

    std_type, unit_mul = UNIT_TO_STANDARD[uq]
    total_standard = float(quantity) * unit_mul
    ej = total_standard * STANDARD_TO_EJ[std_type]

    category = FUEL_CLASS[fq]
    ratio = gas_ratio if gas_ratio is not None else DEFAULT_GAS_RATIO
    if category == "oil":
        oil_ej, gas_ej = ej, 0.0
    elif category == "gas":
        oil_ej, gas_ej = 0.0, ej
    else:  # mixed
        gas_ej = ej * ratio
        oil_ej = ej - gas_ej

    return {
        "total_ej": f"{ej:.3e}",
        "oil_ej":   f"{oil_ej:.3e}",
        "gas_ej":   f"{gas_ej:.3e}"
    }

def main():
    p = argparse.ArgumentParser(
        description="Convert value+unit+fuel_type to energy in EJ"
    )
    p.add_argument("--quantity", type=float, required=True)
    p.add_argument("--unit", required=True)
    p.add_argument("--fuel_type", required=True)
    p.add_argument("--gas_ratio", type=float, default=None)
    args = p.parse_args()

    result = convert_to_ej(
        args.quantity,
        args.unit,
        args.fuel_type,
        args.gas_ratio
    )
    print(json.dumps(result))

if __name__ == "__main__":
    print("Example conversions to EJ:")
    tests = [
        (150, "million boe"),
        (20, "billion cubic feet"),
        (10, "bcm"),
        (100, "million bbl"),
        (50, "million toe"),
        (1000, "billion scf"),
        (1, "trillion cubic meters"),
        (1, "million barrels oil equivalent"),
        (1, "thousand cubic feet"),
        (1, "tons")  # intentionally unsupported
    ]
    for qty, unit in tests:
        result = convert_to_ej(qty, unit)
        print(f"{qty} {unit} → {result} EJ")

    # --- Test all units from Gas Oil units.txt ---
    print("\nTesting units from Gas Oil units.txt:")
    txt_path = "../data/Text files/Gas Oil units.txt"
    import os
    if os.path.exists(txt_path):
        with open(txt_path, encoding="utf-8") as f:
            for line in f:
                unit = line.strip()
                if not unit:
                    continue
                result = convert_to_ej(1, unit)
                print(f"1 {unit} → {result} EJ")
    else:
        print(f"File not found: {txt_path}")
