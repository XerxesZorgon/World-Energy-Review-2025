#!/usr/bin/env python3
"""
convert_coal_to_ej.py
-----------------------
Convert coal reserves or production given in **million tonnes (Mt)** to energy

Only supports these coal types:
  - Bituminous
  - Subbituminous
  - Lignite
  - Anthracite
  - Bituminous and Subbituminous
  - Anthracite & Bituminous
  - Subbituminous / Lignite

How it works:
  1. Uses standardized net heating values (upper-range LHV) per tonne:
       - Anthracite:          27.70 GJ/t
       - Bituminous:          27.60 GJ/t
       - Subbituminous:       18.80 GJ/t
       - Lignite:             14.40 GJ/t
     (Source: Canada Energy Regulator, 2021–22 conversion tables) :contentReference[oaicite:1]{index=1}

  2. Mixed types are calculated as straight averages of appropriate base types.

  3. Conversion to EJ:
        energy_EJ = coal_Mt × heating_GJ_per_t × 1e6 / 1e18
                  = Mt × heating_GJ_per_t × 1e-3
     The result is rounded and formatted in scientific notation with **3 significant digits**.

Usage (as a module):
    from convert_coal_to_ej import convert_coal_to_ej

    r = convert_coal_to_ej(50, "Bituminous")
    # r -> 1.380e+00 (i.e. 1.380 EJ)

Usage (CLI):
    $ python convert_coal_to_ej.py --quantity 100 --coal_type "Subbituminous / Lignite"
    {"coal_type": "Subbituminous / Lignite", "quantity_Mt": 100.0, "energy_EJ": "1.660e+00"}

Author: John Peach (Wild Peaches)
Date:   2025‑08‑04
"""

import argparse
import json

# Net calorific values (lower heating value) in GJ per tonne (t)
# Source: Canada Energy Regulator conversion tables, 2021–22 edition :contentReference[oaicite:2]{index=2}
_HEAT_GJ_PER_T = {
    "Anthracite":            27.70,
    "Bituminous":            27.60,
    "Subbituminous":         18.80,
    "Lignite":               14.40
}

# Map exact supported coal types to their energy density (GJ/t)
_COAL_ENERGY = {
    "Anthracite":            _HEAT_GJ_PER_T["Anthracite"],
    "Bituminous":            _HEAT_GJ_PER_T["Bituminous"],
    "Subbituminous":         _HEAT_GJ_PER_T["Subbituminous"],
    "Lignite":               _HEAT_GJ_PER_T["Lignite"],
    "Bituminous and Subbituminous": (_HEAT_GJ_PER_T["Bituminous"] + _HEAT_GJ_PER_T["Subbituminous"]) / 2,
    "Anthracite & Bituminous":       (_HEAT_GJ_PER_T["Anthracite"] + _HEAT_GJ_PER_T["Bituminous"]) / 2,
    "Subbituminous / Lignite":       (_HEAT_GJ_PER_T["Subbituminous"] + _HEAT_GJ_PER_T["Lignite"]) / 2
}

def convert_coal_to_ej(quantity_Mt: float, coal_type: str) -> str:
    """
    Convert coal quantity (Mt) into energy in Exajoules (EJ).

    Parameters:
      quantity_Mt : float
        Coal mass in million tonnes.
      coal_type : str
        One of the seven supported coal types (case-insensitive).

    Returns:
      energy_EJ : str
        Energy content in EJ formatted in 1.234e+01 style (scientific, 3 significant digits).

    Raises:
      ValueError : if coal_type is not one of the predefined types.
    """
    ct = coal_type.strip()
    # Standardize case (must match exactly one of the keys)
    found = None
    for name in _COAL_ENERGY:
        if name.lower() == ct.lower():
            found = name
            break
    if found is None:
        raise ValueError(f"Unsupported coal type: '{coal_type}'. "
                         f"Supported types: {sorted(_COAL_ENERGY.keys())}")

    # GJ/t value
    gj_per_t = _COAL_ENERGY[found]

    # Mt to EJ conversion:
    #   energy_EJ = quantity * GJ_per_t * 1e-3
    total_ej = quantity_Mt * gj_per_t * 1e-3

    return f"{total_ej:.3e}"

def main():
    parser = argparse.ArgumentParser(
        description='Convert coal mass (Mt) and coal_type to energy in EJ'
    )
    parser.add_argument('--quantity', type=float, required=True,
                        help='Coal quantity in million tonnes (Mt)')
    parser.add_argument('--coal_type', type=str, required=True,
                        help='Coal type (exact case-insensitive match): '
                             f'{", ".join(_COAL_ENERGY.keys())}')
    args = parser.parse_args()

    energy_str = convert_coal_to_ej(args.quantity, args.coal_type)
    output = {
        "coal_type": args.coal_type.strip(),
        "quantity_Mt": args.quantity,
        "energy_EJ": energy_str
    }
    print(json.dumps(output))

if __name__ == "__main__":
    main()
