#!/usr/bin/env python3
"""Fix hardcoded Quantity_initial_EJ reference in statistical_tests.py"""

# Read the file with UTF-8 encoding
with open('statistical_tests.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the hardcoded reference
old_line = '    feats = ["logQ" if use_log else ("_Q_eff" if "_Q_eff" in df.columns else "Quantity_initial_EJ")]'
new_lines = '''    base_qty = "reserves_initial_EJ" if "reserves_initial_EJ" in df.columns else "Quantity_initial_EJ"
    feats = ["logQ" if use_log else ("_Q_eff" if "_Q_eff" in df.columns else base_qty)]'''

if old_line in content:
    content = content.replace(old_line, new_lines)
    print("Found and fixed hardcoded reference")
else:
    print("Hardcoded reference not found")

# Write back with UTF-8 encoding
with open('statistical_tests.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fix complete")
