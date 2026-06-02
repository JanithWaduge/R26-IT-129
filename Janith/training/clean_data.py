import pandas as pd
import numpy as np

# ================================================
# LOAD
# ================================================
CSV_PATH = r'D:\R26-IT-129\Janith\keypoints_data.csv'

df = pd.read_csv(CSV_PATH)
print(f"Original shape: {df.shape}")
print(f"Original signs: {df['label'].nunique()}")

# ================================================
# STEP 1 — TYPOS FIX
# ================================================
typo_fixes = {
    'Calender'   : 'Calendar',
    'Lisence'    : 'License',
    'Rsearch'    : 'Research',
    'Quize'      : 'Quiz',
    'Quessionaire': 'Questionnaire',
    'Quize'      : 'Quiz',
    'whiteboard Marker': 'Whiteboard Marker',
}

df['label'] = df['label'].replace(typo_fixes)
print(f"\n✅ Typos fixed!")

# ================================================
# STEP 2 — REMOVE 1-SAMPLE SIGNS
# ================================================
sign_counts = df['label'].value_counts()

# 1 sample ඇති signs list
one_sample_signs = sign_counts[sign_counts < 2].index.tolist()
print(f"\n⚠️  Removing {len(one_sample_signs)} signs with only 1 sample:")
for s in one_sample_signs:
    print(f"   ❌ {s}")

# Remove
df = df[~df['label'].isin(one_sample_signs)]

# ================================================
# STEP 3 — FINAL CHECK
# ================================================
print(f"\n{'='*50}")
print(f"✅ Final samples : {len(df)}")
print(f"✅ Final signs   : {df['label'].nunique()}")
print(f"\n📋 Samples per sign:")
print(df['label'].value_counts().to_string())

# ================================================
# SAVE CLEANED CSV
# ================================================
CLEAN_CSV = r'D:\R26-IT-129\Janith\keypoints_clean.csv'
df.to_csv(CLEAN_CSV, index=False)
print(f"\n✅ Saved: {CLEAN_CSV}")