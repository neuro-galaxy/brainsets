# Script to run prepare_data.py for all subjects

import os
from prepare_data_mind import run_pipeline

# Set input and output directories
INPUT_DIR = r'C:\Users\marys\Documents\Projects\Neuro-Galaxy\data\raw\labo\Hand-Expressions'
OUTPUT_DIR = r'C:\Users\marys\Documents\Projects\Neuro-Galaxy\data\processed\labo\Hand-Expressions'

FOLDERS = ["Hand-Expressions_P001_11-07-2025_11-24-26",
           'Hand-Expressions_P002_11-07-2025_11-49-20',
           'Hand-Expressions_P003_11-07-2025_12-49-40',
           'Hand-Expressions_P004_11-10-2025_22-38-35',
           'Hand-Expressions_P005_11-10-2025_23-01-08',
           'Hand-Expressions_P006_11-12-2025_15-57-57']

SUBJECT_IDS = ["P001",
               "P002",
               "P003",
               "P004",
               "P005",
               "P006"]

SEX = ["MALE",
       "MALE",
       "FEMALE",
       "MALE",
       "FEMALE",
       "MALE"]

SESSION_IDS = FOLDERS

HANDEDNESS = ["LEFT",
              "RIGHT",
              "RIGHT",
              "RIGHT",
              "RIGHT",
              "RIGHT"]

OPACITY = [0,
           0,
           0,
           0,
           0,
           0]

# Loop through all subjects
for folder, subject_id, sex, session_id, handedness, opacity in zip(FOLDERS, SUBJECT_IDS, SEX, SESSION_IDS, HANDEDNESS, OPACITY):

    input_dir = os.path.join(INPUT_DIR, folder)

    run_pipeline(
        input_dir=input_dir,
        output_dir=OUTPUT_DIR,
        subject_id=subject_id,
        sex=sex,
        handedness=handedness,
        session_id=session_id
    )

print("All subjects processed.")