"""
Few-shot examples for in-context learning in Dataset Wizard agents.
These examples help guide LLMs to produce more consistent outputs.
"""

from typing import Dict, List

# Metadata extraction examples
METADATA_EXAMPLES = [
    {
        "input": "Dataset ds001234 about motor imagery with 20 subjects",
        "output": """{
    "name": "Motor Imagery EEG Dataset",
    "brainset_name": "smith_motor_ds001234_2023",
    "version": "1.0.0",
    "dataset_id": "ds001234",
    "dataset_summary": "This dataset contains EEG recordings from 20 healthy participants performing motor imagery tasks. Participants were asked to imagine left or right hand movements while EEG was recorded using a 64-channel BioSemi system. The data includes multiple sessions per participant with rest periods between trials.",
    "task_description": "Participants performed motor imagery tasks involving imagined left and right hand movements. Each trial consisted of a cue period, imagination period, and rest period. The task was designed to study motor cortex activation patterns during motor imagery.",
    "task_category": "motor-imagery",
    "task_subcategory": "hand-movement",
    "authors": ["Smith, John", "Doe, Jane", "Johnson, Robert"],
    "date": "15-03-2023"
}""",
    },
    {
        "input": "Dataset ds005678 on sleep staging with polysomnography",
        "output": """{
    "name": "Overnight Sleep EEG Study",
    "brainset_name": "jones_sleep_ds005678_2022",
    "version": "2.1.0",
    "dataset_id": "ds005678",
    "dataset_summary": "Overnight polysomnography recordings from 35 participants across different age groups. The dataset includes full-night EEG, EOG, and EMG recordings following standard sleep monitoring protocols. Data was collected to study sleep architecture and sleep stage transitions across the lifespan.",
    "task_description": "Participants underwent overnight polysomnography in a sleep laboratory. Standard sleep monitoring included EEG, EOG, and EMG recordings throughout the night. Sleep stages were scored by certified technicians according to AASM guidelines.",
    "task_category": "sleep-staging",
    "task_subcategory": "overnight-sleep",
    "authors": ["Jones, Maria", "Lee, Kevin"],
    "date": "22-08-2022"
}""",
    },
]

# Channel mapping examples
CHANNEL_EXAMPLES = [
    {
        "input": "BioSemi 64-channel EEG system with standard 10-20 electrode names",
        "output": """{
    "channel_maps": {
        "biosemi_64": {
            "map_name": "biosemi_64",
            "device_name": "ActiveTwo",
            "device_manufacturer": "BioSemi",
            "channels": {
                "Fp1": {
                    "new_name": "Fp1",
                    "modality": "EEG",
                    "unit": "uV",
                    "x": -0.0349,
                    "y": 0.0838,
                    "z": -0.0054,
                    "confidence": 1.0
                },
                "Fp2": {
                    "new_name": "Fp2",
                    "modality": "EEG",
                    "unit": "uV",
                    "x": 0.0349,
                    "y": 0.0838,
                    "z": -0.0054,
                    "confidence": 1.0
                },
                "HEOG": {
                    "new_name": "HEOG",
                    "modality": "EOG",
                    "unit": "uV",
                    "x": null,
                    "y": null,
                    "z": null,
                    "confidence": 1.0
                },
                "VEOG": {
                    "new_name": "VEOG",
                    "modality": "EOG",
                    "unit": "uV",
                    "x": null,
                    "y": null,
                    "z": null,
                    "confidence": 1.0
                }
            }
        }
    }
}""",
    },
    {
        "input": "EGI 128-channel system with non-standard E1, E2, E3... naming",
        "output": """{
    "channel_maps": {
        "egi_128": {
            "map_name": "egi_128",
            "device_name": "HydroCel GSN 128",
            "device_manufacturer": "EGI",
            "channels": {
                "E1": {
                    "new_name": "Cz",
                    "modality": "EEG",
                    "unit": "uV",
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0896,
                    "confidence": 0.95
                },
                "E36": {
                    "new_name": "Fp1",
                    "modality": "EEG",
                    "unit": "uV",
                    "x": -0.0349,
                    "y": 0.0838,
                    "z": -0.0054,
                    "confidence": 0.95
                },
                "E125": {
                    "new_name": "HEOG_L",
                    "modality": "EOG",
                    "unit": "uV",
                    "x": null,
                    "y": null,
                    "z": null,
                    "confidence": 0.9
                }
            }
        }
    }
}""",
    },
]

# Recording info examples
RECORDING_EXAMPLES = [
    {
        "input": "Dataset with 2 participants (sub-01, sub-02), BIDS format EEG recordings. Each recording has different duration and channels based on its specific *_eeg.json and *_channels.tsv files.",
        "output": """{
    "recording_info": [
        {
            "recording_id": "sub-01_ses-01_task-rest",
            "subject_id": "01",
            "task_id": "rest",
            "channel_map_id": "biosemi_64",
            "duration_seconds": 600.0,
            "num_channels": 64,
            "participant_info": {
                "age": "25",
                "sex": "M",
                "handedness": "right"
            },
            "channels_to_remove": []
        }, 
        {
            "recording_id": "sub-02_ses-01_task-rest",
            "subject_id": "02",
            "task_id": "rest",
            "channel_map_id": "biosemi_64",
            "duration_seconds": 605.0,
            "num_channels": 63,
            "participant_info": {
                "age": "24",
                "sex": "F",
                "handedness": "right"
            },
            "channels_to_remove": []
        },
        {
            "recording_id": "sub-02_ses-02_task-action",
            "subject_id": "02",
            "task_id": "action",
            "channel_map_id": "biosemi_64",
            "duration_seconds": 610.0,
            "num_channels": 64,
            "participant_info": {
                "age": "24",
                "sex": "F",
                "handedness": "right"
            },
            "channels_to_remove": []
        }
    ]
}""",
    }
]


def format_examples_for_prompt(
    examples: List[Dict[str, str]], max_examples: int = 2
) -> str:
    """
    Format few-shot examples for inclusion in prompts.

    Args:
        examples: List of example dictionaries with 'input' and 'output' keys
        max_examples: Maximum number of examples to include

    Returns:
        Formatted string with examples
    """
    formatted = ["--- EXAMPLES OF CORRECT OUTPUT ---\n"]

    for i, example in enumerate(examples[:max_examples], 1):
        formatted.append(f"Example {i}:")
        formatted.append(f"Input: {example['input']}")
        formatted.append(f"Output:\n{example['output']}\n")

    formatted.append("--- END EXAMPLES ---\n")
    return "\n".join(formatted)


def get_metadata_examples_prompt(num_examples: int = 2) -> str:
    """Get formatted metadata extraction examples for prompts."""
    return format_examples_for_prompt(METADATA_EXAMPLES, num_examples)


def get_channel_examples_prompt(num_examples: int = 2) -> str:
    """Get formatted channel mapping examples for prompts."""
    return format_examples_for_prompt(CHANNEL_EXAMPLES, num_examples)


def get_recording_examples_prompt(num_examples: int = 1) -> str:
    """Get formatted recording info examples for prompts."""
    return format_examples_for_prompt(RECORDING_EXAMPLES, num_examples)
