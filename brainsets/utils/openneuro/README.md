# Creating OpenNeuro Pipelines

> Process brain recordings from [OpenNeuro](https://openneuro.org/) with minimal boilerplate code

This guide shows you how to build a pipeline that automatically downloads, processes, and standardizes publicly available EEG and iEEG datasets.

## What's an OpenNeuro Pipeline?

A pipeline is a Python class that that extends either `OpenNeuroEEGPipeline` or `OpenNeuroIEEGPipeline`, and automates the entire workflow:

```
OpenNeuro S3 → Download → Process → Save to HDF5
```

**Your pipeline handles:**

- 🔍 Discovering recordings from OpenNeuro's S3 bucket based on modality (EEG or iEEG)
- ⬇️ Downloading BIDS-compliant files
- 🔧 Processing: extracting data (signal) and metadata (channel names and types)
- 💾 Storing: organized HDF5 files with full provenance

## Get Started in 3 Minutes

Here's a working minimal pipeline:

```python
from brainsets.utils.openneuro import OpenNeuroEEGPipeline

class Pipeline(OpenNeuroEEGPipeline):
    brainset_id = "my_sleep_study_2024"
    dataset_id = "ds005555"
    origin_version = "1.0.0"  # Check OpenNeuro for this!
    
    description = "Sleep recordings from OpenNeuro"
```

**That's it!** The rest is inherited. To run it:

```bash
uv run brainsets prepare my_sleep_study_2024
```

---

## Real-World Examples

Before diving into details, check out working implementations:


| Example                                                                                                 | Use When                                             | Complexity  |
| ------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- | ----------- |
| `[shirazi_hbnr1_ds005505_2024](../../../brainsets_pipelines/shirazi_hbnr1_ds005505_2024/pipeline.py)`   | All recordings have identical channels               | ⭐ Simple    |
| `[klinzing_sleep_ds005555_2024](../../../brainsets_pipelines/klinzing_sleep_ds005555_2024/pipeline.py)` | Different recordings need different channel mappings | ⭐⭐⭐ Complex |


---

## The Three Required Attributes

Every pipeline **must** have these three:

### 1. `dataset_id` - Which OpenNeuro dataset?

The dataset identifier. OpenNeuro auto-normalizes formats:

```python
dataset_id = "ds005555"      # ✅ Standard format
dataset_id = "5555"          # ✅ Also works (auto-normalized to ds005555)
dataset_id = "ds5555"        # ✅ Also works
```

### 2. `brainset_id` - Your unique name

A descriptive ID for your processed brainset:

```python
brainset_id = "klinzing_sleep_ds005555_2024"
#             └─ institution
#                        └─ dataset nickname
#                                    └─ year
```

### 3. `origin_version` - The dataset version you used

⚠️ **CRITICAL: This must be hardcoded to the exact version you tested with.**

```python
origin_version = "1.0.0"
```

**Why hardcode?** OpenNeuro datasets are versioned. A newer version might have different subjects, missing files, or structural changes. Hardcoding ensures you record which version your pipeline was originally built and tested with—so if the dataset evolves, you know what version was intended.

**What happens on mismatch?** If a newer version exists on OpenNeuro when your pipeline runs, the code will download the latest version but emit a warning:

```
⚠️ Dataset version '1.0.0' was used to create the brainset pipeline for dataset 'ds005555',
but the latest available version on OpenNeuro is '1.2.0'.
Downloading data or running the pipeline now will use the latest version,
which may differ from the original version used, potentially causing errors or inconsistencies.
Check the CHANGES file of the dataset for details about the differences between versions.
```

**How to find the version:** 
1. Visit [https://openneuro.org](https://openneuro.org/) and navigate to your dataset
2. Look for "Snapshots" or "Version History"
3. Note the latest version tag (e.g., `1.0.0`, `2.1.3`)
4. For details on changes between versions, check the `CHANGES` file in the dataset

**When versions don't match:**

| Option | When | How |
|--------|------|-----|
| Ignore warning | Changes are minor/unrelated to your pipeline | Just run it |
| Update version | You've tested with the new version | Update `origin_version`, test thoroughly, and re-create your pipeline |
| Use old version | Changes break your pipeline | Download from OpenNeuro archives manually |

---

## Optional Attributes (with sensible defaults)

Want to customize? These are all optional:

### `description` (str)

Human-readable description that appears in metadata:

```python
description = (
    "The Bitbrain Open Access Sleep (BOAS) dataset contains simultaneous "
    "recordings from a clinical PSG system and wearable EEG headband."
)
```

### `subject_ids` (list[str])

Process only specific subjects (default: all):

```python
subject_ids = ["sub-01", "sub-02", "sub-03"]
```

### `derived_version` (str)

Version of your processed brainset (default: `"1.0.0"`):

```python
derived_version = "1.0.0"
```

Increment this when you change processing logic or channel configs.

### `split_ratios` (tuple[float, float])

Train/validation time split (default: `(0.9, 0.1)`):

```python
split_ratios = (0.8, 0.2)  # 80% train, 20% validation
```

### `CHANNEL_NAME_REMAPPING` (dict, EEG only)

Rename raw channel names from the original dataset to standardized names.  
**Dictionary structure:**  

- **Keys** are the original/recorded channel names as strings (e.g., those found in the raw data).  
- **Values** are the standardized names to which you wish to map them as strings.

```python
CHANNEL_NAME_REMAPPING = {
    "PSG_F3": "F3",
    "PSG_F4": "F4",
    "PSG_EOG": "EOG",
}
```

### `TYPE_CHANNELS_REMAPPING` (dict)

Group channels by physiological type.

**Dictionary structure:**  

- **Keys:** Strings representing physiological channel types (e.g., `"EEG"`, `"EOG"`, `"EMG"`).
- **Values:** Lists of string channel names that belong to the given type.  
These names can be either the original channel names as found in the raw dataset or the standardized names you have mapped them to.

```python
TYPE_CHANNELS_REMAPPING = {
    "EEG": ["F3", "F4", "C3", "C4"],
    "EOG": ["EOG"],
    "EMG": ["EMG"],
}
```

> **Best Practice:**  
> While brainsets permits flexibility in naming schemes to accommodate various datasets, we recommend using **UPPERCASE names** for all channel names and types—both keys and values—wherever possible. This helps to align with widespread EEG and iEEG naming conventions and ensures consistency across datasets, promoting standardization without reducing adaptability.

---

## Advanced channel name and type remappings: Customize Per Recording

If you need different channel mappings or groupings for particular recordings (e.g., based on acquisition type, subject, or any other property), override these methods:

### `get_channel_name_remapping(recording_id)`

Return different channel mappings based on the recording:

```python
def get_channel_name_remapping(self, recording_id):
    if "acq-headband" in recording_id:
        return {
            "HB_1": "AF7",
            "HB_2": "AF8",
            "HB_PULSE": "PULSE",
        }
    return {
        "PSG_F3": "F3",
        "PSG_F4": "F4",
        "PSG_EOG": "EOG",
    }
```

### `get_type_channels_remapping(recording_id)`

Return different channel groupings based on the recording:

```python
def get_type_channels_remapping(self, recording_id):
    if "acq-headband" in recording_id:
        return {
            "EEG": ["AF7", "AF8"],
            "PPG": ["PULSE"],
        }
    return {
        "EEG": ["F3", "F4"],
        "EOG": ["EOG"],
    }
```

### `process(download_output)`

Add custom processing beyond the default:

```python
def process(self, download_output):
    # Get default processing
    result = self._process_common(download_output)
    if result is None:
        return  # Already processed
    
    data, store_path = result
    
    # Add your custom processing here
    # e.g., apply filters, remove artifacts
    
    # Save the result
    import h5py
    from brainsets import serialize_fn_map
    with h5py.File(store_path, "w") as file:
        data.to_hdf5(file, serialize_fn_map=serialize_fn_map)
```

### `generate_splits(domain, subject_id, session_id)`

Customize how train/validation splits are created:

```python
def generate_splits(self, domain, subject_id, session_id):
    # Custom split logic here
    # Return a Data object with split information
    ...
```

---

## What's Next?

1. ✅ Pick a dataset from [OpenNeuro](https://openneuro.org/)
2. ✅ Copy the minimal example above
3. ✅ Update `dataset_id`, `brainset_id`, `origin_version`
4. ✅ Add channel name and/or type mappings if needed
5. ✅ Run: `uv run brainsets prepare <my_brainset_id>`
6. ✅ Done!

Questions? Check the base class docstrings in `[pipeline.py](pipeline.py)` for detailed API docs.