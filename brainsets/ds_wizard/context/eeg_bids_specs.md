# BIDS EEG Specification

## EEG Recording Data

### File Templates
```
sub-<label>/
[ses-<label>/]
eeg/
  sub-<label>[_ses-<label>]_task-<label>[_acq-<label>][_run-<index>]_eeg.<extension>
  sub-<label>[_ses-<label>]_task-<label>[_acq-<label>][_run-<index>]_eeg.json
  sub-<label>[_ses-<label>]_task-<label>[_acq-<label>][_run-<index>]_events.json
  sub-<label>[_ses-<label>]_task-<label>[_acq-<label>][_run-<index>]_events.tsv
  sub-<label>[_ses-<label>]_task-<label>[_acq-<label>][_run-<index>][_recording-<label>]_physio.json
  sub-<label>[_ses-<label>]_task-<label>[_acq-<label>][_run-<index>][_recording-<label>]_physio.tsv.gz
  sub-<label>[_ses-<label>]_task-<label>[_acq-<label>][_run-<index>][_recording-<label>]_stim.json
  sub-<label>[_ses-<label>]_task-<label>[_acq-<label>][_run-<index>][_recording-<label>]_stim.tsv.gz
```

### Supported Formats
| Format | Extension(s) | Description |
|--------|-------------|-------------|
| European data format | `.edf` | Single file, edf+ permitted |
| BrainVision Core Data Format | `.vhdr`, `.vmrk`, `.eeg` | File triplet |
| EEGLAB | `.set`, `.fdt` | .set with optional .fdt |
| Biosemi | `.bdf` | Single file, bdf+ permitted |

**Recommended:** European data format or BrainVision data format.

## Sidecar JSON (*_eeg.json)

### Required Fields
| Key | Data Type | Description |
|-----|-----------|-------------|
| EEGReference | string | Reference scheme description |
| SamplingFrequency | number | Sampling frequency in Hz |
| PowerLineFrequency | number or "n/a" | Power grid frequency (50/60 Hz) |
| SoftwareFilters | object or "n/a" | Applied temporal software filters |

### Recommended Fields
| Key | Data Type | Description |
|-----|-----------|-------------|
| CapManufacturer | string | Cap manufacturer name |
| CapManufacturersModelName | string | Cap model designation |
| EEGChannelCount | integer | Number of EEG channels |
| ECGChannelCount | integer | Number of ECG channels |
| EMGChannelCount | integer | Number of EMG channels |
| EOGChannelCount | integer | Number of EOG channels |
| MISCChannelCount | integer | Number of miscellaneous channels |
| TriggerChannelCount | integer | Number of trigger channels |
| RecordingDuration | number | Recording length in seconds |
| RecordingType | string | "continuous", "discontinuous", or "epoched" |
| EpochLength | number | Individual epoch duration (if epoched) |
| EEGGround | string | Ground electrode location |
| HeadCircumference | number | Head circumference in cm |
| EEGPlacementScheme | string | Electrode placement scheme |
| HardwareFilters | object or "n/a" | Applied temporal hardware filters |
| SubjectArtefactDescription | string | Description of observed artifacts |

### Hardware/Task/Institution Information
| Key | Level | Description |
|-----|--------|-------------|
| Manufacturer | RECOMMENDED | Equipment manufacturer |
| ManufacturersModelName | RECOMMENDED | Equipment model name |
| SoftwareVersions | RECOMMENDED | Software version |
| DeviceSerialNumber | RECOMMENDED | Equipment serial number |
| TaskName | REQUIRED | Task name |
| TaskDescription | RECOMMENDED | Longer task description |
| Instructions | RECOMMENDED | Instructions given to participants |
| InstitutionName | RECOMMENDED | Institution name |
| InstitutionAddress | RECOMMENDED | Institution address |

## Channels Description (*_channels.tsv)

### Template
```
sub-<label>[_ses-<label>]_task-<label>[_acq-<label>][_run-<index>]_channels.tsv
sub-<label>[_ses-<label>]_task-<label>[_acq-<label>][_run-<index>]_channels.json
```

### Required Columns (in order)
| Column | Data Type | Description |
|--------|-----------|-------------|
| name | string | Channel label (unique) |
| type | string | Channel type (uppercase) |
| units | string | Physical units |

### Optional Columns
| Column | Data Type | Description |
|--------|-----------|-------------|
| description | string | Brief description |
| sampling_frequency | number | Channel sampling rate in Hz |
| reference | string | Reference electrode name(s) |
| low_cutoff | number | High-pass filter frequency |
| high_cutoff | number | Low-pass filter frequency |
| notch | string | Notch filter frequencies |
| status | string | "good" or "bad" |
| status_description | string | Noise/artifact description |

### Channel Types
| Type | Description |
|------|-------------|
| AUDIO | Audio signal |
| EEG | Electroencephalogram channel |
| EOG | Generic electrooculogram |
| ECG | Electrocardiogram |
| EMG | Electromyogram |
| HEOG | Horizontal EOG |
| VEOG | Vertical EOG |
| MISC | Miscellaneous |
| TRIG | Trigger channel |
| REF | Reference channel |
| Others | RESP, TEMP, PUPIL, GSR, PPG, SYSCLOCK, EYEGAZE |

## Electrodes Description (*_electrodes.tsv)

### Template
```
sub-<label>[_ses-<label>][_task-<label>][_acq-<label>][_run-<index>][_space-<label>]_electrodes.tsv
sub-<label>[_ses-<label>][_task-<label>][_acq-<label>][_run-<index>][_space-<label>]_electrodes.json
```

### Required Columns (in order)
| Column | Data Type | Description |
|--------|-----------|-------------|
| name | string | Electrode contact point name (unique) |
| x | number | Position along x-axis |
| y | number | Position along y-axis |
| z | number | Position along z-axis |

### Recommended Columns
| Column | Data Type | Description |
|--------|-----------|-------------|
| type | string | Electrode type (cup, ring, clip-on, wire, needle) |
| material | string | Electrode material (Tin, Ag/AgCl, Gold) |
| impedance | number | Impedance in kOhm |

## Coordinate System JSON (*_coordsystem.json)

### Template
```
sub-<label>[_ses-<label>][_task-<label>][_acq-<label>][_space-<label>]_coordsystem.json
```

### Required Fields
| Key | Data Type | Description |
|-----|-----------|-------------|
| EEGCoordinateSystem | string | Coordinate system for EEG sensors |
| EEGCoordinateUnits | string | Units: "m", "mm", "cm", "n/a" |

### Recommended Fields
| Key | Data Type | Description |
|-----|-----------|-------------|
| EEGCoordinateSystemDescription | string | Coordinate system description |
| FiducialsDescription | string | How fiducials were placed/measured |
| FiducialsCoordinates | object | Fiducial 3D positions |
| FiducialsCoordinateSystem | string | Fiducial coordinate system |
| FiducialsCoordinateUnits | string | Fiducial coordinate units |
| AnatomicalLandmarkCoordinates | object | Anatomical landmark 3D positions |
| AnatomicalLandmarkCoordinateSystem | string | Landmark coordinate system |
| AnatomicalLandmarkCoordinateUnits | string | Landmark coordinate units |

### Optional Fields
| Key | Description |
|-----|-------------|
| IntendedFor | BIDS URIs for associated files |

## Landmark Photos (*_photo.<extension>)

### Template
```
sub-<label>[_ses-<label>][_acq-<label>]_photo.jpg
sub-<label>[_ses-<label>][_acq-<label>]_photo.png
sub-<label>[_ses-<label>][_acq-<label>]_photo.tif
```

Photos of anatomical landmarks/fiducials (optional). May need cropping/blurring for privacy.

## Key Notes

- **Electrode vs Channel**: Electrode = physical contact point; Channel = analog-to-digital converter output
- **Reference/Ground**: Usually not recorded as channels themselves
- **Inheritance Principle**: Files should not be duplicated across runs; use inheritance
- **Coordinate Systems**: Must be consistent between electrodes, fiducials, and anatomical landmarks
- **Required Pairing**: If `*_electrodes.tsv` exists, `*_coordsystem.json` is required
- **RecordingType**: continuous (1 segment), epoched (multiple equal segments), discontinuous (multiple different segments)
