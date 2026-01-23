# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

**brainsets** is a Python package for processing neural datasets into a standardized format. It provides a CLI and framework for downloading raw neural data from various sources and transforming it into a consistent HDF5-based format using the `temporaldata` library.

## Common Commands

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest                              # Full test suite
pytest tests/test_signal.py         # Single test file

# Linting (via pre-commit)
pre-commit install
pre-commit run --all-files

# CLI usage
brainsets config                    # Configure data directories
brainsets list                      # List available brainsets
brainsets prepare <brainset>        # Download and process a brainset
brainsets prepare <brainset> --cores 8  # Parallel processing

# Run pipeline directly (bypasses CLI, useful for development)
python -m brainsets.runner <pipeline.py> --raw-dir=<path> --processed-dir=<path> -c 4
python -m brainsets.runner <pipeline.py> --raw-dir=<path> --processed-dir=<path> --single=<session_id>  # Single session
```

## Architecture

### Core Components

- **`brainsets/pipeline.py`**: `BrainsetPipeline` abstract base class that all dataset pipelines inherit from. Defines the three-step workflow: `get_manifest()` → `download()` → `process()`

- **`brainsets/runner.py`**: Executes pipelines with Ray-based parallelization. Handles CLI argument parsing, worker pool management, and progress tracking

- **`brainsets/descriptions.py`**: Pydantic dataclasses for metadata (`BrainsetDescription`, `SubjectDescription`, `SessionDescription`, `DeviceDescription`)

- **`brainsets/processing/signal.py`**: Signal processing utilities (LFP band extraction, downsampling). Uses MNE library for filtering

- **`brainsets/taxonomy/`**: Enums for standardized labels (Species, Task, RecordingTech, BrainRegion, etc.)

### Pipeline Structure

Each dataset has a pipeline in `brainsets_pipelines/<brainset_id>/pipeline.py` containing:
- A `Pipeline` class inheriting from `BrainsetPipeline`
- Optional inline metadata block (`# /// brainset-pipeline`) for dependencies and Python version
- Implementation of `get_manifest()`, `download()`, and `process()` methods

Output files are saved as HDF5 using `temporaldata.Data.to_hdf5()`.

## Key Data Types

From `temporaldata` library:
- `Data`: Main container for session data
- `IrregularTimeSeries`: Spike times, behavioral events
- `RegularTimeSeries`: Continuous signals (LFP bands)
- `Interval`: Time intervals (trials, train/valid/test splits)
- `ArrayDict`: Metadata arrays (unit properties)

## Data Paths (Mila cluster)

- Raw data: `/network/projects/neuro-galaxy/data/raw`
- Processed data: `$SCRATCH/data/processed` (i.e., `/network/scratch/h/hee-woon.ryoo/data/processed`)

## SLURM Job Submission

- Use `--partition=long` (not `main`) when submitting multiple short jobs
- The `main` partition has stricter CPU limits that cause job queuing issues
- For LFP processing jobs that load large broadband files, use `--mem=64G` (32G may OOM)
