# OpenNeuro Utilities

This package provides utilities for building pipelines that process publicly available EEG and iEEG datasets from [OpenNeuro](https://openneuro.org/).

**📖 Documentation can be found in the official brainsets docs:** See [Creating an OpenNeuro Pipeline](https://brainsets.readthedocs.io/en/latest/concepts/openneuro_pipeline.html) for the complete guide on building OpenNeuro pipelines, including:

- Quick start (minimal 3-line pipeline)
- Required and optional configuration attributes
- Advanced channel remapping and processing customization
- Real-world examples

## API Reference

For detailed API documentation, see [`brainsets.utils.openneuro`](https://brainsets.readthedocs.io/en/latest/package/utils.html#openneuro).

## Key Classes

- `OpenNeuroPipeline` - Base class for OpenNeuro pipelines
- `OpenNeuroEEGPipeline` - Specialized for EEG datasets
- `OpenNeuroIEEGPipeline` - Specialized for iEEG datasets