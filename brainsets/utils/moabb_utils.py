"""Utility functions for working with MOABB datasets.

This module provides general-purpose utilities for downloading and working with
MOABB (Mother of All BCI Benchmarks) datasets.

Example Usage in Pipeline
--------------------------
Instead of manually setting up directories and environment variables:

    # OLD approach (manual setup)
    from mne import get_config
    from moabb.utils import set_download_dir
    from moabb.datasets import PhysionetMI
    
    original_path = get_config("MNE_DATA")
    set_download_dir(str(raw_dir))
    os.environ["MNE_DATASETS_EEGBCI_PATH"] = str(raw_dir)
    dataset = PhysionetMI(imagined=True, executed=False)
    raw_data = dataset._get_single_subject_data(subject)["0"]

You can now use the download_file utility:

    # NEW approach (using utility)
    from brainsets.utils.moabb_utils import download_file
    from moabb.datasets import PhysionetMI
    
    dataset = PhysionetMI(imagined=True, executed=False)
    raw_data = download_file(dataset, subject=1, download_dir=raw_dir, session="0")

This works for any MOABB dataset (PhysionetMI, BNCI2014_001, etc.) without
needing to know the specific environment variable names or setup requirements.
"""

import os
from pathlib import Path
from typing import Union, Optional, Dict, Any
from mne import get_config
import numpy as np
from moabb.utils import set_download_dir
from moabb.datasets.base import BaseDataset
from moabb.paradigms import BaseParadigm


def download_file(
    dataset: BaseDataset,
    # paradigm: BaseParadigm,
    subject: Union[int, str],
    raw_dir: Union[str, Path],
    session: Optional[Union[int, str]] = None,
    env_var_suffix: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Download data for a single subject from a MOABB dataset.
    
    This is a general-purpose function that works with all MOABB datasets by handling
    the download directory setup and environment variable configuration.
    
    Parameters
    ----------
    dataset : BaseDataset
        An instantiated MOABB dataset object (e.g., PhysionetMI(), BNCI2014_001())
    subject : int or str
        The subject number/ID to download. Must be in dataset.subject_list
    raw_dir : str or Path
        Directory where the raw data will be downloaded
    session : int or str, optional
        Session identifier for datasets with multiple sessions per subject.
        If None, returns all sessions. Default is None.
    env_var_suffix : str, optional
        Suffix for the MNE_DATASETS environment variable (e.g., 'EEGBCI', 'BNCI').
        If None, attempts to infer from dataset class name. Default is None.
        
    Returns
    -------
    raw_data : dict
        Dictionary containing the raw MNE objects for the subject.
        - If session is None: returns all sessions as a dict with session keys
        - If session is specified: returns the raw data for that specific session
        
    Raises
    ------
    ValueError
        If the subject is not in the dataset's subject list
        If the specified session does not exist for the subject
        
    Examples
    --------
    Download data from PhysionetMI dataset:
    
    >>> from moabb.datasets import PhysionetMI
    >>> dataset = PhysionetMI(imagined=True, executed=False)
    >>> raw_data = download_file(dataset, subject=1, download_dir="./raw")
    
    Download specific session from BNCI2014_001:
    
    >>> from moabb.datasets import BNCI2014_001
    >>> dataset = BNCI2014_001()
    >>> raw_data = download_file(
    ...     dataset, 
    ...     subject=1, 
    ...     download_dir="./raw",
    ...     session="0train"
    ... )
    
    Notes
    -----
    - The function sets up the MNE download directory and appropriate environment
      variables for MOABB to find the data.
    - The original MNE_DATA config is preserved and not modified.
    - Some datasets (like PhysionetMI) have single sessions returned with key "0"
    - Other datasets (like BNCI2014_001) have multiple sessions with various keys
    """
    # Convert download_dir to Path object
    download_dir = Path(raw_dir)
    download_dir.mkdir(exist_ok=True, parents=True)
    
    # Validate subject is in dataset
    if subject not in dataset.subject_list:
        raise ValueError(
            f"Subject {subject} not found in dataset. "
            f"Available subjects: {dataset.subject_list}"
        )
        
    # Infer environment variable suffix from dataset class name if not provided
    if env_var_suffix is None:
        dataset_class_name = dataset.__class__.__name__
        # Common mappings for known datasets
        env_var_map = {
            'PhysionetMI': 'EEGBCI',
            'BNCI2014_001': 'BNCI',
            'BNCI2014_004': 'BNCI',
            'BNCI2015_001': 'BNCI',
            # 'Cho2017': 'GIGADB',
            # 'Weibo2014': 'EEGBCI',
            # 'Zhou2016': 'EEGBCI',
        }
        
        # Try to find a match
        env_var_suffix = env_var_map.get(dataset_class_name)
        
        # If not found, try to extract from dataset code
        if env_var_suffix is None and hasattr(dataset, 'code'):
            # Use the dataset code in uppercase, replacing hyphens with underscores
            env_var_suffix = dataset.code.upper().replace('-', '_')
    
    # Set environment variable for dataset-specific path
    if env_var_suffix:
        env_var_name = f"MNE_DATASETS_{env_var_suffix}_PATH"
        os.environ[env_var_name] = str(download_dir)
    
    # Download data for the subject
    # subject_data = dataset._get_single_subject_data(subject)
    subject_data = dataset.get_data(subjects=[subject])
    print("Subject:", subject_data)
    print("Subject Data:", subject_data.keys())
        
    # Remap session keys using encoded session values
    session_map = encode_session_values(list(subject_data.keys()), return_dict=True)
    subject_data = {str(session_map[key]): value for key, value in subject_data.items()}

    # Handle session selection
    if session is not None:
        session_key = str(session)
        if session_key not in subject_data:
            raise ValueError(
                f"Session '{session}' not found for subject {subject}. "
                f"Available sessions: {list(subject_data.keys())}"
            )
        return subject_data[session_key]
    
    # Return all sessions
    return subject_data

def encode_session_values(values, return_dict=False):
    """
    Encode a list of session values by mapping unique values to integers.

    Parameters
    ----------
    values : list
        List of values to encode

    Returns
    -------
    encoded_values : list
        List of encoded integer values
    """
    unique_values = list(set(values))
    value_to_int = {val: idx for idx, val in enumerate(unique_values)}
    encoded_values = [value_to_int[val] for val in values]
    if return_dict:
        return value_to_int
    else:
        return np.array(encoded_values)
