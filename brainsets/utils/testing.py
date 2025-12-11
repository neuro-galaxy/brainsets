import atexit
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Union

import click

from brainsets._cli.cli_prepare import prepare

DEFAULT_TEST_SESSIONS = {
    "pei_pandarinath_nlb_2021": None,
    "perich_miller_population_2018": "c_20131219_center_out_reaching",
    "allen_visual_coding_ophys_2016": "717913184",
    "churchland_shenoy_neural_2012": "jenkins_20090912",
    "flint_slutzky_accurate_2012": "flint_2012_e1",
}


def download_single_sessions(
    brainsets: Optional[List[str]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    keep_raw: bool = False,
    verbose: bool = False,
) -> Path:
    """Download and process a single session from specified brainsets.

    This utility is designed for CI and local testing purposes. It downloads
    one session from each specified brainset using predefined session IDs
    and downloads it to the specified output directory.

    Parameters
    ----------
    brainsets : list of str, optional
        List of brainset names to download. If None, downloads from all
        available brainsets in DEFAULT_TEST_SESSIONS.
    output_dir : str or Path, optional
        Directory to store downloaded data. If None, creates a temporary
        directory that is cleaned up when the program exits.
    keep_raw : bool, default False
        If True, keeps raw data after processing. If False, raw data is
        deleted immediately after each brainset is processed.
    verbose : bool, default False
        If True, prints detailed progress information.

    Returns
    -------
    Path
        Path to the processed data directory.

    Examples
    --------
    >>> from brainsets.utils.testing import download_single_sessions
    >>> # Download from specific brainsets
    >>> path = download_single_sessions(
    ...     brainsets=["pei_pandarinath_nlb_2021"],
    ...     keep_raw=False
    ... )
    >>> # Download to a specific directory
    >>> path = download_single_sessions(
    ...     brainsets=["churchland_shenoy_neural_2012"],
    ...     output_dir="/tmp/test_data",
    ...     keep_raw=True
    ... )
    """
    if output_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="brainsets_test_")
        base_dir = Path(temp_dir)
        atexit.register(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
    else:
        base_dir = Path(output_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = base_dir / "raw"
    processed_dir = base_dir / "processed"

    if brainsets is None:
        brainsets = list(DEFAULT_TEST_SESSIONS.keys())

    for brainset in brainsets:
        if brainset not in DEFAULT_TEST_SESSIONS:
            raise ValueError(
                f"Brainset '{brainset}' not in DEFAULT_TEST_SESSIONS. "
                f"Available: {list(DEFAULT_TEST_SESSIONS.keys())}"
            )

        session_id = DEFAULT_TEST_SESSIONS[brainset]

        extra_args = []
        if session_id is not None:
            extra_args.extend(["-s", session_id])

        ctx = click.Context(prepare)
        ctx.args = extra_args

        ctx.invoke(
            prepare,
            brainset=brainset,
            cores=1,
            verbose=verbose,
            use_active_env=False,
            raw_dir=str(raw_dir),
            processed_dir=str(processed_dir),
            local=False,
        )

        if not keep_raw:
            shutil.rmtree(raw_dir / brainset, ignore_errors=True)

    return processed_dir
