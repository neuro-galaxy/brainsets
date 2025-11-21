import argparse
import os
import subprocess


def download_sleep_cassette(output_dir):
    """Download Sleep Cassette (SC) study files from PhysioNet."""
    base_url = "https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/"

    sc_dir = os.path.join(output_dir, "sleep-cassette")
    os.makedirs(sc_dir, exist_ok=True)

    wget_cmd = [
        "wget",
        "-r",
        "-N",
        "-c",
        "-np",  # Don't ascend to parent directories
        "-nH",  # Don't create hostname directory
        "--cut-dirs=4",  # Skip 4 URL path components to avoid deep nesting
        "-P",
        sc_dir,
        "-A",
        "*.edf",
        base_url,
    ]

    subprocess.run(wget_cmd, check=True)


def download_sleep_telemetry(output_dir):
    """Download Sleep Telemetry (ST) study files from PhysioNet."""
    base_url = "https://physionet.org/files/sleep-edfx/1.0.0/sleep-telemetry/"

    st_dir = os.path.join(output_dir, "sleep-telemetry")
    os.makedirs(st_dir, exist_ok=True)

    wget_cmd = [
        "wget",
        "-r",
        "-N",
        "-c",
        "-np",  # Don't ascend to parent directories
        "-nH",  # Don't create hostname directory
        "--cut-dirs=4",  # Skip 4 URL path components to avoid deep nesting
        "-P",
        st_dir,
        "-A",
        "*.edf",
        base_url,
    ]

    subprocess.run(wget_cmd, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Sleep-EDF Database files from PhysioNet"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./raw",
        help="Output directory for downloaded files",
    )
    parser.add_argument(
        "--study_type",
        type=str,
        choices=["sc", "st", "both"],
        default="both",
        help="Which study to download: 'sc' (Sleep Cassette), 'st' (Sleep Telemetry), or 'both'",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.study_type in ["sc", "both"]:
        print("Downloading Sleep Cassette (SC) files...")
        download_sleep_cassette(args.output_dir)

    if args.study_type in ["st", "both"]:
        print("Downloading Sleep Telemetry (ST) files...")
        download_sleep_telemetry(args.output_dir)

    print("Download complete!")
