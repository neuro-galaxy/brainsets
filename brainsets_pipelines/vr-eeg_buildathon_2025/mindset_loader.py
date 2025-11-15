"""
mindset_loader.py
Module for loading LABO CSV data and creating a Mindset object
Maryse Thomas & Benjamin Alsbury-Nealy, SilicoLabs 2025
"""

import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import mind_config as config


class Mindset:
    """
    Mindset object storing:
    - filename (str)
    - folder_name (str)
    - relational_data (DataFrame containing timing and index columns)
    - mindset_data (DataFrame containing behavioral data columns)
    - start_date (date)
    - start_time (str)
    - end_time (str)
    """

    def __init__(self, folder_name, filename, datatype, mindset_headers, df):
        self.folder_name = folder_name
        self.filename = filename
        self.datatype = datatype
        self.relational_data = df[list(config.RELATIONAL_COLUMNS.keys())].copy()
        self.mindset_data = df[mindset_headers].copy()
        self.start_time = self._infer_start_time()
        self.end_time = self._infer_end_time()
        self.timestamps = self.get_elapsed_timestamp(timestamp_col="FrameEndTimestamp")
        self.framestart_timestamps = self.get_elapsed_timestamp(timestamp_col="FrameStartTimestamp")
        self.fixed_timestamps = self.get_elapsed_timestamp(timestamp_col="FixedIntervalTimestamp")
        self.start_date = self._infer_date()

        self.elapsed_time = self.timestamps[-1] if len(self.timestamps) > 0 else None

        #Datatype-specific attributes
        if datatype in ["BodyTracking", "LeftHandTracking", "RightHandTracking"]:
            self.joints, self.joints_data = self.get_body_joints()

        elif datatype == "FaceTracking":
            self.blendshapes, self.blendshapes_data = self.get_face_blendshapes()

        elif datatype == "EyeTracking":
            self.gaze, self.gaze_data = self.get_gaze_data()

    def _infer_date(self):
        try:
            # Expecting something like: "Task_Participant_MM-DD-YYYY_HH-MM-SS"
            date_str = self.folder_name.split("_")[-2]
            return datetime.strptime(date_str, "%m-%d-%Y").date()
        except Exception:
            return None

    def _infer_start_time(self):
        starttimes = []
        for col in ["FrameStartTimestamp", "FrameEndTimestamp", "FixedIntervalTimestamp"]:
            if col in self.relational_data.columns:
                col_values = self.relational_data[col].dropna()
                if len(col_values) > 0:
                    starttimes.append(col_values.min())
        return min(starttimes) if starttimes else None

    def _infer_end_time(self):
        endtimes = []
        for col in ["FrameStartTimestamp", "FrameEndTimestamp", "FixedIntervalTimestamp"]:
            if col in self.relational_data.columns:
                col_values = self.relational_data[col].dropna()
                if len(col_values) > 0:
                    endtimes.append(col_values.max())
        return max(endtimes) if endtimes else None

    def get_elapsed_timestamp(self, timestamp_col="FrameEndTimestamp"):
        """
        Convert timestamp strings in the specified column to elapsed time
        since the recording starttime, in seconds.

        Returns:
            np.ndarray of floats: elapsed time in seconds
        """
        if timestamp_col not in self.relational_data.columns:
            raise ValueError(f"Column '{timestamp_col}' not found in relational_data.")

        if self.start_time is None:
            raise ValueError("start_time has not been inferred. Cannot compute elapsed timestamps.")

        time_format = "%H.%M.%S.%f"
        times = pd.to_datetime(self.relational_data[timestamp_col], format=time_format)
        start_time = pd.to_datetime(self.start_time, format=time_format)
        elapsed = (times - start_time).dt.total_seconds()

        return elapsed.to_numpy()

    def get_body_joints(self):
        """
        For BodyTracking and HandTracking datatypes, extract joint names and their corresponding data
        for column names of pattern 'Bone_JOINT_PARAMETER' or 'Hand_JOINT_PARAMETER'

        Returns:
            joints (list[str]): List of joint names
            joint_data (dict): Dictionary mapping each measurement type to a DataFrame of shape (n_rows, n_joints)
        """
        joints = []
        param_columns = {}

        for col in self.mindset_data.columns:
            parts = col.split("_")
            if len(parts) == 3:
                prefix, joint, param = parts[0], parts[1], parts[2]
            else:
                prefix, joint, param, suffix = parts[0], parts[1], parts[2], parts[3]
                param = param + '_' + suffix

            if joint not in joints:
                joints.append(joint)

            if param not in param_columns:
                param_columns[param] = []

            param_columns[param].append(col)

        # Create 2D array for each param
        joint_data = {}
        for param, cols in param_columns.items():
            joint_data[param] = self.mindset_data[cols].to_numpy()

        return joints, joint_data

    def get_face_blendshapes(self):
        """
        For FaceTracking datatype, extract blendshape names and their corresponding data
        for column names of pattern 'Face_BLENDSHAPE'

        Returns:
            blendshapes (list[str]): List of blendshape names
            blendshape_data (2D array of shape n_rows, n_blendshapes)
        """
        blendshapes = []

        for col in self.mindset_data.columns:
            parts = col.split("_")
            prefix, blendshape = parts[0], parts[1]
            blendshapes.append(blendshape)

        blendshape_data = self.mindset_data.to_numpy()

        return blendshapes, blendshape_data

    def get_gaze_data(self):
        """
        For Eyetracking datatype, extract eye names and their corresponding data
        for column names of pattern 'Eye_EYE_PARAMETER'

        Returns:
            gaze (list[str]): gaze data origin (left eye, right eye, center=average of both eyes)
            gaze_data (dict): Dictionary mapping each measurement type to a DataFrame of shape (n_rows, n_gaze_origins)
        """
        gaze = []
        param_columns = {}

        for col in self.mindset_data.columns:
            parts = col.split("_")
            if len(parts) == 3:
                prefix, eye, param = parts[0], parts[1], parts[2]

            elif len(parts) == 4:
                prefix, eye, param, suffix = parts[0], parts[1], parts[2], parts[3]
                param = param + '_' + suffix

            elif len(parts) == 5:
                prefix, eye, param, suffix1, suffix2 = parts[0], parts[1], parts[2], parts[3], parts[4]
                param = param + '_' + suffix1 + '_' + suffix2

            if eye not in gaze:
                gaze.append(eye)

            if param not in param_columns:
                param_columns[param] = []

            param_columns[param].append(col)

        # Create 2D array for each param
        gaze_data = {}
        for param, cols in param_columns.items():
            gaze_data[param] = self.mindset_data[cols].to_numpy()

        return gaze, gaze_data

    def __repr__(self):
        return (
            f"<Mindset("
            f"folder_name={self.folder_name}, "
            f"filename={self.filename}, "
            f"datatype={self.datatype}, "
            f"date={self.start_date}, "
            f"start_time={self.start_time}, "
            f"end_time={self.end_time}, "
            f"elapsed_time={self.timestamps[-1]} s, "
            f"relational_rows={len(self.relational_data)}, "
            f"mindset_rows={len(self.mindset_data)}"
            f")>"
        )


def load_csv(directory, filename=None, datatype=None):
    """
    Load CSV file from directory.
    If filename is None, use datatype to select default file.
    """
    if filename is None:
        if datatype is None:
            raise ValueError("No filename or datatype specified. Please provide one.")
        elif datatype not in config.VALID_DATATYPES:
            raise ValueError(f"Invalid datatype '{datatype}'. Valid options are: {', '.join(config.VALID_DATATYPES)}")
        else:
            filename = f"{datatype}.csv"
    path = os.path.join(directory, filename)
    folder_name = os.path.basename(os.path.dirname(path))

    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")

    # Read CSV with pandas
    df = pd.read_csv(path, header=0, index_col=False, on_bad_lines="skip")
    df.columns = df.columns.str.strip() # Strip column names of spaces

    # Drop columns with empty headers
    empty_cols = [col for col in df.columns if col == ""]
    if empty_cols:
        df = df.drop(columns=empty_cols)
        print(f"Dropped columns with empty headers: {empty_cols}")

    # Drop columns with headers that were not expected
    mindset_headers = config.DATA_COLUMN_KEY[datatype]
    valid_columns = list(config.RELATIONAL_COLUMNS.keys()) + list(mindset_headers)
    df = df[[col for col in valid_columns if col in df.columns]]

    return folder_name, filename, mindset_headers, df

def check_mindset_data(mindset, mindset_headers):
    """
    Inspect Mindset object for possible issues in relational_data and mindset_data.

    Reports:
        - Columns that are all NaNs
        - Columns that have no non-NaN values
        - Optional: mismatched row counts
        - Optional: duplicate columns
        - Optional: missing expected columns
    """
    issues = {}

    for df_name, df in [("relational_data", mindset.relational_data), ("mindset_data", mindset.mindset_data)]:
        df_issues = {}

        # Columns that are all NaNs
        all_nan_cols = df.columns[df.isna().all()].tolist()
        if all_nan_cols:
            df_issues["all_nan_columns"] = all_nan_cols

        # Columns that have no data (all empty strings or NaNs)
        empty_cols = df.columns[(df == "").all()].tolist()
        if empty_cols:
            df_issues["empty_columns"] = empty_cols

        # Duplicate columns
        dup_cols = df.columns[df.columns.duplicated()].tolist()
        if dup_cols:
            df_issues["duplicate_columns"] = dup_cols

        # Optional: missing expected columns (based on config)
        if df_name == "relational_data":
            expected_cols = list(config.RELATIONAL_COLUMNS.keys())
        else:
            expected_cols = mindset_headers
        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols:
            df_issues["missing_expected_columns"] = missing_cols

        # Check FrameNumber continuity
        if df_name == "relational_data" and "FrameNumber" in df.columns:
            if mindset.datatype not in config.IRREGULAR_DATATYPES:
                indices = df["FrameNumber"].to_numpy()
                if not np.all(np.diff(indices) == 1):
                    df_issues["non_contiguous_frame_number"] = True

        if df_issues:
            issues[df_name] = df_issues

    if not issues:
        print("No issues detected in Mindset data.")
    else:
        print("Issues found in Mindset data:")
        for df_name, df_issues in issues.items():
            print(f"  {df_name}:")
            for issue_type, cols in df_issues.items():
                print(f"    {issue_type}: {cols}")

    return issues

def load_mindset(directory, filename=None, datatype=None):
    folder_name, filename, mindset_headers, df = load_csv(directory, filename=filename, datatype=datatype)
    mindset = Mindset(folder_name, filename, datatype, mindset_headers, df)
    check_mindset_data(mindset, mindset_headers)
    return mindset

def main():
    parser = argparse.ArgumentParser(description="Load CSV by specifying either filename and datatype, or just datatype, and create Mindset object")
    parser.add_argument("--directory", type=str, required=True, help="Directory containing CSV files")
    parser.add_argument("--filename", type=str, default=None, help="Filename ending in .csv")
    parser.add_argument("--datatype", type=str, default=None, help="Datatype")
    parser.add_argument("--list-datatypes", action="store_true", help="Print a list of valid datatypes and exit")
    args = parser.parse_args()

    if args.list_datatypes:
        print(config.VALID_DATATYPES)
        return

    load_mindset(args.directory, filename=args.filename, datatype=args.datatype)

if __name__ == "__main__":
    main()