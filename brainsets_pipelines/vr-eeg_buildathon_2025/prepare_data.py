import argparse
import datetime
import logging
import h5py
import os
import warnings

import numpy as np
import pandas as pd

import mne
import antio

from mindset_loader import load_mindset

from temporaldata import (
    Data,
    RegularTimeSeries,
    IrregularTimeSeries,
    Interval,
    ArrayDict,
)

from brainsets.descriptions import (
    BrainsetDescription,
    SubjectDescription,
    SessionDescription,
    DeviceDescription,
)
from brainsets.taxonomy import RecordingTech, Species, Sex
from brainsets import serialize_fn_map


def main():
    # use argparse to get arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./raw")
    parser.add_argument("--output_dir", type=str, default="./processed")
    parser.add_argument("--subject_id", type=int, default=None)
    parser.add_argument("--sex", type=str, default=Sex.UNKNOWN)
    parser.add_argument("--handedness", type=str, default=None)
    parser.add_argument("--hand_opacity", type=int, default=0)
    parser.add_argument("--session_id", type=str, default=None)
    args = parser.parse_args()

    run_pipeline(args.input_dir, args.output_dir, args.subject_id, args.sex, args.session_id)

def run_pipeline(input_dir, output_dir, subject_id, sex="UNKNOWN", handedness=None, hand_opacity=0, session_id=None):

    os.makedirs(output_dir, exist_ok=True)
    print(f"Processing subject {subject_id} ({sex}) ({handedness}) (hand opacity:{hand_opacity}) ({session_id}...")

    # ------ Description ------

    brainset_description = BrainsetDescription(
        id="thomas_alsbury-nealy_hand-expressions_torchbrain-buildathon_2025",
        origin_version="0.0.0",
        derived_version="0.0.0",
        source="SilicoLabs",
        description=(
            "Bimanual gesture reproduction task in VR. Participants are asked to"
            "reproduce hand gestures shown to them as images in a virtual environment."
            "Gestures are either symmetric (both hands perform the same gesture) or"
            "asymmetric (each hand performs a different gesture). Data collected as"
            "part of the torch-brain build-a-thon at UPenn November 2025."
        )
    )

    # ------ Load EEG data ------

    # Load EEG data [ANT Neuro CNT/EDF format]
    # EEG data is taken from the CNT files and event markers from the EDF files
    eeg_dir = input_dir + '/eeg'

    # Load CNT files
    cnt_files = [f for f in os.listdir(eeg_dir) if f.endswith(".cnt")]

    if len(cnt_files) == 0:
        raise FileNotFoundError(f"No .cnt files found in '{input_dir}'.")
    elif len(cnt_files) > 1:
        raise RuntimeError(f"More than one .cnt file found in '{input_dir}': {cnt_files}")

    cnt_path = os.path.join(eeg_dir + '', cnt_files[0])
    raw_eeg = mne.io.read_raw_ant(cnt_path, preload=True)

    # Load EDF files
    edf_files = [f for f in os.listdir(eeg_dir) if f.endswith(".edf")]

    segment0_files = [f for f in edf_files if f.endswith("Segment_0.edf")]
    segment1_files = [f for f in edf_files if f.endswith("Segment_1.edf")]

    if len(edf_files) == 1:
        edf_path = os.path.join(eeg_dir, edf_files[0])
        raw_edf = mne.io.read_raw_edf(edf_path, preload=True)

        # Combine Annotations
        # It is normal to get the error: RuntimeWarning: Omitted 24 annotation(s) that were outside data range.
        # This refers to the impedence check annotations at the start of the recording
        raw_eeg.set_annotations(raw_edf.annotations)

    elif len(edf_files) == 0:
        raise FileNotFoundError(f"No .edf files found in '{input_dir}'.")

    # Case: exactly two files AND they are Segment_0 and Segment_1
    # This happens when the EEG recording is paused and resumed without closing the session
    # In this case, discard the first segment and only use the .edf file for the data
    elif len(edf_files) == 2 and len(segment0_files) == 1 and len(segment1_files) == 1:
        # use only the Segment_1 file
        edf_files = segment1_files
        edf_path = os.path.join(eeg_dir, edf_files[0])
        raw_eeg = mne.io.read_raw_edf(edf_path, preload=True)

    else:
        raise RuntimeError(
            f"Unexpected EDF file structure in '{input_dir}'. "
            f"Found EDF files: {edf_files}."
        )

    # ------ Load MIND data ------
    # MIND data (also called MINDsets) includes behavioural data types recorded with VR

    # Load Mindset data

    # Continuous data streams
    mindset_body = load_mindset(input_dir, filename=None, datatype="BodyTracking")
    mindset_eye = load_mindset(input_dir, filename=None, datatype="EyeTracking")
    mindset_face = load_mindset(input_dir, filename=None, datatype="FaceTracking")
    mindset_left_hand = load_mindset(input_dir, filename=None, datatype="LeftHandTracking")
    mindset_right_hand = load_mindset(input_dir, filename=None, datatype="RightHandTracking")

    # Variable update data for trial information
    mindset_variables = load_mindset(input_dir, filename=None, datatype="Variables")

    # Optionally inspect MINDset data
    print(mindset_body)

    # Use one of the continuous variables as the reference for elapsed time
    # DO NOT USE LEFT HAND OR RIGHT HAND (can drop frames if hands go out of view)
    # i.e. not eventpoints or variables data, which can be shorted than the full session
    mindset_ref = mindset_body

    # ------ Assign metadata ------

    # Convert string to Sex enum
    sex_enum = getattr(Sex, sex.upper(), Sex.UNKNOWN)

    subject = SubjectDescription(
        id=subject_id,
        species=Species.HOMO_SAPIENS,
        sex=sex_enum,  # Sex.FEMALE, Sex.OTHER, Sex.UNKNOWN
    )

    session = SessionDescription(
        id=session_id,
        recording_date=mindset_ref.start_date,
    )

    device = DeviceDescription(
        id="NA-245",
        #recording_tech='ANTNEURO_WAVEGUARD_NET', #TODO: Add VR hardware & ANTNeuro EEG (e.g. RecordingTech.UTAH_ARRAY_SPIKES)
        processing=None,
        chronic=False,
        start_date=mindset_ref.start_date,
    )

    # ------ Extract neural data ------

    # Apply notch filtering to eeg data
    # FYI: the reference electrode is subtracted from all channels at the time of recording
    raw_eeg_notch_filtered = raw_eeg.copy().notch_filter(freqs=[60.0])
    eeg_data = raw_eeg_notch_filtered.get_data()  # shape (n_channels, n_samples)

    # Store EEG data as regular time series
    eeg_info = ArrayDict(
        ids=np.array(raw_eeg.ch_names) #(nchannels,) array
    )

    eeg = RegularTimeSeries(
        domain="auto",
        sampling_rate=raw_eeg.info['sfreq'],
        signal=eeg_data.T, #Transpose to shape (n_samples, n_channels)
    )

    # ------ Align EEG and MIND data ------
    # Get start time of the MINDset data relative to EEG data and add offset to adjust timestamps
    # This assumes that the EEG recording started before the MINDset data recording

    # Experiment-specific variables
    gesture_list = ['ThumbsUp', 'Peace', 'Rockout', 'Point', 'Heart']
    target_epoch_list = ['Expressions-Epoch-Set', '5-Expressions-Epoch-Congruent-Incongruent']

    # Make dataframe to track experiment trials (epochs)
    # Trial order (after tutorial trials) is:
    # 1. Ready - a 3s timer counts down once participant has their hands on the ready position
    # 2. Gesture - an image of the target gesture(s) is shown and participants reproduce the gesture(s)
    # 3. Result - the result of the trial and points tally are shown for 2s
    epoch_df_relational = mindset_variables.relational_data.reset_index().groupby(['EpochNumber', 'EpochName'], as_index=False).first()
    epoch_df = mindset_variables.relational_data[['EpochNumber', 'EpochName']].drop_duplicates()

    #Drop first epoch, which does not have corresponding annotation in the EEG data
    epoch_df_relational = epoch_df_relational.drop(index=0).reset_index(drop=True)
    epoch_df = epoch_df.drop(index=0).reset_index(drop=True)

    # Initialize columns for MIND and EEG onset times
    epoch_df['MIND_onset'] = mindset_variables.timestamps[epoch_df_relational['index']]
    epoch_df['EEG_onset'] = np.nan

    # Find target epoch name corresponding to the current experiment
    epoch_names = epoch_df['EpochName'].unique()
    target_epoch_name = next(e for e in target_epoch_list if e in epoch_names)

    # LSL annotations recorded on the EEG timestamp
    # First column is the LSL stream value
    # Second column is the LSL stream name
    annotation_onsets = raw_eeg.annotations.onset
    annotation_descriptions = raw_eeg.annotations.description
    split_annotations = [row.split("###") for row in annotation_descriptions]
    annotation_values = [row[0] for row in split_annotations if len(row) > 0]
    annotation_df = pd.DataFrame({
        'EpochName': annotation_values,
        'EEG_onset': annotation_onsets
    })

    # CORRECTION FOR P004 and P005
    if subject_id in ['P004', 'P005']:
        annotation_df.loc[annotation_df['EpochName'] == '2-Tutorial-Expressions-Congruent-Incongr', 'EpochName'] = target_epoch_name
        annotation_df.loc[annotation_df['EpochName'] == 'Ready-Epoch', 'EpochName'] = '4-Ready-Epoch'
        annotation_df.loc[annotation_df['EpochName'] == 'Results-Epoch-Congruent-Incongruent', 'EpochName'] = '6-Results-Epoch-Congruent-Incongruent'
        annotation_df.loc[annotation_df['EpochName'] == 'Endcard-Epoch', 'EpochName'] = '7-Endcard-Epoch'
    else:
        # CORRECTION WHERE NAMES ARE TRUNCATED
        annotation_df.loc[annotation_df['EpochName'] == '5-Expressions-Epoch-Congruent-Incongruen', 'EpochName'] = target_epoch_name
        annotation_df.loc[annotation_df['EpochName'] == '2-Tutorial-Expressions-Congruent-Incongr', 'EpochName'] = '2-Tutorial-Expressions-Congruent-Incongruent'

    annotation_df = annotation_df[annotation_df['EpochName'].isin(epoch_names)].reset_index(drop=True)
    epoch_df['EEG_onset'] = annotation_df['EEG_onset']

    # Throw error if any trial times are missing (suggest misalignment)
    if epoch_df['EEG_onset'].isna().any() or epoch_df['MIND_onset'].isna().any():
        raise RuntimeError("Missing trial onset times in EEG or MIND data. Please check trial alignment.")

    # Compute offset between eeg and MIND timestamps
    epoch_df['Offset'] = epoch_df['EEG_onset'] - epoch_df['MIND_onset']
    eeg_mind_offset = epoch_df['Offset'].dropna().iloc[0] #offset value to add to all MIND timestamps

    # Add other trial labels to epoch dataframe
    epoch_df['LeftHand'] = ''
    epoch_df['RightHand'] = ''
    epoch_df['TrialType'] = ''
    epoch_df['Result'] = ''
    epoch_df['TrialDuration'] = np.nan

    # Loop through each row of epoch_df
    for i, row in epoch_df.iterrows():

        if row['EpochName'] == target_epoch_name:
            # Find the matching row(s) in relational_data
            mask_rel = (
                    (mindset_variables.relational_data['EpochNumber'] == row['EpochNumber']) &
                    (mindset_variables.relational_data['EpochName'] == row['EpochName'])
            )
            rel_idx = mindset_variables.relational_data.index[mask_rel]
            md_rows = mindset_variables.mindset_data.loc[rel_idx]

            # --- LeftHand and RightHand ---
            lh_row = md_rows[md_rows['Variable_Name'] == 'Expression-Target-Index-LH']
            rh_row = md_rows[md_rows['Variable_Name'] == 'Expression-Target-Index-RH']

            if not lh_row.empty and not rh_row.empty:
                lh_val = lh_row['Variable_SingleValue'].iloc[0]
                rh_val = rh_row['Variable_SingleValue'].iloc[0]
            else:
                # fallback to Expression-Target-Index
                both_row = md_rows[md_rows['Variable_Name'] == 'Expression-Target-Index']
                lh_val = rh_val = both_row['Variable_SingleValue'].iloc[0] if not both_row.empty else ''

            epoch_df.at[i, 'LeftHand'] = gesture_list[int(lh_val)]
            epoch_df.at[i, 'RightHand'] = gesture_list[int(rh_val)]

            if lh_val == rh_val:
                epoch_df.at[i, 'TrialType'] = 'symmetric'
            else:
                epoch_df.at[i, 'TrialType'] = 'asymmetric'

            # --- Result ---
            result_row = md_rows[md_rows['Variable_Name'] == 'Trial-Result']
            result_val = result_row['Variable_SingleValue'].iloc[0] if not result_row.empty else ''
            epoch_df.at[i, 'Result'] = result_val

    # ReactionTime
    # Subtract current MIND_onset from next one
    epoch_df['TrialDuration'] = epoch_df['MIND_onset'].shift(-1) - epoch_df['MIND_onset']

    # ------ Extract behavioral data ------

    # 27 body joints x 6 positions
    body_info = ArrayDict(
        ids=np.array(mindset_body.joints) #(j,) array
    )

    body = IrregularTimeSeries(
        domain="auto",
        timestamps=mindset_body.timestamps + eeg_mind_offset, #(n,) array
        position_x=mindset_body.joints_data['Position_X'], #(n,j) array
        position_y=mindset_body.joints_data['Position_Y'],
        position_z=mindset_body.joints_data['Position_Z'],
        rotation_x=mindset_body.joints_data['Rotation_X'],
        rotation_y=mindset_body.joints_data['Rotation_Y'],
        rotation_z=mindset_body.joints_data['Rotation_Z'],
        timekeys=['timestamps']
    )

    #27 hand joints x 6 positions
    left_hand_info = ArrayDict(
        ids=np.array(mindset_left_hand.joints) #(j,) array
    )

    left_hand = IrregularTimeSeries(
        domain="auto",
        timestamps=mindset_left_hand.timestamps + eeg_mind_offset, #(n,) array
        position_x=mindset_left_hand.joints_data['Position_X'], #(n,j) array
        position_y=mindset_left_hand.joints_data['Position_Y'],
        position_z=mindset_left_hand.joints_data['Position_Z'],
        rotation_x=mindset_left_hand.joints_data['Rotation_X'],
        rotation_y=mindset_left_hand.joints_data['Rotation_Y'],
        rotation_z=mindset_left_hand.joints_data['Rotation_Z'],
        timekeys=['timestamps']
    )

    #27 hand joints x 6 positions
    right_hand_info = ArrayDict(
        ids=np.array(mindset_right_hand.joints) #(j,) array
    )

    right_hand = IrregularTimeSeries(
        domain="auto",
        timestamps=mindset_right_hand.timestamps + eeg_mind_offset, #(n,) array
        position_x=mindset_right_hand.joints_data['Position_X'], #(n,j) array
        position_y=mindset_right_hand.joints_data['Position_Y'],
        position_z=mindset_right_hand.joints_data['Position_Z'],
        rotation_x=mindset_right_hand.joints_data['Rotation_X'],
        rotation_y=mindset_right_hand.joints_data['Rotation_Y'],
        rotation_z=mindset_right_hand.joints_data['Rotation_Z'],
        timekeys=['timestamps']
    )

    #72 blendshapes x 1 value
    face_info = ArrayDict(
        ids=np.array(mindset_face.blendshapes) #(j,) array
    )

    face = IrregularTimeSeries(
        domain="auto",
        timestamps=mindset_face.timestamps + eeg_mind_offset, #(n,) array
        value=mindset_face.blendshapes_data, #(n,j) array
        timekeys=['timestamps']
    )

    eye_info = ArrayDict(
        ids=np.array(mindset_eye.gaze)  # (j,) array
    )

    eye = IrregularTimeSeries(
        domain="auto",
        timestamps=mindset_eye.timestamps + eeg_mind_offset,  # (n,) array
        origin_x=mindset_eye.gaze_data['Origin_X'],  # (n,j) array
        origin_y=mindset_eye.gaze_data['Origin_Y'],
        origin_z=mindset_eye.gaze_data['Origin_Z'],
        direction_x=mindset_eye.gaze_data['Direction_X'],
        direction_y=mindset_eye.gaze_data['Direction_Y'],
        direction_z=mindset_eye.gaze_data['Direction_Z'],
        hit_point_x=mindset_eye.gaze_data['Hit_Point_X'],
        hit_point_y=mindset_eye.gaze_data['Hit_Point_Y'],
        hit_point_z=mindset_eye.gaze_data['Hit_Point_Z'],
        hit_object_position_x=mindset_eye.gaze_data['Hit_ObjectPosition_X'],
        hit_object_position_y=mindset_eye.gaze_data['Hit_ObjectPosition_Y'],
        hit_object_position_z=mindset_eye.gaze_data['Hit_ObjectPosition_Z'],
        hit_distance=mindset_eye.gaze_data['Hit_Distance'],
        timekeys=['timestamps']
    )

    # ------ Extract trial information ------

    # Extract trial information from variables data
    target_epoch_df = epoch_df[epoch_df['EpochName'] == target_epoch_name].reset_index(drop=True)
    trial_start = target_epoch_df['EEG_onset'].values.tolist()
    trial_end = trial_start[1:]
    trial_end = np.append(trial_end, trial_end[-1] + target_epoch_df['TrialDuration'].iloc[-1])

    trials = Interval(
        start=np.array(trial_start),
        end=np.array(trial_end),
        timestamp=np.mean([trial_start, trial_end], axis=0),
        trial_type=target_epoch_df['TrialType'].to_numpy(), #whether the trial is symmetric or asymmetric
        gesture_left=target_epoch_df['LeftHand'].to_numpy(), #name of the gesture reproduced with the left hand
        gesture_right=target_epoch_df['RightHand'].to_numpy(), #name of the gesture reproduced with the right hand
        result=target_epoch_df['Result'].to_numpy(), #Correct (correct gestures made with both hands) or Timeout (correct gestures not made within 3 seconds)
        reaction_time=target_epoch_df['TrialDuration'].to_numpy(), #time in s to produce the correct gestures (equal to trial length),
        trial_number=np.arange(1, len(trial_start) + 1), #trial number starting at 1
        timekeys=['start','end','timestamp']
    )

    # ------ Create data object ------

    data = Data(
        # metadata
        brainset=brainset_description,
        subject=subject,
        session=session,
        device=device,

        #subject-level data
        handedness=handedness,
        hand_opactiy=hand_opacity,

        # neural data
        eeg_info=eeg_info,
        eeg=eeg,

        # behavioral data
        body_info=body_info,
        body=body,
        left_hand_info=left_hand_info,
        left_hand=left_hand,
        right_hand_info=right_hand_info,
        right_hand=right_hand,
        face_info=face_info,
        face=face,
        eye_info=eye_info,
        eye=eye,

        # trial data
        trials=trials,
        domain="auto",
    )

    # ------ Split the data  ------

    # Split trials into train/valid/test sets
    successful_trials = trials.select_by_mask(trials.result == 'Correct')
    train_trials, valid_trials, test_trials = successful_trials.split(
        [0.7, 0.1, 0.2],  # proportions for train/valid/test
        shuffle=True,  # randomly shuffle trials
        random_seed=42  # for reproducibility
    )

    # Set domains based on trial splits
    data.set_train_domain(train_trials)
    data.set_valid_domain(valid_trials)
    data.set_test_domain(test_trials)

    # ------ Save the data  ------

    # save data to disk
    path = os.path.join(output_dir, f"{session_id}.h5")

    with h5py.File(path, "w") as file:
        data.to_hdf5(file, serialize_fn_map=serialize_fn_map)

    # save epoch_df results to csv
    csv_path = os.path.join(output_dir, f"{session_id}.csv")
    target_epoch_df['Handedness'] = handedness
    target_epoch_df['Hand_Opacity'] = hand_opacity
    target_epoch_df['Subject_ID'] = subject_id
    target_epoch_df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    main()