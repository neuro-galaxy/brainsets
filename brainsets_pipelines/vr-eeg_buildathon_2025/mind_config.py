"""
mind_config.py
This configuration file specifies the data types and column names of MIND data outputs
Maryse Thomas & Benjamin Alsbury-Nealy, SilicoLabs 2025
"""

# ------------ GENERAL ------------

# LABO data types corresponding to CSV filenames
BODY_FILE = "BodyTracking"
EYE_FILE = "EyeTracking"
FACE_FILE = "FaceTracking"
LEFTHAND_FILE = "LeftHandTracking"
RIGHTHAND_FILE = "RightHandTracking"
EVENTPOINTS_FILE = "EventPoints"
VARIABLES_FILE = "Variables"
INTERACTIVES_FILE = "InteractivesData"
EXPRESSIONS_FILE = "Expressions"

# Store datatypes in a list to reference for validation
VALID_DATATYPES = [
    BODY_FILE,
    EYE_FILE,
    FACE_FILE,
    LEFTHAND_FILE,
    RIGHTHAND_FILE,
    EVENTPOINTS_FILE,
    VARIABLES_FILE,
    INTERACTIVES_FILE,
    EXPRESSIONS_FILE,
]

# Regular datatypes are continuously recorded, meaning data exists for every timestamp
REGULAR_DATATYPES = [
    BODY_FILE,
    EYE_FILE,
    FACE_FILE,
    LEFTHAND_FILE,
    RIGHTHAND_FILE,
    INTERACTIVES_FILE
]

# Irregular datatypes are not continuously recorded, meaning data exists only at certain timestamps
IRREGULAR_DATATYPES = [
    EVENTPOINTS_FILE,
    VARIABLES_FILE,
    EXPRESSIONS_FILE
]

# Columns that are common to all CSVs and their expected data types
RELATIONAL_COLUMNS = {
    "FrameStartTimestamp": "time",
    "FrameEndTimestamp": "time",
    "FixedIntervalTimestamp": "time",
    "FrameNumber": int,
    "FixedIntervalNumber": int,
    "EpochNumber": int,
    "EpochName": str,
    "EventNumber": int,
}

# Define the JOINTSUFFIX for body and hand column headers
JOINTSUFFIX = [
    "Position_X",
    "Position_Y",
    "Position_Z",
    "Rotation_X",
    "Rotation_Y",
    "Rotation_Z",
    "Touching",
]

# ------------ BODY TRACKING COLUMNS ------------
# Each body JOINT has columns of format: Bone_JOINT_JOINTSUFFIX

BODY_JOINTS = [
    "Start",
    "Root",
    "Hips",
    "SpineLower",
    "SpineMiddle",
    "SpineUpper",
    "Chest",
    "Neck",
    "Head",
    "LeftShoulder",
    "LeftScapula",
    "LeftArmUpper",
    "LeftArmLower",
    "LeftHandWristTwist",
    "RightShoulder",
    "RightScapula",
    "RightArmUpper",
    "RightArmLower",
    "RightHandWristTwist",
    "LeftUpperLeg",
    "LeftLowerLeg",
    "LeftFootAnkle",
    "LeftFootBall",
    "RightUpperLeg",
    "RightLowerLeg",
    "RightFootAnkle",
    "RightFootBall",
]

# Generate BODY_COLUMNS
BODY_COLUMNS = [f"Bone_{loc}_{suffix}" for loc in BODY_JOINTS for suffix in JOINTSUFFIX]

# ------------ HAND TRACKING COLUMNS ------------
# Each hand JOINT has columns of format: [Left/Right]Hand_JOINT_JOINTSUFFIX

HAND_JOINTS = [
    "ForearmStub",
    "Wrist",
    "Palm",
    "ThumbMetacarpal",
    "ThumbProximal",
    "ThumbDistal",
    "ThumbTip",
    "IndexMetacarpal",
    "IndexProximal",
    "IndexIntermediate",
    "IndexDistal",
    "IndexTip",
    "MiddleMetacarpal",
    "MiddleProximal",
    "MiddleIntermediate",
    "MiddleDistal",
    "MiddleTip",
    "RingMetacarpal",
    "RingProximal",
    "RingIntermediate",
    "RingDistal",
    "RingTip",
    "LittleMetacarpal",
    "LittleProximal",
    "LittleIntermediate",
    "LittleDistal",
    "LittleTip",
]

# Generate LEFTHAND and RIGHTHAND columns
LEFTHAND_COLUMNS = [f"LeftHand_{loc}_{suffix}" for loc in HAND_JOINTS for suffix in JOINTSUFFIX]
RIGHTHAND_COLUMNS = [f"RightHand_{loc}_{suffix}" for loc in HAND_JOINTS for suffix in JOINTSUFFIX]


# ------------ EYE TRACKING COLUMNS ------------
# Each eye has columns of format: Eye_NAME_EYESUFFIX

EYE_NAMES = ['Center','Left','Right']

EYE_SUFFIXES = ['Origin_X',
                'Origin_Y',
                'Origin_Z',
                'Direction_X',
                'Direction_Y',
                'Direction_Z',
                'Hit_Texture',
                'Hit_Point_X',
                'Hit_Point_Y',
                'Hit_Point_Z',
                'Hit_ObjectPosition_X',
                'Hit_ObjectPosition_Y',
                'Hit_ObjectPosition_Z',
                'Hit_Distance',
                'Hit_Tag',
                'Hit_Object',
                'Hit_InteractiveName',
                'Hit_InteractiveID']

# Generate EYE_COLUMNS
EYE_COLUMNS = [f"Eye_{loc}_{suffix}" for loc in EYE_NAMES for suffix in EYE_SUFFIXES]

# ------------ FACE TRACKING COLUMNS ------------
# Each face blendshape has its own column of format: Face_BLENDSHAPE

FACE_COLUMNS = [
    "Face_Invalid",
    "Face_BrowLowererL",
    "Face_BrowLowererR",
    "Face_CheekPuffL",
    "Face_CheekPuffR",
    "Face_CheekRaiserL",
    "Face_CheekRaiserR",
    "Face_CheekSuckL",
    "Face_CheekSuckR",
    "Face_ChinRaiserB",
    "Face_ChinRaiserT",
    "Face_DimplerL",
    "Face_DimplerR",
    "Face_EyesClosedL",
    "Face_EyesClosedR",
    "Face_EyesLookDownL",
    "Face_EyesLookDownR",
    "Face_EyesLookLeftL",
    "Face_EyesLookLeftR",
    "Face_EyesLookRightL",
    "Face_EyesLookRightR",
    "Face_EyesLookUpL",
    "Face_EyesLookUpR",
    "Face_InnerBrowRaiserL",
    "Face_InnerBrowRaiserR",
    "Face_JawDrop",
    "Face_JawSidewaysLeft",
    "Face_JawSidewaysRight",
    "Face_JawThrust",
    "Face_LidTightenerL",
    "Face_LidTightenerR",
    "Face_LipCornerDepressorL",
    "Face_LipCornerDepressorR",
    "Face_LipCornerPullerL",
    "Face_LipCornerPullerR",
    "Face_LipFunnelerLB",
    "Face_LipFunnelerLT",
    "Face_LipFunnelerRB",
    "Face_LipFunnelerRT",
    "Face_LipPressorL",
    "Face_LipPressorR",
    "Face_LipPuckerL",
    "Face_LipPuckerR",
    "Face_LipStretcherL",
    "Face_LipStretcherR",
    "Face_LipSuckLB",
    "Face_LipSuckLT",
    "Face_LipSuckRB",
    "Face_LipSuckRT",
    "Face_LipTightenerL",
    "Face_LipTightenerR",
    "Face_LipsToward",
    "Face_LowerLipDepressorL",
    "Face_LowerLipDepressorR",
    "Face_MouthLeft",
    "Face_MouthRight",
    "Face_NoseWrinklerL",
    "Face_NoseWrinklerR",
    "Face_OuterBrowRaiserL",
    "Face_OuterBrowRaiserR",
    "Face_UpperLidRaiserL",
    "Face_UpperLidRaiserR",
    "Face_UpperLipRaiserL",
    "Face_UpperLipRaiserR",
    "Face_TongueTipInterdental",
    "Face_TongueTipAlveolar",
    "Face_TongueFrontDorsalPalate",
    "Face_TongueMidDorsalPalate",
    "Face_TongueBackDorsalVelar",
    "Face_TongueOut",
    "Face_TongueRetreat",
    "Face_Max",
]

# ------------ EVENTPOINTS & VARIABLES COLUMNS ------------

EVENTPOINTS_COLUMNS = [
    "Event_Type",
    "Event_Description",
    "Event_Note",
    "Event_Source",
    "Event_Source_Parent",
    "Event_Trigger",
    "Action_Number",
    "Action_Type",
]

VARIABLES_COLUMNS = [
    "Variable_Name",
    "Variable_DataType",
    "Variable_Scope",
    "Variable_SingleValue",
    #"Variable_ListValues",
    #"Variable_ModifyingVariable",
    #"Variable_Modifier",
    #"Variable_UpdateValue",
    #"Variable_Index",
]

# ------------ INTERACTIVES COLUMNS ------------

INTERACTIVES_COLUMNS = [
    "Interactive_Name",
    "Interactive_Description",
    "Interactive_InstanceID",
    "Interactive_Position_X",
    "Interactive_Position_Y",
    "Interactive_Position_Z",
    "Interactive_Rotation_X",
    "Interactive_Rotation_Y",
    "Interactive_Rotation_Z",
    "Interactive_Scale_X",
    "Interactive_Scale_Y",
    "Interactive_Scale_Z",
    "Interactive_Interactors_Hovering",
    "Interactive_Interactors_Selecting",
    "Interactive_Physics_Type",
    "Interactive_Physics_Object",
    "Interactive_Physics_IsKinematic",
    "Interactive_Physics_Gravity",
    "Interactive_Physics_Mass",
    "Interactive_Physics_Drag",
    "Interactive_Physics_AngularDrag",
    "Interactive_Physics_PositionLock_X",
    "Interactive_Physics_PositionLock_Y",
    "Interactive_Physics_PositionLock_Z",
    "Interactive_Physics_RotationLock_X",
    "Interactive_Physics_RotationLock_Y",
    "Interactive_Physics_RotationLock_Z",
    "Interactive_PhysicMaterial_Enabled",
    "Interactive_PhysicMaterial_DynamicFriction",
    "Interactive_PhysicMaterial_StaticFriction",
    "Interactive_PhysicMaterial_FrictionCombined",
    "Interactive_PhysicMaterial_Bounce",
    "Interactive_PhysicMaterial_BounceCombined",
]

# ------------ EXPRESSIONS COLUMNS ------------

EXPRESSIONS_COLUMNS = [
    "Expression_Name",
    "Expression_State",
    "Expression_Threshold",
    "Expression_Left_Similarity",
    "Expression_Right_Similarity",
    "Expression_Left_Met",
    "Expression_Right_Met",
    "Expression_Both_Met",
]

# ------------ MAPPING ------------
# Map each data type to its corresponding columns

DATA_COLUMN_KEY = {
    BODY_FILE: BODY_COLUMNS,
    EYE_FILE: EYE_COLUMNS,
    FACE_FILE: FACE_COLUMNS,
    LEFTHAND_FILE: LEFTHAND_COLUMNS,
    RIGHTHAND_FILE: RIGHTHAND_COLUMNS,
    EVENTPOINTS_FILE: EVENTPOINTS_COLUMNS,
    VARIABLES_FILE: VARIABLES_COLUMNS,
    INTERACTIVES_FILE: INTERACTIVES_COLUMNS,
    EXPRESSIONS_FILE: EXPRESSIONS_COLUMNS
}