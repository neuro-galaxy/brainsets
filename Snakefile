from pathlib import Path

# You can override the default TMP_DIR and COMPRESSED_DIR by passing them as arguments to snakemake.
# snakemake --config TMP_DIR=/path/to/new/tmp/dir COMPRESSED_DIR=/path/to/new/compressed/dir
TMP_DIR = config.get("TMP_DIR", os.environ["SLURM_TMPDIR"])
PERM_DIR = config.get("PERM_DIR", "/network/projects/neuro-galaxy/data")
RAW_DIR = str(Path(PERM_DIR) / "raw")
COMPRESSED_DIR = str(Path(PERM_DIR) / "compressed")

######################################################
# O'Doherty & Sabes (2017) 
######################################################

DATASET = "odoherty_sabes"
# It's much preferable with shell-based tools to stick to no-space directory names,
# but I didn't change it quite yet because I wanted to keep the github diff readable.
FOLDER_NAME = "Philip, Makin"

BROADBAND_DATASETS = [
    "indy_20160622_01", "indy_20160624_03", "indy_20160627_01", "indy_20160630_01", "indy_20160915_01",
    "indy_20160916_01", "indy_20160921_01", "indy_20160927_04", "indy_20160927_06", "indy_20160930_02",
    "indy_20160930_05", "indy_20161005_06", "indy_20161006_02", "indy_20161007_02", "indy_20161011_03",
    "indy_20161013_03", "indy_20161014_04", "indy_20161017_02", "indy_20161024_03", "indy_20161025_04",
    "indy_20161026_03", "indy_20161027_03", "indy_20161206_02", "indy_20161207_02", "indy_20161212_02",
    "indy_20161220_02", "indy_20170123_02", "indy_20170124_01", "indy_20170127_03", "indy_20170131_02"
]

ZENODO_IDS = [
    1488440, 1486147, 1484824, 1473703, 1467953, 1467050, 1451793, 1433942, 1432818, 1421880, 1421310, 1419774,
    1419172, 1413592, 1412635, 1412094, 1411978, 1411882, 1411474, 1410423, 1321264, 1321256, 1303720, 1302866,
    1302832, 1301045, 1167965, 1163026, 1161225, 854733
]

rule compress_data:
    input:
        description = f"{TMP_DIR}/processed/{DATASET}/description.yaml"
    output:
        train_tar = f"{COMPRESSED_DIR}/{DATASET}/train.tar.lz4",
        test_tar = f"{COMPRESSED_DIR}/{DATASET}/test.tar.lz4",
        valid_tar = f"{COMPRESSED_DIR}/{DATASET}/valid.tar.lz4",
        desc_out = f"{COMPRESSED_DIR}/{DATASET}/description.yaml"
    shell:
        f"""
        mkdir -p {COMPRESSED_DIR}/{DATASET}
        mkdir -p {TMP_DIR}/compressed
        for split in train valid test; do
            # Single lz4 archive.
            cd {TMP_DIR}/processed/{DATASET}/$split && \
                tar -cf - . | lz4 -1 > {COMPRESSED_DIR}/{DATASET}/$split.tar.lz4
            cd - > /dev/null
            # Multiple shards for webdataset usage.
            python split_and_tar.py --input_dir "{TMP_DIR}/processed/{DATASET}/$split" --output_dir "{COMPRESSED_DIR}/{DATASET}" --prefix $split
        done
        cp {{input.description}} {COMPRESSED_DIR}/{DATASET}/description.yaml
        """

rule prepare_data:
    input:
        py_script = f"data/{FOLDER_NAME}/prepare_data.py",
        mat_files = expand(f"{RAW_DIR}/{DATASET}/{{dataset}}.mat", dataset=BROADBAND_DATASETS),
        nwb_files = expand(f"{RAW_DIR}/{DATASET}/broadband/{{dataset}}.nwb", dataset=BROADBAND_DATASETS)
    output:
        description = f"{TMP_DIR}/processed/{DATASET}/description.yaml"
    shell:
        f"""
        mkdir -p {TMP_DIR}/raw/{DATASET}
        cp -r {RAW_DIR}/{DATASET} {TMP_DIR}/raw/
        mkdir -p {TMP_DIR}/processed/{DATASET}
        cd "data/{FOLDER_NAME}" && \
            python prepare_data.py --input_dir {TMP_DIR}/raw/{DATASET} --output_dir {TMP_DIR}/processed/{DATASET}
        """

rule download_primary_dataset:
    output:
        mat_files = expand(f"{RAW_DIR}/{DATASET}/{{dataset}}.mat", dataset=BROADBAND_DATASETS)
    shell:
        f"""
        mkdir -p {TMP_DIR}/raw/{DATASET}
        zenodo_get 583331 -o {RAW_DIR}/{DATASET}
        """

rule download_broadband_dataset:
    output:
        nwb_file = f"{RAW_DIR}/{DATASET}/broadband/{{dataset_name}}.nwb"
    params:
        zenodo_id = lambda wildcards: ZENODO_IDS[BROADBAND_DATASETS.index(wildcards.dataset_name)],
        parent_dir = f"{RAW_DIR}/{DATASET}/broadband/"
    shell:
        """
        zenodo_get {params.zenodo_id} -o {params.parent_dir}
        """

