Working with the Neuroprobe Benchmark
=====================================

`Neuroprobe <https://neuroprobe.dev>`_ is a standardized benchmark for evaluating
neural decoding models on human intracranial EEG (iEEG) data. It defines 15
binary classification tasks spanning audio, language, and vision domains, all
derived from the `BrainTreebank <https://braintreebank.dev>`_ dataset — 40 hours
of sEEG recordings from 10 human subjects watching naturalistic movies.

For full details, see the `Neuroprobe paper <https://arxiv.org/abs/2509.21671>`_.

Preparing the data
------------------

Download and process the Neuroprobe data using the brainsets CLI::

    brainsets prepare neuroprobe_2025

.. note::

   Processing includes downloading raw BrainTreebank data and computing all
   benchmark splits. This may take several hours depending on your hardware and
   network connection. Use ``--cores <N>`` to parallelize.


Key concepts
------------

**Tasks.** Each of the 15 tasks is a binary classification problem where the
input is a 1-second window of neural data aligned to a word onset.
Available tasks:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Task
     - Domain
     - Description
   * - ``volume``
     - Auditory
     - Low vs. high average RMS audio volume
   * - ``pitch``
     - Auditory
     - Low vs. high average voice pitch
   * - ``delta_volume``
     - Auditory
     - Low vs. high volume change around word onset
   * - ``speech``
     - Language
     - Whether speech is present in the time interval
   * - ``onset``
     - Language
     - Whether a new sentence starts in the interval
   * - ``gpt2_surprisal``
     - Language
     - Low vs. high GPT-2 word surprisal
   * - ``word_length``
     - Language
     - Short vs. long word duration
   * - ``word_gap``
     - Language
     - Short vs. long inter-word gap
   * - ``word_index``
     - Language
     - First word in sentence vs. other
   * - ``word_head_pos``
     - Language
     - Left vs. right dependency tree head position
   * - ``word_part_speech``
     - Language
     - Verb vs. non-verb
   * - ``frame_brightness``
     - Visual
     - Low vs. high average frame brightness
   * - ``global_flow``
     - Visual
     - Low vs. high global optical flow
   * - ``local_flow``
     - Visual
     - Low vs. high local optical flow
   * - ``face_num``
     - Visual
     - No faces vs. one or more faces

**Regimes (split types).** Neuroprobe defines three evaluation regimes that test
different levels of generalization:

- **SS-SM** (*within-session*): Train and test on data from the same subject
  watching the same movie. Uses 2-fold cross-validation on contiguous blocks to
  prevent temporal autocorrelation leakage.
- **SS-DM** (*cross-session*): Train on one movie session and test on a
  different movie from the same subject.
- **DS-DM** (*cross-subject*): Train on a fixed anchor recording (Subject 2,
  Trial 4) and test on a different subject and movie. This is the most
  challenging regime.

The default leaderboard ranking uses the **cross-session (SS-DM)** split.

**Subset tiers.** Three subset sizes control the number of subject/trial pairs
and electrodes included:

- ``"full"``: All eligible subject/trial pairs.
- ``"lite"``: A curated subset of 6 subjects with 2 trials each (12 sessions),
  capped at 120 electrodes per subject. This is the standard benchmark
  configuration.
- ``"nano"``: A single trial per subject for rapid prototyping. Only supports
  the within-session regime.

**Metric.** The primary evaluation metric is AUROC (Area Under the ROC Curve).


Loading benchmark splits
------------------------

The :class:`~brainsets.datasets.Neuroprobe2025` class handles split resolution
automatically. Specify the benchmark parameters to get the correct train/test
partition:

.. code-block:: python

    from brainsets.datasets import Neuroprobe2025

    train_ds = Neuroprobe2025(
        subset_tier="lite",
        test_subject=1,
        test_session=1,
        split="train",
        task="speech",
        regime="SS-DM",
    )

    test_ds = Neuroprobe2025(
        subset_tier="lite",
        test_subject=1,
        test_session=1,
        split="test",
        task="speech",
        regime="SS-DM",
    )

The constructor resolves which recordings to load and which channel subset
and time intervals to use based on the requested split.

**Within-session (SS-SM)** uses 2-fold cross-validation. You can iterate over
folds:

.. code-block:: python

    from brainsets.datasets import Neuroprobe2025

    for fold in range(Neuroprobe2025.num_folds_for_regime("SS-SM")):
        train_ds = Neuroprobe2025(
            subset_tier="lite",
            test_subject=1,
            test_session=1,
            split="train",
            task="speech",
            regime="SS-SM",
            fold=fold,
        )
        test_ds = Neuroprobe2025(
            subset_tier="lite",
            test_subject=1,
            test_session=1,
            split="test",
            task="speech",
            regime="SS-SM",
            fold=fold,
        )


Accessing neural data and labels
---------------------------------

Each recording exposes sEEG data as a :obj:`~temporaldata.RegularTimeSeries`
sampled at 2048 Hz, along with split-specific sampling intervals and
channel inclusion masks:

.. code-block:: python

    intervals = train_ds.get_sampling_intervals()
    for recording_id, interval in intervals.items():
        rec = train_ds.get_recording(recording_id)
        print(rec.seeg_data.data.shape)
        print(interval.start[:5], interval.end[:5])
        print(interval.label[:5])

The ``interval.label`` array contains the binary labels for each trial window.

Channel metadata (electrode names, coordinates, inclusion masks) is available
via :meth:`~brainsets.datasets.Neuroprobe2025.get_channel_metadata`:

.. code-block:: python

    meta = train_ds.get_channel_metadata(recording_id)
    print(meta["names"])
    print(meta["coords"])           # LIP coordinates
    print(meta["included_mask"])    # benchmark electrode subset


Loading raw recordings
----------------------

If you want access to full continuous recordings without benchmark splits
(e.g. for pre-training), pass explicit ``recording_ids``:

.. code-block:: python

    from brainsets.datasets import Neuroprobe2025

    ds = Neuroprobe2025(recording_ids=["sub_1_trial001", "sub_2_trial004"])

In this mode, no split/task/regime resolution is performed; you get the
complete neural data for the requested sessions.


Running a complete benchmark evaluation
---------------------------------------

A typical benchmark loop iterates over all tasks, regimes, and subject/trial
pairs. Here is a minimal skeleton:

.. code-block:: python

    from brainsets.datasets import Neuroprobe2025
    from brainsets.datasets.Neuroprobe2025 import (
        VALID_TASKS,
        NEUROPROBE_LITE_SUBJECT_TRIALS,
    )

    regime = "SS-DM"
    results = {}

    for task in VALID_TASKS:
        for subject, session in sorted(NEUROPROBE_LITE_SUBJECT_TRIALS):
            for fold in range(Neuroprobe2025.num_folds_for_regime(regime)):
                train_ds = Neuroprobe2025(
                    subset_tier="lite",
                    test_subject=subject,
                    test_session=session,
                    split="train",
                    task=task,
                    regime=regime,
                    fold=fold,
                )
                test_ds = Neuroprobe2025(
                    subset_tier="lite",
                    test_subject=subject,
                    test_session=session,
                    split="test",
                    task=task,
                    regime=regime,
                    fold=fold,
                )

                # Train your model on train_ds, evaluate on test_ds
                # auroc = evaluate(model, test_ds)
                # results[(task, subject, session, fold)] = auroc

Report the mean AUROC across all subject/session pairs for each task, along
with the overall mean. Submit results to the
`Neuroprobe leaderboard <https://neuroprobe.dev>`_ following the instructions
in the `Neuroprobe code repository <https://github.com/azaho/neuroprobe>`_.


References
----------

.. code-block:: bibtex

    @article{zahorodnii2025neuroprobe,
        title={Neuroprobe: Evaluating Intracranial Brain Responses to Naturalistic Stimuli},
        author={Zahorodnii, Andrii and Wang, Christopher and Stankovits, Bennett
                and Moraitaki, Charikleia and Chau, Geeling and Barbu, Andrei
                and Katz, Boris and Fiete, Ila R},
        journal={arXiv preprint arXiv:2509.21671},
        year={2025}
    }
