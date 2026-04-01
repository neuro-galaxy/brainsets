# Dataset formats
All data files are stored in MATLAB .mat format (version 7). The files may be read with the MATLAB built-in `load` function.

## Navigation data files
All .mat files within a folder named `data/navigation/<session_type>` contain data acquired during awake navigation sessions of the type `<session_type>`. Each file contains a single structure named `Dsession`, containing all data for the given animal ID and session type.

### Data loading:
The example below loads data from session type 'OF' (open field), rat #26034, recording #2:

```matlab
load("data\navigation\of\26034_2.mat", 'Dsession')
```

### Dataset organization:
The contents of the `Dsession` structure falls into the following categories:

- Neural and behavior timeseries data
- Single-neuron analysis results
- Neural population analysis results
- Assorted metadata describing behavioral, electrophysiological and analysis parameters

Some important considerations:

- Sometimes, more than one session of a particular type were performed within the same contiguous recording period; in such cases the data from these multiple sessions was concatenated.
- All timeseries data has been 'speed-filtered' to discard samples when the animal's locomotion speed was below 5 cm/s.
- All timestamps indicate the number of seconds elapsed since the beginning of data acquisition. The first timestamp within a dataset file will typically have a positive offset; this is because the behavior session usually commenced some time after starting data acquisition.

Below is a table of contents of the root `Dsession` structure:

```
Dsession = 

  struct with fields:

                   t: [500000Ã—1 double]    times (s) of head tracking data samples and binned analyses
                   x: [500000Ã—1 single]    head x-position relative to arena center (m)
                   y: [500000Ã—1 single]    head y-position relative to arena center (m)
                   z: [500000Ã—1 single]    head z-position relative to floor (m)
                  hd: [500000Ã—1 single]    2D head direction (azimuth)
               speed: [500000Ã—1 single]    horizontal head speed (m/s)
                  id: [500000Ã—1 single]    decoded internal direction (based on LMT model) 
               theta: [500000Ã—1 single]    theta phase (radians)
                  dt: 0.0100               sampling time interval (s)
                  nt: 500000               total number of samples
                  gv: [1Ã—1 struct]         gridded spatial coordinates (bin centers)
                 gve: [1Ã—1 struct]         gridded spatial coordinates (bin edges)
                 lmt: [1Ã—1 struct]         fitted LMT model results
            sessions: [1Ã—1 Session]        list of individual sessions (time epochs) included
         thetaChunks: [1Ã—1 struct]         timeseries data binned at theta cycle times
               units: [1Ã—1 struct]         single-unit data
       unitAcorrClus: [1Ã—1 struct]         grid module clustering analysis
    probeChannelMaps: [1Ã—2 struct]         locations of electrophysiological recording channels on probe
```

### Detailed field descriptions

#### gv, gve
These fields contain standardized spatial coordinate grids used in tuning maps and other spatial analyses. Each contains a structure, with field names indicating the spatial variables that the grids were used in conjunction with.

```
Dsession.gv/gve = 

  struct with fields:

    theta: [60Ã—1 single]    theta phase        (angular coordniates)
       hd: [60Ã—1 single]    head direction     (angular coordinates)
       id: [60Ã—1 single]    internal direction (angular coordinates)
      pos: [96Ã—1 single]    2D position,       (1D coordinates, applied to both x- and y-axis)
```

#### lmt
The `Dsession.lmt` field contains results from fitting the LMT model to neural population data from the anatomical locations available in the currrent animal. The currently loaded example animal has an implant in both MEC and HC, which yields three different populations for fitting the LMT model (MEC only, HC only, MEC+HC combined). In animals with an implant in only one location, the `Dsession.lmt` struct will contain only a single field for that location.

```
Dsession.lmt =

  struct with fields:

       mec: [1Ã—1 struct]    MEC units only
        hc: [1Ã—1 struct]    HC units only
    mec_hc: [1Ã—1 struct]    Both MEC and HC units 
```

Each of the above fields has the following contents, with each field containing the fitted results for one LMT component of the complete model:

```
Dsession.lmt.mec = 

  struct with fields:
  
    theta: [1Ã—1 struct]    theta phase        (fixed)  1D circular variable
       hd: [1Ã—1 struct]    head direction     (fixed)  1D circular variable
       id: [1Ã—1 struct]    internal direction (latent) 1D circular variable
      pos: [1Ã—1 struct]    2D position        (latent) 2D linear variable
```

Each of the above fields contains a LMT results struct with the following contents:

```
Dsession.lmt.mec.pos = 

  struct with fields:

         XA: [500000Ã—2 single]    input variable values                (time bins * dimensions)
          F: [900Ã—1410 single]    tuning                               (input variable bins * units)
         gv: {1Ã—2 cell}           original binning vectors
        ggi: [9216Ã—2 single]      mesh-grid for finer interpolation    (interpolation bins * dimensions)
    unitIds: [1410Ã—1 string]      list of IDs of all units
        gsz: [30 30]              size of original binning grid
       gszi: [96 96]              size of fine-grained reinterpolated grid
    hparams: [1Ã—1 struct]         model hyperparameters
```

#### thetaChunks
`Dsession.thetaChunks` is a structure containing timeseries data binned at individual theta cycles. Each field contains an array where the first dimension corresponds to the theta cycle number.

```
Dsession.thetaChunks = 

  struct with fields:

          iStart: [39681Ã—1 double]     index of first time bin after the cycle start time
    iStartInterp: [39681Ã—1 double]     interpolated time-bin index at exact cycle start time
              id: [39681Ã—1 double]     decoded internal direction value
               L: [39681Ã—30 double]    decoded internal direction log-likelikehood distribution (cycles * direction bins)
               P: [39681Ã—30 double]    decoded internal direction probability distribution (cycles * direction bins)
          tStart: [39681Ã—1 double]     cycle start time
```

#### units
`Dsession.units` is contains all data related to single units. Units are first grouped by their anatomical location:

```
Dsession.units = 

  struct with fields:

    mec: [860Ã—1 struct]
     hc: [446Ã—1 struct]
```

Each of the above fields contains an array of structs, with each element describing one unit. If the animal was implanted in only one location, one of the above fields will be empty.

Units are identified by a unique two-part string. The first part is the probe number, and the second part is the unit ID assigned by kilosort. For example, the ID "2_1039" refers to a unit recorded on probe #2, with a kilosort ID of 1039. Whenever single units are referred to by other parts of the dataset, this identifer is used.

```
Dsession.units.mec(1) = 

  struct with fields:

            id: "1_0055"          unique identifier
      ks2Label: 'good'            quality label appended by kilosort
      location: "mec"             anatomical location
      meanRate: 1.5682            mean firing rate (spikes/s)
       nSpikes: 5175              number of spikes
       probeId: 1                 probe number
         shank: 3                 probe shank number
      shankPos: 1.0003e+03        vertical position of unit on shank (micrometers; zero indicates shank tip)
     spikeInds: [5175Ã—1 uint32]   time-bin index for each spike
    spikeTimes: [5175Ã—1 double]   time of each spike (s)
      wvExtent: 12.9473           average spatial extent of action-potential waveform
          lmtF: [1Ã—1 struct]      LMT tuning
           rmc: [1Ã—1 struct]      coarse-grained firing-rate maps
           rmf: [1Ã—1 struct]      fine-grained firing-rate maps
          smdl: [1Ã—1 struct]      fitted GLM "shift model" tuning
            wv: [61Ã—383 single]   average action potential waveform (samples x channels)
      cellType: "id"              functional class assigned to this unit
      acorrClu: "nongrid"         name of parent grid-module cluster ("non-grid", "grid 1", "grid 2" etc.)
        isGrid: 0                 indicates whether cell is an identified grid cell (1) or not (0)
    acorrCluId: 0                 numerical ID of the parent grid-module cluster
``` 

#### unitAcorrClus
`Dsession.unitAcorrClus` contains results of the grid-module clustering procedure.

```
Dsession.unitAcorrClus = 

  struct with fields:

           grid: [1Ã—1 struct]    data for grid clusters
        nongrid: [6Ã—1 struct]    data for non-grid clusters
    sessionType: 'of'            string code specifying session type

```

Each struct element in the `grid` and `nongrid` fields corresponds to a single cluster of units identified by the clustering procedure. These structs contain various data about the cluster:

```
Dsession.unitAcorrClus.grid(1) =

  struct with fields:

               nUnits: 81                   number of units within the cluster
              unitIds: [81Ã—1 string]        identifers of the constituent units
             unitInds: [860Ã—1 logical]      indices of the contituent units
                cluId: 7                    numeric identifier for the cluster
           maskCoarse: [47Ã—47 logical]      mask for coarse-grained autocorrelogram
             maskFine: [191Ã—191 logical]    mask for fine-grained autocorrelogram
      medianGridScore: 1.0137               median gridness score of units in this cluster
    medianGridSpacing: 0.6047               median grid spacing of units in this cluster
         acorrsCoarse: [81Ã—2209 single]     array of coarse-grained correlograms (units * spatial bins)
    medianAcorrCoarse: [47Ã—47 single]       median coarse-grained autocorrelogram
      medianAcorrFine: [191Ã—191 single]     median fine-grained autocorrelogram
            gridScore: 1.2064               grid score of median autocorrelogram
            gridStats: [1Ã—1 struct]         other grid statistics of median autocorrelogram
    medianAcorrFineSm: 29                   width of smoothing applied to median autocorrelogram
      gridSpacingBins: 24.2074              grid spacing of median autocorrelogram (bins)
    gridSpacingMeters: 0.6052               grid spacing of median autocorrelogram (meters)
              selfSim: [81Ã—1 single]        similarity of each unit's autocorrelogram to the cluster average
        medianSelfSim: 0.7966               median of single-unit 'selfSim' values
               isGrid: 1                    indicates whether cluster is classified as 'grid' (1) or 'non-grid' (0)
                 name: "grid_1"             name string of the cluster
```

#### probeChannelMaps
`Dsession.probeChannelMaps` describes the physical layout of the Neuropixels probe recording channels on the probe shank(s). It contains a struct array, with one element for each probe. In the example below, there are two elements because the animal had two probes implanted.

```
Dsession.probeChannelMaps = 

  1Ã—2 struct array with fields:

    xcoords
    ycoords
    chanMap
    chanMap0ind
    shankInd
    connected
```
Let's examine the contents of the fields in the first probe. If you are familiar with Kilosort 2.5, you may recognize the channel map format. Each field contains a vector with 384 elements (one for each recording channel).

```
Dsession.probeChannelMaps(1) = 

  struct with fields:

        xcoords: [384Ã—1 double]     x-coordinate of each recording channel (micrometers)
        ycoords: [384Ã—1 double]     y-coordinate of each recording channel (micrometers)
        chanMap: [384Ã—1 double]     1-based numerical index in the set of channels
    chanMap0ind: [384Ã—1 double]     0-based numerical index in the set of channels
       shankInd: [384Ã—1 double]     1-based numerical index of the shank
      connected: [384Ã—1 logical]    indicates whether the channel was enabled in Kilosort
```


## Sleep data files
All .mat files in the folder `data/sleep/` contain data acquired during rest-chamber sessions. Each file contains a single structure named `Dsleep`, containing all data for the given animal ID and session type.

### Data loading:
The example below loads data from rat #25691, recording #1:

```matlab
load("data\sleep\25691_1.mat", 'Dsleep')
```

### Dataset organization:
Below is a table of contents of the root `Dsleep` structure:

```
Dsleep = 

  struct with fields:

                times: [1Ã—1 struct]      times of identified sleep epochs
               params: [1Ã—1 struct]      parameters used in sleep-detection analyis
                 data: [1Ã—1 struct]      timeseries data from sleep-detection analysis
    sessionTimeRanges: [2Ã—2 double]      start/end times of all rest-chamber sessions
            totalTime: [1Ã—1 struct]      total sleep time in seconds (given separately for REM and SWS)
                units: [629Ã—1 struct]    single-unit spike-time data
```	

### Detailed field descriptions

#### times
These fields contain standardized spatial coordinate grids used in tuning maps and other spatial analyses. Each contains a structure, with field names indicating the spatial variables that the grids were used in conjunction with.

```
Dsleep.times =

  struct with fields:

    sws: [19Ã—2 double]    array of SWS epoch times (rows are epochs; columns are [start_time, end_time])
    rem: [6Ã—2 double]     array of REM epoch times (same format as for 
```

#### data
`Dsleep.data` contains timeseries data used in the sleep-detection analysis.

```
Dsleep.data =

  struct with fields:

                  t: [2164701Ã—1 double]     sample timestamps
              speed: [2164701Ã—1 double]     horizontal head speed (m/s)
       angularSpeed: [2164701Ã—1 double]     angular head speed (radians/s)
            isSleep: [2164701Ã—1 logical]    boolean vector, true during immobile ('sleep') periods
       sleepPeriods: [12Ã—2 double]          array of start/end times of immobile ('sleep') periods 
    thetaDeltaRatio: [2164701Ã—1 double]     amplitude ratio of theta (5-10 Hz) to delta (1-4 Hz) activity
```

#### sessionTimeRanges
`Dsleep.sessionTimeRanges` contains the start and end times of all rest-chamber sessions in the recording. The value is a 2-column array, with rows corresponding to sessions. The first and second columns respectively indicate the session start and end time in seconds.

```
 Dsleep.sessionTimeRanges = 

        9576       18812    <-- session 1 starts at 9576s, ends at 18812 s
       25403       27007    <-- session 2 starts at 25403s, ends at 27007 s
```

#### totalTime
`Dsleep.totalTime` contains the total combined length (in seconds) of all REM and SWS periods.

```
Dsleep.totalTime = 

  struct with fields:

    sws: 6.4439e+03
    rem: 1.0375e+03
```

#### units
`Dsleep.units` is a struct array containing single-unit spike-time data.


```
Dsleep.units =

  629Ã—1 struct array with fields:

    spikeTimes:    times (s) of spikes detected during SWS and REM periods 
    id:            unit identifier

```