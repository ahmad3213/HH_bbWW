## Overview
The scripts `nanoaod_SL.py` and `nanoaod_DL.py` are used for the single-lepton and dilepton signal efficiency studies, respectively. The basic workflow consists of:
- Checking input files (NANOAOD format).
- Processing events.
- Applying selection criteria.

## Main Tasks
The following functions are used to generate histograms:
- `Report_cutflow_hist`
- `EventsDistribution1D_hist`
- `EventsDistribution2D_hist`
- `TEff_RecoGenMatch`
- `comparecut_hist`

## Running the Scripts
To successfully run the scripts, you need to include some additional functions **before execution**. These functions are **commented in the script** (see *"the functions I have added to `studyYumengAnalysisTools.h`"*).

The run options can be defined at the end of script.