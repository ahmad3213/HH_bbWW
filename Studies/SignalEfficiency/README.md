## Overview
The 2 scripts nanoaod_SL.py and nanoaod_DL.py are for the single and dilepton signal efficiency study. The basic workflow consists of:
- Checking input files (NANOAOD format).
- Processing events.
- Applying selection criteria.

## Main Tasks
Then the following functions are used for generating histograms:
- `Report_cutflow_hist`
- `EventsDistribution1D_hist`
- `EventsDistribution2D_hist`
- `TEff_RecoGenMatch`
- `comparecut_hist`

## Running the Scripts
To successfully run the scripts, you need to include some additional functions, which I have commented in the script ("the functions I have added to `studyYumengAnalysisTools.h`), before executing the scripts.