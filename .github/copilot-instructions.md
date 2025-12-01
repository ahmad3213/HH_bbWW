# HH_bbWW Repository - Copilot Instructions

## Overview

HH→bbWW physics analysis for CMS Run 3 data. Part of cms-flaf/FLAF ecosystem. **Size**: ~71MB. **Language**: Python 3.12+ with ROOT (PyROOT). **Framework**: Law/Luigi + CMSSW environment. **No build system, no unit tests** - interpreted Python with JIT-compiled C++.

**Key Dependencies**: ROOT, Law/Luigi, TensorFlow, ONNX Runtime, Awkward/Uproot, NumPy/pandas/PyYAML, 6 git submodules (FLAF, Corrections, DeepHME, StatInference, SyncTool, inference).

**CRITICAL**: Submodules require SSH access to GitHub/GitLab CERN. Without keys, initialization fails (expected). Code in main repo can still be edited.

## Structure

**Main Directories** (33 .py files, 36 .yaml in config):
- **`AnaProd/`** (2 .py): Input definitions (`anaTupleDef.py`), baseline selections (`baseline.py`)
- **`Analysis/`** (9 .py): Core logic - `hh_bbww.py` (main, 600+ lines), `histTupleDef.py`, `tasks.py` (Law tasks), DNN/HME producers
- **`Studies/`** (22 .py): Research code - `DNN/` (training), `HME/` (mass estimator), `Purity/`, `SignalEfficiency/`
- **`config/`** (36 .yaml): `global.yaml` (main, 400+ lines), `ci_custom.yaml`, `law.cfg`, `background_samples.yaml`, era-specific (Run3_2022/2022EE/2023/2023BPix), `DNN/` (.onnx via Git LFS), `DeepHME/`, `Datacards/`
- **Submodules** (empty until init): FLAF, Corrections, DeepHME, StatInference, SyncTool, inference

**Root Files**: `env.sh` (setup), `.gitignore` (ignores /soft, /data, /.law, __pycache__), `.gitattributes` (Git LFS for .onnx/.keras), `.gitmodules`

## Environment & Execution

**Setup**: `source env.sh` (sets ANALYSIS_PATH, HH_INFERENCE_PATH, sources FLAF/env.sh). **Requires FLAF submodule** - fails without it (expected). Scripts use `sys.path.append(os.environ["ANALYSIS_PATH"])` for imports like `import Analysis.hh_bbww`, `from FLAF.Common.HistHelper import *`.

**No build system** (no Make/CMake/setup.py/pip). Python interpreted, C++ JIT-compiled via `ROOT.gInterpreter.Declare()`. **No unit tests** - validation via CI integration tests, physics checks, SyncTool.

## CI/CD Workflows (3 GitHub Actions)

**All workflows delegate to FLAF repository workflows:**

1. **formatting-check.yaml**: PR trigger, enforces Python formatting (rules in FLAF)
2. **repo-sanity-checks.yaml**: PR trigger, validates structure/YAML syntax/imports
3. **trigger-flaf-integration.yaml**: Comment trigger (`@cms-flaf-bot test`), runs full integration on GitLab CERN (task: HistPlotTask, dataset: XtoYHto2B2Wto2B2L2Nu_MX_300_MY_125, output: /builds/cms-flaf/flaf_integration/output/HH_bbWW). Authorized users in `.github/integration_cfg.yaml`: kandrosov, aebid, ahmad3213, abolshov, valeriadamante.

**CI Config**: `.github/integration_cfg.yaml` (authorized users, version pins), `config/ci_custom.yaml` (CI output path, test variables)

## Running Tasks

**Law framework** (workflow management). Tasks in `Analysis/tasks.py`, `Studies/DNN/tasks.py`, inherited from `FLAF/AnaProd/tasks.py`. Config: `config/law.cfg` (modules, job dir: $ANALYSIS_PATH/data/jobs, local scheduler). Commands: `law index` (list tasks), `law run TaskName --param value --workers N`. Exact commands in FLAF docs/task definitions.

## Code Conventions

**Python**: No formatter config (.flake8/.pylintrc/pyproject.toml) - enforced by FLAF CI. Import order: stdlib, third-party (numpy/ROOT/awkward), FLAF, Analysis. Heavy PyROOT + `ROOT.gInterpreter.Declare()` for C++. Expects ANALYSIS_PATH env var.

**YAML Config**: `global.yaml` (main params, payload producers, corrections), era-specific (Run3_2022/2022EE/2023/2023BPix), `background_samples.yaml` (sampleType field), `phys_models.yaml`.

**Physics Terms**: Channels (SL/DL=Single/Double Lepton), Categories (boosted/resolved), Triggers (HLT), B-tagging (ParticleNet: Loose/Medium/Tight WPs per era), HME (Heavy Mass Estimator), DNN (signal/background discrimination).

## Making Changes

**Code**: Analysis logic → `Analysis/`, input selections → `AnaProd/anaTupleDef.py`/`baseline.py`, config → `config/*.yaml`, studies → `Studies/`.

**Dependencies**: No requirements.txt/setup.py - managed in CMSSW/FLAF. Adding packages requires FLAF maintainer coordination.

**Git LFS**: Models (.onnx/.keras) in `config/DNN/vX/`. Ensure `.gitattributes` coverage, use `git lfs track` for new types.

**Formatting**: Follow existing style (4-space indent, imports organized, lines <120 chars). CI enforces via FLAF workflows.

## Common Issues

**Submodule access fails (SSH)**: Expected without keys. Can edit code but cannot run `env.sh`/tasks (need FLAF imports).

**ImportError/ANALYSIS_PATH missing**: Run `source env.sh` first (requires FLAF submodule).

**ROOT import fails**: Requires CMSSW environment or configured ROOT with Python bindings.

**Missing .onnx files**: Install Git LFS (`git lfs install`), pull models (`git lfs pull`).

## Key Config (`config/global.yaml`)

`anaTupleDef`: AnaProd/anaTupleDef.py, `histTupleDef`: Analysis/histTupleDef.py, `analysis_import`: Analysis.hh_bbww, `phys_model`: Run3_Model, `treeName`: "Events", `tagger_name`: "particleNet", `nEventsPerFile`: 100_000, `corrections`: [mu, trgSF, ele, JEC, JER, btagShape, ...], `payload_producers`: HME/DNN producers with resources.

**Physics**: HH→bbWW (di-Higgs) search in Run 3 data (2022/2022EE/2023/2023BPix), SL/DL channels, ParticleNet b-tagging, DeepHME mass estimation, DNN signal extraction, histograms for StatInference.

## For Coding Agents

**Trust these validated instructions.** Search only if: info incomplete, errors contradict this, need FLAF internals.

**Checklist**:
- Physics analysis repo, not traditional software (no build/tests)
- Interpreted Python + JIT C++, validation via physics checks
- Check config changes, maintain import structure/style
- Preserve YAML structure, don't modify submodules
- CI test via `@cms-flaf-bot test` if needed
- Code execution requires CMSSW (likely unavailable)

**Validation** (CMSSW unavailable):
1. Syntax: `python -m py_compile file.py`
2. YAML: `python -c "import yaml; yaml.safe_load(open('file.yaml'))"`
3. Check imports match patterns, config consistency
4. Let CI workflows validate formatting/structure
