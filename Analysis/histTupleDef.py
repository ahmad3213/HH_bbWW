import importlib
from FLAF.Common.Utilities import *
from FLAF.Common.HistHelper import *
from Corrections.Corrections import Corrections
from Corrections.CorrectionsCore import getSystName, central
from FLAF.Common.Setup import Setup
import ROOT

if __name__ == "__main__":
    sys.path.append(os.environ["ANALYSIS_PATH"])

initialized = False
analysis = None


def Initialize():
    global initialized
    if not initialized:
        headers_dir = os.path.dirname(os.path.abspath(__file__))
        ROOT.gROOT.ProcessLine(f".include {os.environ['ANALYSIS_PATH']}")
        ROOT.gInterpreter.Declare(f'#include "FLAF/include/HistHelper.h"')
        ROOT.gInterpreter.Declare(f'#include "FLAF/include/Utilities.h"')
        ROOT.gROOT.ProcessLine(f'#include "FLAF/include/MT2.h"')
        ROOT.gROOT.ProcessLine(f'#include "FLAF/include/Lester_mt2_bisect.cpp"')
        ROOT.gROOT.ProcessLine('#include "FLAF/include/AnalysisTools.h"')
        ROOT.gROOT.ProcessLine('#include "FLAF/include/AnalysisMath.h"')
        initialized = True


def analysis_setup(setup):
    global analysis
    analysis_import = setup.global_params["analysis_import"]
    analysis = importlib.import_module(f"{analysis_import}")


def GetDfw(df, setup, dataset_name):
    global_params = setup.global_params
    isData = dataset_name == "data"
    period = global_params["era"]
    dfw = analysis.DataFrameBuilderForHistograms(df, global_params, period)
    new_dfw = analysis.PrepareDfForHistograms(dfw, isData)
    return new_dfw


central_df_weights_computed = False


def DefineWeightForHistograms(
    *,
    dfw,
    isData,
    uncName,
    uncScale,
    unc_cfg_dict,
    hist_cfg_dict,
    global_params,
    final_weight_name,
    df_is_central,
):
    global central_df_weights_computed
    is_central = uncName == central
    corrections = Corrections.getGlobal()
    lepton_legs = ["lep1", "lep2"]
    offline_legs = ["lep1", "lep2"]
    if not isData and (not central_df_weights_computed or not df_is_central):
        #corrections = Corrections.getGlobal()
        triggers_to_use = set()
        channels = global_params["channelSelection"]
        for channel in channels:
            trigger_list = global_params.get("triggers", {}).get(channel, [])
            for trigger in trigger_list:
                if trigger not in corrections.trigger_dict.keys():
                    raise RuntimeError(
                        f"Trigger does not exist in triggers.yaml, {trigger}"
                    )
                triggers_to_use.add(trigger)

        dfw.df, all_weights = corrections.getNormalisationCorrections(
            dfw.df,
            lepton_legs=lepton_legs,
            offline_legs=offline_legs,
            trigger_names=triggers_to_use,
            unc_source=uncName,
            unc_scale=uncScale,
            ana_caches=None,
            return_variations=is_central and global_params["compute_unc_histograms"],
            use_genWeight_sign_only=True,
        )
        if df_is_central:
            central_df_weights_computed = True

    # Now define fake factor branches (just save, no multiplication)
    # Ensure ROOT sees the real C++ function signatures
    if "fake_factor" in corrections.to_apply:
        dfw.df = corrections.fake_factor.define_branches(dfw.df, lepton_legs)
    print(dfw.df.GetColumnType("weight_lep1_FakeFactor"))
    print(dfw.df.GetColumnType("weight_lep1_FakeFactorErr"))
    print(dfw.df.GetColumnType("weight_lep2_FakeFactor"))
    print(dfw.df.GetColumnType("weight_lep2_FakeFactorErr"))
    categories = global_params["categories"]
    boosted_categories = global_params.get("boosted_categories", [])
    process_group = global_params["process_group"]
    total_weight_expression = (
        # channel, cat, boosted_categories --> these are not needed in the GetWeight function therefore I just put some placeholders
        analysis.GetWeight("", "", boosted_categories)
        if process_group != "data"
        else "1"
    )  # are we sure?
    weight_name = "final_weight"
    if weight_name not in dfw.df.GetColumnNames():
        dfw.df = dfw.df.Define(weight_name, total_weight_expression)
    weight_name_fakes = "final_weight_fakes"
    total_weight_expression_fakes = f"total_weight_expression * weight_lep1_FakeFactor"
    if weight_name_fakes not in dfw.df.GetColumnNames():
        dfw.df = dfw.df.Define(weight_name_fakes, total_weight_expression_fakes)
    if not is_central and type(unc_cfg_dict) == dict:
        if (
            uncName in unc_cfg_dict["norm"].keys()
            and "expression" in unc_cfg_dict["norm"][uncName].keys()
        ):
            weight_name = unc_cfg_dict["norm"][uncName]["expression"].format(
                scale=uncScale
            )
    dfw.df = dfw.df.Define(final_weight_name, weight_name)
