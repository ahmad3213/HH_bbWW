import ROOT
import json
import numpy as np

if __name__ == "__main__":
    sys.path.append(os.environ["ANALYSIS_PATH"])

from FLAF.Common.HistHelper import *
from FLAF.Common.Utilities import *

WorkingPointsParticleNet = {
    "Run3_2022": {"Loose": 0.047, "Medium": 0.245, "Tight": 0.6734},
    "Run3_2022EE": {"Loose": 0.0499, "Medium": 0.2605, "Tight": 0.6915},
    "Run3_2023": {"Loose": 0.0358, "Medium": 0.1917, "Tight": 0.6172},
    "Run3_2023BPix": {"Loose": 0.0359, "Medium": 0.1919, "Tight": 0.6133},
}

def createKeyFilterDict(global_params, period):
    filter_dict = {}
    filter_str = ""
    channels_to_consider = global_params["channels_to_consider"]
    categories = global_params["categories"]
    ### add custom categories eventually:
    custom_categories = []
    custom_categories_name = global_params.get(
        "custom_categories", None
    )  # can be extended to list of names
    if custom_categories_name:
        custom_categories = list(global_params.get(custom_categories_name, []))
        if not custom_categories:
            print("No custom categories found")
    ### regions
    custom_regions = []
    custom_regions_name = global_params.get(
        "custom_regions", None
    )  # can be extended to list of names, if for example adding QCD regions + other control regions
    if custom_regions_name:
        custom_regions = list(global_params.get(custom_regions_name, []))
        if not custom_regions:
            print("No custom regions found")
    all_categories = categories + custom_categories
    custom_subcategories = list(global_params.get("custom_subcategories", []))
    triggers_dict = global_params["triggers"]
    for ch in channels_to_consider:
        triggers_list = triggers_dict[ch]
        triggers_list_complete = [f"HLT_{trg}" for trg in triggers_list]
        triggers_str = "(" + " || ".join(triggers_list_complete) + ")"
        print(triggers_str)
        # if period in triggers_dict[ch].keys():
        #     triggers = triggers_dict[ch][period]
        for reg in custom_regions:
            for cat in all_categories:

                filter_base = f" ( (channelId == {global_params['channelDefinition'][ch]})  && {triggers_str} && {reg} && {cat} ) "
                if custom_subcategories:
                    for subcat in custom_subcategories:
                        # filter_base += f"&& {custom_subcat}"
                        filter_str = f"(" + filter_base + f" && {subcat}"
                        filter_str += ")"
                        key = (ch, reg, cat, subcat)
                        filter_dict[key] = filter_str
                else:
                    filter_str = f"(" + filter_base
                    filter_str += ")"
                    key = (ch, reg, cat)
                    filter_dict[key] = filter_str
    return filter_dict


def GetBTagWeight(global_cfg_dict, cat, applyBtag=False):
    btag_weight = "1"
    btagshape_weight = "1"
    if applyBtag:
        if global_cfg_dict["btag_wps"][cat] != "":
            btag_weight = f"weight_bTagSF_{btag_wps[cat]}_Central"
    else:
        if cat != "btag_shape" and cat != "boosted":
            btagshape_weight = "weight_bTagShape_Central"
    return f"{btag_weight}*{btagshape_weight}"


def GetWeight(channel, cat, boosted_categories):  # do you need all these args?
    # weights_to_apply = ["weight_MC_Lumi_pu", "ExtraDYWeight"]
    weights_to_apply = ["weight_MC_Lumi_pu"]
    total_weight = "*".join(weights_to_apply)
    for lep_index in [1, 2]:
        total_weight = f"{total_weight} * {GetLepWeight(lep_index)}"
    total_weight = f"{total_weight} * {GetTriggerWeight()}"
    return total_weight


def GetLepWeight(lep_index):
    weight_Ele = f"(lep{lep_index}_legType == static_cast<int>(Leg::e) ? weight_lep{lep_index}_EleSF_wp80iso_EleIDCentral : 1.0)"

    # Medium pT Muon SF
    weight_Mu = f"(lep{lep_index}_legType == static_cast<int>(Leg::mu) ? weight_lep{lep_index}_MuonID_SF_TightID_TrkCentral * weight_lep{lep_index}_MuonID_SF_LoosePFIso_TightIDCentral : 1.0)"

    # High pT Muon SF
    # weight_Mu = f"(lep{lep_index}_legType == static_cast<int>(Leg::mu) ? weight_lep{lep_index}_HighPt_MuonID_SF_HighPtIDCentral * weight_lep{lep_index}_HighPt_MuonID_SF_RecoCentral * weight_lep{lep_index}_HighPt_MuonID_SF_TightIDCentral : 1.0)"

    # No Muon SF
    # weight_Mu = f"(lep{lep_index}_legType == static_cast<int>(Leg::mu) ? 1.0 : 1.0)"

    return f"{weight_Mu} * {weight_Ele}"


def GetTriggerWeight():
    weight_MuTrg = f"(lep1_legType == static_cast<int>(Leg::mu) ? weight_lep1_TrgSF_singleIsoMu_Central : 1.0)"
    weight_EleTrg = f"(lep1_legType == static_cast<int>(Leg::e) ? weight_lep1_TrgSF_singleEleWpTight_Central : 1.0)"

    return f"{weight_MuTrg} * {weight_EleTrg}"


class DataFrameBuilderForHistograms(DataFrameBuilderBase):

    def defineCutFlow(self):
        self.df = self.df.Define("cutflow", "int(0)")
        cutflow_cuts = ["event_selection", "OS_Iso", "Zveto || OppFlavor", "mbb_SR"]
        for i, cut in enumerate(cutflow_cuts):
            self.df = self.df.Redefine(
                "cutflow", f"{cut} && cutflow >= {i} ? cutflow+1 : cutflow"
            )

    def defineCategories(self):
        self.DefineAndAppend("SL", "channelId == 1 || channelId == 2")
        self.DefineAndAppend(
            "DL", "channelId == 11 || channelId == 12 || channelId == 22"
        )

        self.DefineAndAppend(
            "nSelBtag_jets",
            # f"int(bjet1_btagPNetB >= {self.bTagWP}) + int(bjet2_btagPNetB >= {self.bTagWP})",
            # f"int(bjet1_btagPNetB >= {self.bTagWP_Loose}) + int(bjet2_btagPNetB >= {self.bTagWP_Loose})",
            f"int(bjet1_idbtagPNetB >= 1) + int(bjet2_idbtagPNetB >= 1)",  # ID 1 is loose
        )
        self.DefineAndAppend(
            "nSelBtag_fatjets", f"int( SelectedFatJet_particleNet_XbbVsQCD[0] >= 0.8 )"
        )
        self.DefineAndAppend(
            "resolved",
            f"(DL && centralJet_pt.size() >= 2) || (SL && centralJet_pt.size() >= 4)",
        )
        self.DefineAndAppend("res2b", f"resolved && nSelBtag_jets >= 2")
        self.DefineAndAppend("res0b", f"resolved && nSelBtag_jets == 0")
        self.DefineAndAppend(
            "boosted",
            f"!res2b && nSelBtag_fatjets > 0 && (DL || (SL && SelectedFatJet_pt.size() > 1) || (SL && centralJet_pt.size() >= 2) ) ",
        )  # Greater than zero, but logic should only allow 0 or 1
        self.DefineAndAppend(
            "recovery",
            f"SelectedFatJet_pt.size() == 0 && resolved && nSelBtag_jets == 1",
        )
        # We are throwing away events with a FatJet that are not b-tagged in this method

        self.DefineAndAppend("inclusive", f"res2b || boosted || recovery")
        self.DefineAndAppend("baseline", f"return true;")

    def defineLeptonPreselection(self):
        self.df = self.df.Define(
            "leadingleppT_ele",
            "((lep1_legType == 1 && lep1_pt  > 32 ) || (lep2_legType == 1 && lep2_pt  > 32))",
        )
        self.df = self.df.Define(
            "leadingleppT_Mu",
            "((lep1_legType == 2 && lep1_pt  > 25 ) || (lep2_legType == 2 && lep2_pt  > 25))",
        )
        self.df = self.df.Define(
            "leadingleppT", "(leadingleppT_ele || leadingleppT_Mu)"
        )  # 32 need to be changed to 25 for DL channel once Double lepton trigger SF are integrated
        self.df = self.df.Define(
            "subleadleppT", "(lep2_legType < 1 || (lep1_pt > 10 && lep2_pt > 10))"
        )
        self.df = self.df.Define(
            "tightlep",
            "((lep1_legType == 2 && lep1_Muon_tightId == 1) || (lep1_legType == 1 && lep1_Electron_mvaIso_WP80 == 1)) && (lep2_legType < 1 || ((lep2_legType == 2 && lep2_Muon_tightId == 1 ) || (lep2_legType == 1 && lep2_Electron_mvaIso_WP80 == 1)))",
        )
        self.df = self.df.Define(
            "tightlep_Iso",
            # " (((lep1_legType == 1 && lep1_Electron_pfRelIso03_all < 0.15) || (lep1_legType == 2 && lep1_Muon_pfRelIso04_all < 0.15)) || ((lep2_legType < 1 ) || ((lep2_legType == 1 && lep2_Electron_pfRelIso03_all < 0.15) || (lep2_legType == 2 && lep2_Muon_pfRelIso04_all < 0.15)) ) )",
            " (((lep1_legType == 1) || (lep1_legType == 2 && lep1_Muon_pfRelIso04_all < 0.15)) || ((lep2_legType < 1 ) || ((lep2_legType == 1) || (lep2_legType == 2 && lep2_Muon_pfRelIso04_all < 0.15)) ) )",  # Remove any electron Iso since we use iso in the ID
        )
        self.df = self.df.Define(
            "Single_lep_trg",
            "(HLT_singleIsoMu && lep1_legType == 2 && lep1_HasMatching_singleIsoMu) || (HLT_singleEleWpTight && lep1_legType == 1 && lep1_HasMatching_singleEleWpTight) ",
        )
        self.df = self.df.Define(
            "event_selection",
            "leadingleppT &&  subleadleppT && Single_lep_trg && tightlep && ( lep2_legType < 1 ||  diLep_mass > 12 )",
        )

    def defineJetSelections(self, isData):
        self.df = self.df.Define("Njets", "centralJet_pt.size()")
        self.df = self.df.Define("jet1_isvalid", "Njets > 0")
        self.df = self.df.Define("jet2_isvalid", "Njets > 1")
        self.df = self.df.Define("fatjet_isvalid", "SelectedFatJet_pt.size() > 0")
        self.df = self.df.Define("fatbjet_isValid", "fatjet_isvalid")
        self.df = self.df.Define(
            "fatsubjet1_isvalid",
            "(SelectedFatJet_SubJet1_isValid == 1  && SelectedFatJet_SubJet1_pt > 20 && abs(SelectedFatJet_SubJet1_eta) < 2.5)",
        )
        self.df = self.df.Define(
            "fatsubjet2_isvalid",
            "(SelectedFatJet_SubJet2_isValid == 1 && SelectedFatJet_SubJet2_pt > 20 && abs(SelectedFatJet_SubJet2_eta) < 2.5)",
        )

        bjet_vars = ["pt", "phi", "eta", "mass", "btagPNetB", "idbtagPNetB"]
        for var in bjet_vars:
            self.df = self.df.Define(
                f"bjet1_{var}", f"jet1_isvalid ? centralJet_{var}[0] : -1.0"
            )
            self.df = self.df.Define(
                f"bjet2_{var}", f"jet2_isvalid ? centralJet_{var}[1] : -1.0"
            )

        wjet_vars = ["pt", "phi", "eta", "mass"]
        for var in wjet_vars:

            self.df = self.df.Define(
                f"wjet1_{var}", f"Njets > 2 ? centralJet_{var}[2] : -10.0"
            )
            self.df = self.df.Define(
                f"wjet2_{var}", f"Njets > 3 ? centralJet_{var}[3] : -10.0"
            )

        fatjet_vars = [
            "pt",
            "phi",
            "eta",
            "mass",
            "particleNet_XbbVsQCD",
            "particleNetWithMass_HbbvsQCD",
        ]
        fatjet_mc_vars = ["hadronFlavour"]
        for var in fatjet_vars:
            self.df = self.df.Define(
                f"fatbjet_{var}", f"fatjet_isvalid ? SelectedFatJet_{var}[0] : -10.0"
            )
        if not isData:
            for var in fatjet_mc_vars:
                self.df = self.df.Define(
                    f"fatbjet_{var}",
                    f"fatjet_isvalid ? SelectedFatJet_{var}[0] : -10.0",
                )

        self.df = self.df.Define(
            "bsubjet1_btagDeepB",
            "fatjet_isvalid ? SelectedFatJet_SubJet1_btagDeepB[0] : -1.0",
        )  # needs to be updated for ak8 PNet
        self.df = self.df.Define(
            "bsubjet2_btagDeepB",
            "fatjet_isvalid ? SelectedFatJet_SubJet2_btagDeepB[0] : -1.0",
        )  # needs to be updated for ak8 PNet

    def defineQCDRegions(self):
        self.DefineAndAppend(
            "OS", "(lep2_legType < 1) || (lep1_charge*lep2_charge < 0)"
        )
        self.DefineAndAppend("SS", "!OS")
        self.DefineAndAppend("Iso", "tightlep_Iso")
        self.DefineAndAppend("AntiIso", f"!Iso")
        self.DefineAndAppend("OS_Iso", f"OS && Iso && event_selection")
        self.DefineAndAppend("SS_Iso", f"SS && Iso && event_selection")
        self.DefineAndAppend("OS_AntiIso", f"OS && AntiIso && event_selection")
        self.DefineAndAppend("SS_AntiIso", f"SS && AntiIso && event_selection")
        # Tight-ID MR
        self.DefineAndAppend(
            "MR_mbbCR_TightID_TightIso",
            "Single_lep_trg && "
            "SL && "
            "tightlep && "
            "resolved && "
            "((lep1_legType == 1) || (lep1_legType == 2 && lep1_Muon_pfRelIso04_all < 0.15)) && "
            "!(centralJet_pt.size() >= 3 && nSelBtag_jets >= 2 && "
            "  bb_mass_PNetRegPtRawCorr_PNetRegPtRawCorrNeutrino > 70 && "
            "  bb_mass_PNetRegPtRawCorr_PNetRegPtRawCorrNeutrino < 150)"
        )
        self.DefineAndAppend(
            "MR_mbbCR_TightID",
            "Single_lep_trg && "
            "SL && "
            "tightlep && "
            "resolved && "
            "!(centralJet_pt.size() >= 3 && nSelBtag_jets >= 2 && "
            "  bb_mass_PNetRegPtRawCorr_PNetRegPtRawCorrNeutrino > 70 && "
            "  bb_mass_PNetRegPtRawCorr_PNetRegPtRawCorrNeutrino < 150)"
        )
        # Anti-ID MR
        self.DefineAndAppend(
            "MR_mbbCR_AntiTightID",
            "Single_lep_trg && "
            "SL && "
            "!tightlep && "
            "resolved && "
            "!(centralJet_pt.size() >= 3 && nSelBtag_jets >= 2 && "
            "  bb_mass_PNetRegPtRawCorr_PNetRegPtRawCorrNeutrino > 70 && "
            "  bb_mass_PNetRegPtRawCorr_PNetRegPtRawCorrNeutrino < 150)"
        )
        self.DefineAndAppend(
            "MR_mbbCR_AntiTightID_or_AntiTightIso",
            "Single_lep_trg && "
            "SL && "
            "resolved && "
            "(!tightlep || (lep1_legType == 2 && lep1_Muon_pfRelIso04_all > 0.15)) && "
            "!(centralJet_pt.size() >= 3 && nSelBtag_jets >= 2 && "
            "bb_mass_PNetRegPtRawCorr_PNetRegPtRawCorrNeutrino > 70 && "
            "bb_mass_PNetRegPtRawCorr_PNetRegPtRawCorrNeutrino < 150)"
        )
        self.DefineAndAppend(
            "MR_mbbCR_AntiIso",
            "Single_lep_trg && "
            "resolved && "
            "(lep1_legType == 2 && lep1_Muon_pfRelIso04_all > 0.15  ) && "
            "!(centralJet_pt.size() >= 3 && nSelBtag_jets >= 2 && "
            "bb_mass_PNetRegPtRawCorr_PNetRegPtRawCorrNeutrino > 70 && "
            "bb_mass_PNetRegPtRawCorr_PNetRegPtRawCorrNeutrino < 150)"
        )
        self.DefineAndAppend(
            "MR_mbbCR_Iso",
            "Single_lep_trg && "
            "SL && "
            "resolved && "
            "(lep1_legType == 2 && lep1_Muon_pfRelIso04_all < 0.15) && "
            "!(centralJet_pt.size() >= 3 && nSelBtag_jets >= 2 && "
            "bb_mass_PNetRegPtRawCorr_PNetRegPtRawCorrNeutrino > 70 && "
            "bb_mass_PNetRegPtRawCorr_PNetRegPtRawCorrNeutrino < 150)"
        )
        # Tight-ID AR
        self.DefineAndAppend(
            "AR_mbbCR_TightID_TightIso",
            "MR_mbbCR_TightID_TightIso && "
            "centralJet_pt.size() >= 3 && nSelBtag_jets >= 1"
        )
        self.DefineAndAppend(
            "AR_mbbCR_TightID",
            "MR_mbbCR_TightID && "
            "centralJet_pt.size() >= 3 && nSelBtag_jets >= 1"
        )
        # Anti-ID AR
        self.DefineAndAppend(
            "AR_mbbCR_AntiTightID",
            "MR_mbbCR_AntiTightID && "
            "centralJet_pt.size() >= 3 && nSelBtag_jets >= 1"
        )
        self.DefineAndAppend(
            "AR_mbbCR_AntiTightID_or_AntiTightIso",
            "MR_mbbCR_AntiTightID_or_AntiTightIso && "
            "centralJet_pt.size() >= 3 && nSelBtag_jets >= 1"
        )
        self.DefineAndAppend(
            "AR_mbbCR_AntiIso",
            "MR_mbbCR_AntiIso && "
            "centralJet_pt.size() >= 3 && nSelBtag_jets >= 1"
        )
        self.DefineAndAppend(
            "AR_mbbCR_Iso",
            "MR_mbbCR_Iso && "
            "centralJet_pt.size() >= 3 && nSelBtag_jets >= 1"
        )

    def defineControlRegions(self):
        # Define Single Muon Control Region (W Region) -- Require Muon + High MT (>50)
        # Define Double Muon Control Region (Z Region) -- Require lep1 lep2 are opposite sign muons, and combined mass is within 10GeV of 91
        self.DefineAndAppend(
            "Zpeak",
            f"(lep1_legType == lep2_legType ) && (abs(diLep_mass - 91.1876) < 10)",
        )
        self.DefineAndAppend(
            "Zveto",
            f"(lep1_legType == lep2_legType ) && (abs(diLep_mass - 91.1876) > 10)",
        )

        self.DefineAndAppend("OppFlavor", f"(lep1_legType != lep2_legType)")

        self.DefineAndAppend("ZVeto_OS_Iso", f"(Zveto || OppFlavor) && OS_Iso")

        self.DefineAndAppend("ZVeto_SS_Iso", f"(Zveto || OppFlavor) && SS_Iso")

        self.DefineAndAppend("ZPeak_OS_Iso", f"(Zpeak || OppFlavor) && OS_Iso")

        self.DefineAndAppend(
            "TTbar_CR", f"OS_Iso && lep1_legType == lep2_legType && diLep_mass > 100 "
        )
        self.DefineAndAppend(
            "mbb_SR",
            f"bb_mass_PNetRegPtRawCorr_PNetRegPtRawCorrNeutrino > 70 && bb_mass_PNetRegPtRawCorr_PNetRegPtRawCorrNeutrino < 150",
        )
        self.DefineAndAppend(
            "Lep1Lep2Jet1Jet2_mass",
            f"(lep1_legType == 2 && lep2_legType == 2) ? Lep1Lep2Jet1Jet2_p4.mass() : 0.0",
        )
        self.DefineAndAppend(
            "Lep1Jet1Jet2_mass", f"(lep1_legType == 2) ? Lep1Jet1Jet2_p4.mass() : 0.0"
        )

    def addDYReweighting(self):
        self.DefineAndAppend(
            "ExtraDYWeight_ee_res2b", f"channelId == 11  && res2b ? 1.4 : 1.0"
        )
        self.DefineAndAppend(
            "ExtraDYWeight_ee_recovery", f"channelId == 11 && recovery ? 1.13 : 1.0"
        )
        self.DefineAndAppend(
            "ExtraDYWeight_mumu_res2b", f"channelId == 22 && res2b ? 1.39 : 1.0"
        )
        self.DefineAndAppend(
            "ExtraDYWeight_mumu_recovery", f"channelId == 22 && recovery ? 1.12 : 1.0"
        )
        self.DefineAndAppend(
            "ExtraDYWeight",
            f"ExtraDYWeight_ee_res2b * ExtraDYWeight_ee_recovery * ExtraDYWeight_mumu_res2b * ExtraDYWeight_mumu_recovery",
        )

    def calculateMT(self):
        self.df = self.df.Define(
            "MT_lep1", f"(lep1_legType > 0) ? Calculate_MT(lep1_p4, PuppiMET_p4) : 0.0"
        )
        self.df = self.df.Define(
            "MT_lep2", f"(lep2_legType > 0) ? Calculate_MT(lep1_p4, PuppiMET_p4) : 0.0"
        )
        self.df = self.df.Define(
            "MT_tot",
            f"(lep1_legType > 0 && lep2_legType > 0) ? Calculate_TotalMT(lep1_p4, lep2_p4, PuppiMET_p4) : 0.0",
        )

    def selectTrigger(self, trigger):
        self.df = self.df.Filter(trigger)

    def addCut(self, cut=""):
        if cut != "":
            self.df = self.df.Filter(cut)

    def defineTriggers(self):
        for ch in self.config["channelSelection"]:
            for trg in self.config["triggers"][ch]:
                trg_name = "HLT_" + trg
                self.colToSave.append(trg_name)
                if trg_name not in self.df.GetColumnNames():
                    print(f"{trg_name} not present in colNames")
                    self.df = self.df.Define(trg_name, "1")
        # singleTau_th_dict = self.config['singleTau_th']
        # singleMu_th_dict = self.config['singleMu_th']
        # singleEle_th_dict = self.config['singleEle_th']
        for trg_name, trg_dict in self.config["application_regions"].items():
            for key in trg_dict.keys():
                region_name = trg_dict["region_name"]
                region_cut = trg_dict["region_cut"].format()
                self.colToSave.append(region_name)
                if region_name not in self.df.GetColumnNames():
                    self.df = self.df.Define(region_name, region_cut)

    def DefineAndAppend(self, varToDefine, var_expression):
        self.df = self.df.Define(varToDefine, var_expression)
        self.colToSave.append(varToDefine)

    def __init__(self, df, config, period, colToSave=[], **kwargs):
        super(DataFrameBuilderForHistograms, self).__init__(df, **kwargs)
        self.config = config
        self.period = period
        self.colToSave = colToSave + ["channelId"]
        self.bTagWP = WorkingPointsParticleNet[period]["Medium"]
        self.bTagWP_Loose = WorkingPointsParticleNet[period][
            "Loose"
        ]  # wp should go to global config.


def defineAllP4(df):
    df = df.Define(f"SelectedFatJet_idx", f"CreateIndexes(SelectedFatJet_pt.size())")
    df = df.Define(
        f"SelectedFatJet_p4",
        f"GetP4(SelectedFatJet_pt, SelectedFatJet_eta, SelectedFatJet_phi, SelectedFatJet_mass, SelectedFatJet_idx)",
    )
    for idx in [0, 1]:
        df = Utilities.defineP4(df, f"lep{idx+1}")
    df = df.Define(
        f"centralJet_p4",
        f"GetP4(centralJet_pt, centralJet_eta, centralJet_phi, centralJet_mass)",
    )
    df = df.Define(
        f"centralJet_PNetRegPtRawCorr_p4",
        f"GetP4(centralJet_pt*(1.0-centralJet_rawFactor)*centralJet_PNetRegPtRawCorr, centralJet_eta, centralJet_phi, centralJet_mass)",
    )
    df = df.Define(
        f"centralJet_PNetRegPtRawCorr_PNetRegPtRawCorrNeutrino_p4",
        f"GetP4(centralJet_pt*(1.0-centralJet_rawFactor)*centralJet_PNetRegPtRawCorr*centralJet_PNetRegPtRawCorrNeutrino, centralJet_eta, centralJet_phi, centralJet_mass)",
    )
    for met_var in ["PuppiMET"]:
        df = df.Define(
            f"{met_var}_p4",
            f"ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>({met_var}_pt,0.,{met_var}_phi,0.)",
        )

    return df


def AddDNNVariables(df):
    df = df.Define("HT", f"Sum(centralJet_pt)")

    df = df.Define("dR_dilep", f"ROOT::Math::VectorUtil::DeltaR(lep1_p4, lep2_p4)")
    df = df.Define(
        "dR_dibjet",
        f"ROOT::Math::VectorUtil::DeltaR(centralJet_p4[0], centralJet_p4[1])",
    )
    df = df.Define(
        "dR_dilep_dibjet",
        f"ROOT::Math::VectorUtil::DeltaR((lep1_p4+lep2_p4), (centralJet_p4[0]+centralJet_p4[1]))",
    )
    df = df.Define(
        "dR_dilep_dijet",
        f"(centralJet_pt.size() >= 4) ? ROOT::Math::VectorUtil::DeltaR((lep1_p4+lep2_p4), (centralJet_p4[2]+centralJet_p4[3])) : -100.",
    )
    df = df.Define(
        "dPhi_lep1_lep2", f"ROOT::Math::VectorUtil::DeltaPhi(lep1_p4,lep2_p4)"
    )
    df = df.Define(
        "dPhi_jet1_jet2",
        f"ROOT::Math::VectorUtil::DeltaPhi(centralJet_p4[0],centralJet_p4[1])",
    )
    df = df.Define(
        "dPhi_MET_dilep",
        f"ROOT::Math::VectorUtil::DeltaPhi(PuppiMET_p4,(lep1_p4+lep2_p4))",
    )
    df = df.Define(
        "dPhi_MET_dibjet",
        f"ROOT::Math::VectorUtil::DeltaPhi(PuppiMET_p4,(centralJet_p4[0]+centralJet_p4[1]))",
    )
    df = df.Define("min_dR_lep0_jets", f"MinDeltaR(lep1_p4, centralJet_p4)")
    df = df.Define("min_dR_lep1_jets", f"MinDeltaR(lep2_p4, centralJet_p4)")

    df = df.Define(
        "MT",
        f"(lep1_legType > 0 && lep2_legType > 0) ? Calculate_TotalMT(lep1_p4, lep2_p4, PuppiMET_p4) : -100.",
    )
    df = df.Define(
        "MT2",
        f"(lep1_legType > 0 && lep2_legType > 0) ? float(analysis::Calculate_MT2(lep1_p4, lep2_p4, centralJet_p4[0], centralJet_p4[1], PuppiMET_p4)) : -100.",
    )

    # Functional form of MT2 claculation
    df = df.Define(
        "MT2_ll",
        f"(lep1_legType > 0 && lep2_legType > 0) ? float(analysis::Calculate_MT2_func(lep1_p4, lep2_p4, centralJet_p4[0] + centralJet_p4[1] + PuppiMET_p4, centralJet_p4[0].mass(), centralJet_p4[1].mass())) : -100.",
    )
    df = df.Define(
        "MT2_bb",
        f"(lep1_legType > 0 && lep2_legType > 0) ? float(analysis::Calculate_MT2_func(centralJet_p4[0], centralJet_p4[1], lep1_p4 + lep2_p4 + PuppiMET_p4, 80.4, 80.4)) : -100.",
    )
    df = df.Define(
        "MT2_blbl",
        f"(lep1_legType > 0 && lep2_legType > 0) ? float(analysis::Calculate_MT2_func(lep1_p4 + centralJet_p4[1], lep2_p4 + centralJet_p4[1], PuppiMET_p4, 0.0, 0.0)) : -100.",
    )

    df = df.Define(
        "CosTheta_bb",
        f"(centralJet_pt.size() > 1) ? analysis::Calculate_CosDTheta(centralJet_p4[0], centralJet_p4[1]) : -100.",
    )
    df = df.Define(
        f"bb_mass",
        "centralJet_pt.size() > 1 ? (centralJet_p4[0]+centralJet_p4[1]).mass() : -100.",
    )
    df = df.Define(
        f"bb_mass_PNetRegPtRawCorr",
        "centralJet_pt.size() > 1 ? (centralJet_PNetRegPtRawCorr_p4[0]+centralJet_PNetRegPtRawCorr_p4[1]).mass() : -100.",
    )
    df = df.Define(
        f"bb_mass_PNetRegPtRawCorr_PNetRegPtRawCorrNeutrino",
        "centralJet_pt.size() > 1 ? (centralJet_PNetRegPtRawCorr_PNetRegPtRawCorrNeutrino_p4[0]+centralJet_PNetRegPtRawCorr_PNetRegPtRawCorrNeutrino_p4[1]).mass() : -100.",
    )

    df = df.Define("diLep_p4", "(lep1_p4+lep2_p4)")
    df = df.Define(
        f"ll_mass",
        f"(lep1_legType > 0 && lep2_legType > 0) ? (lep1_p4+lep2_p4).mass() : -1.0",
    )
    df = df.Define(
        f"diLep_mass",
        f"(lep1_legType > 0 && lep2_legType > 0) ? (lep1_p4+lep2_p4).mass() : -1.0",
    )
    df = df.Define(f"pt_ll", "(lep1_p4+lep2_p4).Pt()")
    df = df.Define(
        "Lep1Lep2Jet1Jet2_p4",
        "(centralJet_pt.size() >= 2) ? (lep1_p4+lep2_p4+centralJet_p4[0]+centralJet_p4[1]) : LorentzVectorM()",
    )
    df = df.Define(
        "Lep1Jet1Jet2_p4",
        "(centralJet_pt.size() >= 2) ? (lep1_p4+centralJet_p4[0]+centralJet_p4[1]) : LorentzVectorM()",
    )
    # fixed PT values for mT_fix (decorrelated from lepton pt)
    # 35 GeV for muons, 30 GeV for electrons
    df = df.Define(
        "pT_fix", "(lep1_legType == static_cast<int>(Leg::mu) ? 35.0 : 30.0)"
    )
    # dphi between lepton and MET using VectorUtil
    df = df.Define(
        "dphi", "abs(ROOT::Math::VectorUtil::DeltaPhi(lep1_p4, PuppiMET_p4))"
    )
    # fixed transverse mass
    df = df.Define("mT_fix", "sqrt(2.0 * pT_fix * PuppiMET_pt * (1.0 - cos(dphi)))")
    #transverse mass SL
    df = df.Define("mT_SL", "sqrt(2.0 * lep1_pt * PuppiMET_pt * (1.0 - cos(dphi)))")
    df = df.Define("lep1_pt_FF", "lep1_pt")
    df = df.Define("lep1_eta_FF", "abs(lep1_eta)")
    return df


def PrepareDfForHistograms(dfForHistograms, isData):
    dfForHistograms.df = defineAllP4(dfForHistograms.df)
    dfForHistograms.df = AddDNNVariables(dfForHistograms.df)
    dfForHistograms.defineTriggers()
    dfForHistograms.defineLeptonPreselection()
    dfForHistograms.defineJetSelections(isData)
    dfForHistograms.defineCategories()
    dfForHistograms.defineQCDRegions()
    dfForHistograms.defineControlRegions()
    dfForHistograms.addDYReweighting()
    dfForHistograms.calculateMT()
    dfForHistograms.defineCutFlow()
    return dfForHistograms
