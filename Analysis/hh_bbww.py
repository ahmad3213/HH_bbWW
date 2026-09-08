import ROOT

if __name__ == "__main__":
    sys.path.append(os.environ["ANALYSIS_PATH"])

from FLAF.Common.HistHelper import *
from FLAF.Common.Utilities import *

WorkingPointsParticleNet = {
    "Run3_2022": {"Loose": 0.047, "Medium": 0.245, "Tight": 0.6734},
    "Run3_2022EE": {"Loose": 0.0499, "Medium": 0.2605, "Tight": 0.6915},
    "Run3_2023": {"Loose": 0.0358, "Medium": 0.1917, "Tight": 0.6172},
    "Run3_2023BPix": {"Loose": 0.0359, "Medium": 0.1919, "Tight": 0.6133},
    "Run3_2024": {"Loose": 0.0359, "Medium": 0.1919, "Tight": 0.6133},
    "Run3_2025": {"Loose": 0.0359, "Medium": 0.1919, "Tight": 0.6133},
    "Run3_2026": {"Loose": 0.0359, "Medium": 0.1919, "Tight": 0.6133},
}
WorkingPointsUParTAK4 = {
    # BTV Summer24 NanoAODv15 (L/M/T).
    "Run3_2024": {"Loose": 0.0246, "Medium": 0.1272, "Tight": 0.4648},
    "Run3_2025": {"Loose": 0.0246, "Medium": 0.1272, "Tight": 0.4648},
    "Run3_2026": {"Loose": 0.0246, "Medium": 0.1272, "Tight": 0.4648},
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


def GetWeight(
    weights_this_process, weight_base_name="weight_base"
):  # do you need all these args?
    # weights_this_process is a set of corrections from global.yaml
    # e.g. {'lumi', 'dy_hhbbww', 'trigger', 'base', 'btag', 'dy_hhbbtautau', 'JER', 'pu', 'JEC', 'ele', 'gen', 'muScaRe', 'mu', 'eleES', 'fatjet', 'xs'}

    # weights_to_apply = ["weight_base", "ExtraDYWeight"]
    weights_to_apply = [weight_base_name]
    weights_to_apply_resolved = []
    weights_to_apply_boosted = []

    for lep_index in [1, 2]:
        if "ele" in weights_this_process:
            weights_to_apply.append(f"{GetEleWeight(lep_index)}")
        if "mu" in weights_this_process:
            weights_to_apply.append(f"{GetMuWeight(lep_index)}")
    if "trigger" in weights_this_process:
        weights_to_apply.append(f"{GetTriggerWeight()}")
    if "btag" in weights_this_process:
        weights_to_apply_resolved.append(f"{GetBtagShapeWeight()}")
    if "fatjet" in weights_this_process:
        weights_to_apply_boosted.append(f"{GetFatBtagWeight()}")
    if "dy_hhbbtautau" in weights_this_process:
        weights_to_apply.append(f"{GetDYbbtautauReweight()}")
    if "dy_hhbbww" in weights_this_process:
        weights_to_apply.append(f"{GetDYbbwwReweight()}")

    # total_weight = "*".join(weights_to_apply)
    total_weight_resolved = "*".join(weights_to_apply + weights_to_apply_resolved)
    total_weight_boosted = "*".join(weights_to_apply + weights_to_apply_boosted)

    total_weight = f"(boosted) ? {total_weight_boosted} : {total_weight_resolved}"
    return total_weight


def GetBtagShapeWeight():
    BTag_weight = "weight_bTagShape_Central"
    return BTag_weight


def GetFatBtagWeight():
    FatBTag_weight = "weight_FatJetSF_Central"
    return FatBTag_weight


def GetDYbbtautauReweight():
    DY_bbtautau_weight = "weight_dy_central"
    return DY_bbtautau_weight


def GetDYbbwwReweight():
    DY_bbww_weight = "weight_dy_hhbbww_central"
    return DY_bbww_weight


def GetEleWeight(lep_index):
    weight_Ele = f"(lep{lep_index}_legType == static_cast<int>(Leg::e) ? weight_lep{lep_index}_EleSF_wp80iso_EleIDCentral : 1.0)"
    return weight_Ele


def GetMuWeight(lep_index):
    # Medium pT Muon SF
    weight_Mu = f"(lep{lep_index}_legType == static_cast<int>(Leg::mu) ? weight_lep{lep_index}_MuonID_SF_TightID_TrkCentral * weight_lep{lep_index}_MuonID_SF_TightPFIso_TightIDCentral : 1.0)"

    # High pT Muon SF
    # weight_Mu = f"(lep{lep_index}_legType == static_cast<int>(Leg::mu) ? weight_lep{lep_index}_HighPt_MuonID_SF_HighPtIDCentral * weight_lep{lep_index}_HighPt_MuonID_SF_RecoCentral * weight_lep{lep_index}_HighPt_MuonID_SF_TightIDCentral : 1.0)"

    # No Muon SF
    # weight_Mu = f"(lep{lep_index}_legType == static_cast<int>(Leg::mu) ? 1.0 : 1.0)"

    return weight_Mu


def GetTriggerWeight():
    weight_MuTrg = f"(lep1_legType == static_cast<int>(Leg::mu) ? weight_lep1_TrgSF_singleIsoMu_Central : 1.0)"
    weight_EleTrg = f"(lep1_legType == static_cast<int>(Leg::e) ? weight_lep1_TrgSF_singleEleWpTight_Central : 1.0)"

    return f"{weight_MuTrg} * {weight_EleTrg}"


class DataFrameBuilderForHistograms(DataFrameBuilderBase):

    def defineCutFlow(self):
        self.df = self.df.Define("cutflow", "int(0)")
        cutflow_cuts = [
            "event_selection",
            "OS_Iso",
            "SR",
            "SR_mbb",
            "inclusive",
            "res2b",
        ]
        for i, cut in enumerate(cutflow_cuts):
            self.df = self.df.Redefine(
                "cutflow", f"{cut} && cutflow >= {i} ? cutflow+1 : cutflow"
            )

    def defineLeptonChannel(self):
        self.DefineAndAppend("SL", "channelId == 1 || channelId == 2")
        self.DefineAndAppend(
            "DL", "channelId == 11 || channelId == 12 || channelId == 22"
        )

    def defineCategories(self):
        self.DefineAndAppend("baseline", f"return true;")
        self.DefineAndAppend("baseline_resolved", f"(SelectedFatJet_pt.size() == 0)")
        self.DefineAndAppend("baseline_boosted", f"(SelectedFatJet_pt.size() >= 1)")

        # Test boosted -> res2b -> recovery
        self.DefineAndAppend(
            "HbbCand_isValid", "(bjet1_isValid && bjet2_isValid) || fatbjet_isValid"
        )
        self.DefineAndAppend(
            "WhadCand_isValid", "(wjet1_isValid && wjet2_isValid) || fatwjet_isValid"
        )
        self.DefineAndAppend("inclusive", "HbbCand_isValid && (DL || WhadCand_isValid)")
        self.DefineAndAppend(
            "boosted_H", "inclusive && (fatbjet_isValid && !fatwjet_isValid)"
        )
        self.DefineAndAppend(
            "boosted_W",
            "inclusive && (!fatbjet_isValid && fatwjet_isValid && (bjet1_isBTagged && bjet2_isBTagged))",
        )
        self.DefineAndAppend(
            "boosted_HW", "inclusive && (fatbjet_isValid && fatwjet_isValid)"
        )
        self.DefineAndAppend("boosted", "boosted_H || boosted_W || boosted_HW")
        self.DefineAndAppend("resolved", "inclusive && !boosted")
        self.DefineAndAppend("res2b", "resolved && bjet1_isBTagged && bjet2_isBTagged")
        self.DefineAndAppend("recovery", "resolved && !res2b && bjet1_isBTagged")

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
        # self.df = self.df.Define(
        #     "tightlep",
        #     "((lep1_legType == 2 && lep1_Muon_tightId == 1) || (lep1_legType == 1 && lep1_Electron_mvaIso_WP80 == 1)) && (lep2_legType < 1 || ((lep2_legType == 2 && lep2_Muon_tightId == 1 ) || (lep2_legType == 1 && lep2_Electron_mvaIso_WP80 == 1)))",
        # )

        self.df = self.df.Define(
            "lep1_tight",
            "((lep1_legType == 2 && lep1_Muon_tightId == 1) ||(lep1_legType == 1 && lep1_Electron_mvaIso_WP80 == 1))",
        )
        self.df = self.df.Define(
            "lep2_tight",
            "(lep2_legType < 1) || ((lep2_legType == 2 && lep2_Muon_tightId == 1) || (lep2_legType == 1 && lep2_Electron_mvaIso_WP80 == 1))",
        )
        self.df = self.df.Define(
            "tightlep",
            "(lep1_tight && lep2_tight )",
        )
        self.df = self.df.Define("Antitightlep", "(!lep1_tight || !lep2_tight)")
        self.df = self.df.Define(
            "tightlep_Iso",
            """
            ((lep1_legType != 2) || (lep1_Muon_pfRelIso04_all < 0.15)) &&
            ((lep2_legType != 2) || (lep2_Muon_pfRelIso04_all < 0.15))
            """,
        )
        self.df = self.df.Define(
            "Single_lep_trg",
            """
            (HLT_singleIsoMu && lep1_legType == 2 && lep1_HasMatching_singleIsoMu) || (HLT_singleEleWpTight && lep1_legType == 1 && lep1_HasMatching_singleEleWpTight) ||
            (HLT_singleIsoMu && lep2_legType == 2 && lep2_HasMatching_singleIsoMu) || (HLT_singleEleWpTight && lep2_legType == 1 && lep2_HasMatching_singleEleWpTight)
            """,
        )

        self.df = self.df.Define(
            "event_selection",
            "leadingleppT &&  subleadleppT && Single_lep_trg && tightlep && ( lep2_legType < 1 ||  ll_mass > 12 )",
        )
        self.df = self.df.Define(
            "AntitightID_event_selection",
            "leadingleppT &&  subleadleppT && Single_lep_trg && Antitightlep && ( lep2_legType < 1 ||  ll_mass > 12 )",
        )

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

        self.DefineAndAppend(
            "mbb_SR",
            f"bb_mass_PNetRegPtRawCorr_PNetRegPtRawCorrNeutrino > 70 && bb_mass_PNetRegPtRawCorr_PNetRegPtRawCorrNeutrino < 150",
        )
        # MR
        self.DefineAndAppend(
            "mbbCR_Tight",
            "Single_lep_trg && " "tightlep && " "!mbb_SR",
        )
        self.DefineAndAppend(
            "mbbCR_AntiTight",
            "Single_lep_trg && " "!tightlep && " "!mbb_SR",
        )

    def defineControlRegions(self, stage):
        self.DefineAndAppend(
            "Zpeak",
            f"(lep1_legType == lep2_legType ) && (abs(ll_mass - 91.1876) < 10)",
        )
        self.DefineAndAppend(
            "Zveto",
            f"(lep1_legType == lep2_legType ) && (abs(ll_mass - 91.1876) > 10)",
        )
        self.DefineAndAppend("OppFlavor", f"(lep1_legType != lep2_legType)")
        self.DefineAndAppend("SR", f"ll_mass < 70 && OS_Iso")
        self.DefineAndAppend("SR_mbb", f"ll_mass < 70 && OS_Iso && mbb_SR")
        self.DefineAndAppend("TT_CR", f"ll_mass > 110 && OS_Iso")
        self.DefineAndAppend("DY_CR", f"(abs(ll_mass - 91.1876) < 10) && OS_Iso")
        self.DefineAndAppend("W_CR", f"lep1_MT > 50 && Iso")
        self.DefineAndAppend(
            "AR_AntiTightId",
            f"(Zveto || OppFlavor) && OS && Iso && AntitightID_event_selection ",
        )

        if stage == "HistTuple":
            # Individual mass signal regions
            masspoints = self.config["masspoints"]
            for mp in masspoints:
                self.df = self.df.Define(
                    f"predicted_class_M{mp}",
                    f"""std::vector<double> scores = {{
                            TwoStageDNN_M{mp}_Signal,
                            TwoStageDNN_M{mp}_TT,
                            TwoStageDNN_M{mp}_ST,
                            TwoStageDNN_M{mp}_WJets,
                            TwoStageDNN_M{mp}_DY,
                            TwoStageDNN_M{mp}_H,
                            TwoStageDNN_M{mp}_VV
                        }};
                        auto it = std::max_element(scores.begin(), scores.end());
                        size_t cls = it - scores.begin();
                        return cls;
                    """,
                )

                self.DefineAndAppend(
                    f"SR_SL_M{mp}",
                    f"return predicted_class_M{mp} == 0 && Iso && event_selection;",
                )
                self.DefineAndAppend(
                    f"CR_SL_TT_M{mp}",
                    f"return predicted_class_M{mp} == 1 && Iso && event_selection;",
                )
                self.DefineAndAppend(
                    f"CR_SL_ST_M{mp}",
                    f"return predicted_class_M{mp} == 2 && Iso && event_selection;",
                )
                self.DefineAndAppend(
                    f"CR_SL_WJets_M{mp}",
                    f"return predicted_class_M{mp} == 3 && Iso && event_selection;",
                )
                self.DefineAndAppend(
                    f"CR_SL_DY_M{mp}",
                    f"return predicted_class_M{mp} == 4 && Iso && event_selection;",
                )
                self.DefineAndAppend(
                    f"CR_SL_H_M{mp}",
                    f"return predicted_class_M{mp} == 5 && Iso && event_selection;",
                )
                self.DefineAndAppend(
                    f"CR_SL_VV_M{mp}",
                    f"return predicted_class_M{mp} == 6 && Iso && event_selection;",
                )

    def calculateMT(self):
        self.df = self.df.Define(
            "lep1_MT", f"(lep1_legType > 0) ? Calculate_MT(lep1_p4, PuppiMET_p4) : 0.0"
        )
        self.df = self.df.Define(
            "lep2_MT", f"(lep2_legType > 0) ? Calculate_MT(lep2_p4, PuppiMET_p4) : 0.0"
        )
        self.df = self.df.Define(
            "total_MT",
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
    for met_var in ["PuppiMET"]:
        df = df.Define(
            f"{met_var}_p4",
            f"ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>({met_var}_pt,0.,{met_var}_phi,0.)",
        )
    return df


def AddDNNVariablesDL(df, isData=False):

    # Define needed p4
    df = df.Define(
        "ll_p4",
        "(lep1_legType > 0) && (lep2_legType > 0) ? (lep1_p4 + lep2_p4) : LorentzVectorM()",
    )
    df = df.Define(
        "l1b1_p4",
        "(lep1_legType > 0) && (bjet1_isValid) ? lep1_p4 + bjet1_p4 : LorentzVectorM()",
    )
    df = df.Define(
        "l1b2_p4",
        "(lep1_legType > 0) && (bjet2_isValid) ? lep1_p4 + bjet2_p4 : LorentzVectorM()",
    )
    df = df.Define(
        "l2b1_p4",
        "(lep2_legType > 0) && (bjet1_isValid) ? lep2_p4 + bjet1_p4 : LorentzVectorM()",
    )
    df = df.Define(
        "l2b2_p4",
        "(lep2_legType > 0) && (bjet2_isValid) ? lep2_p4 + bjet2_p4 : LorentzVectorM()",
    )

    # ll variables
    df = df.Define("ll_mass", "ll_p4.mass()")
    df = df.Define(
        "ll_pt", "ll_p4.Pt()"
    )  # Used in bbWW DY reweight, name configured in global.yaml

    if not isData:
        df = df.Define(
            "ll_pt_gen", "LHE_Vpt"
        )  # Used in bbtautau DY reweight, name configured in global.yaml
    else:
        df = df.Define(
            "ll_pt_gen", "-1.0"
        )  # Not required for reweight, but needed to make histograms of the variable

    # mass variables
    df = df.Define(
        "b1leps_mass",
        "ROOT::Math::VectorUtil::DeltaR(bjet1_p4, lep1_p4) < ROOT::Math::VectorUtil::DeltaR(bjet1_p4, lep2_p4) ? (bjet1_p4 + lep1_p4).M() : (bjet1_p4 + lep2_p4).M()",
    )
    df = df.Define(
        "b2leps_mass",
        "ROOT::Math::VectorUtil::DeltaR(bjet2_p4, lep1_p4) < ROOT::Math::VectorUtil::DeltaR(bjet2_p4, lep2_p4) ? (bjet2_p4 + lep1_p4).M() : (bjet2_p4 + lep2_p4).M()",
    )

    df = df.Define("llmet_mass", "(ll_p4 + PuppiMET_p4).M()")
    df = df.Define("bbllmet_mass", "(Hbb_p4 + ll_p4 + PuppiMET_p4).M()")

    # dR variables
    df = df.Define("ll_dR", f"ROOT::Math::VectorUtil::DeltaR(lep1_p4, lep2_p4)")
    df = df.Define("ll_bb_dR", f"ROOT::Math::VectorUtil::DeltaR((ll_p4), (Hbb_p4))")
    df = df.Define("ll_jj_dR", f"ROOT::Math::VectorUtil::DeltaR((ll_p4), (hadW_p4))")

    # dPhi variables
    df = df.Define("ll_dphi", f"ROOT::Math::VectorUtil::DeltaPhi(lep1_p4,lep2_p4)")
    df = df.Define(
        "met_ll_dphi",
        f"ROOT::Math::VectorUtil::DeltaPhi(PuppiMET_p4,(ll_p4))",
    )

    # MT and MT2 variables
    df = df.Define(
        "MT2",
        f"(lep1_legType > 0 && lep2_legType > 0) ? float(analysis::Calculate_MT2(lep1_p4, lep2_p4, bjet1_p4, bjet2_p4, PuppiMET_p4)) : -100.",
    )
    df = df.Define(
        "MT2_ll",
        f"(lep1_legType > 0 && lep2_legType > 0) ? float(analysis::Calculate_MT2_func(lep1_p4, lep2_p4, bjet1_p4 + bjet2_p4 + PuppiMET_p4, bjet1_p4.mass(), bjet2_p4.mass())) : -100.",
    )
    df = df.Define(
        "MT2_bb",
        f"(lep1_legType > 0 && lep2_legType > 0) ? float(analysis::Calculate_MT2_func(bjet1_p4, bjet2_p4, ll_p4 + PuppiMET_p4, 80.4, 80.4)) : -100.",
    )
    # New MT2 implementation for ttbar and returning invisible splitting solution
    # vis=(lep+b, lep+b), invis=MET, chi=0 (neutrino)
    # Both computed via _withSolution which is added into MT2.h to recover the neutrino momentum splitting at the MT2 minimum.
    blbl_pairings = [("l1b1_p4", "l2b2_p4"), ("l1b2_p4", "l2b1_p4")]
    for i, (vis1, vis2) in enumerate(blbl_pairings, start=1):
        df = df.Define(
            f"MT2_blbl{i}_sol",
            f"(lep1_legType > 0 && lep2_legType > 0) ? analysis::Calculate_MT2_func_withSolution({vis1}, {vis2}, PuppiMET_p4, 0.0, 0.0) : analysis::MT2Result{{-100., 0., 0., 0., 0.}}",
        )
        df = df.Define(f"MT2_blbl{i}", f"float(MT2_blbl{i}_sol.mt2)")
        df = df.Define(f"MT2_blbl{i}_nu1_px", f"float(MT2_blbl{i}_sol.pxInvisible1)")
        df = df.Define(f"MT2_blbl{i}_nu1_py", f"float(MT2_blbl{i}_sol.pyInvisible1)")
        df = df.Define(f"MT2_blbl{i}_nu2_px", f"float(MT2_blbl{i}_sol.pxInvisible2)")
        df = df.Define(f"MT2_blbl{i}_nu2_py", f"float(MT2_blbl{i}_sol.pyInvisible2)")
        df = df.Define(
            f"MT2_blbl{i}_nu1_p4",
            f"ROOT::Math::PxPyPzEVector(MT2_blbl{i}_nu1_px, MT2_blbl{i}_nu1_py, 0.0, sqrt(MT2_blbl{i}_nu1_px*MT2_blbl{i}_nu1_px + MT2_blbl{i}_nu1_py*MT2_blbl{i}_nu1_py))",
        )  # MT2 is a transverse quantity so pz = 0
        df = df.Define(
            f"MT2_blbl{i}_nu2_p4",
            f"ROOT::Math::PxPyPzEVector(MT2_blbl{i}_nu2_px, MT2_blbl{i}_nu2_py, 0.0, sqrt(MT2_blbl{i}_nu2_px*MT2_blbl{i}_nu2_px + MT2_blbl{i}_nu2_py*MT2_blbl{i}_nu2_py))",
        )
        df = df.Define(
            f"MT2_blbl{i}_delta_phi",
            f"ROOT::Math::VectorUtil::DeltaPhi(MT2_blbl{i}_nu1_p4, MT2_blbl{i}_nu2_p4)",
        )
        df = df.Define(
            f"MT2_blbl{i}_ptratio", f"MT2_blbl{i}_nu1_p4.Pt() / PuppiMET_p4.Pt()"
        )

    df = df.Define("MT2_blbl_min", "float(min(MT2_blbl1, MT2_blbl2))")
    df = df.Define("MT2_blbl_max", "float(max(MT2_blbl1, MT2_blbl2))")

    # dR pairing: assign lep+b by smallest total deltaR sum
    df = df.Define(
        "MT2_blbl_dR",
        f"(lep1_legType > 0 && lep2_legType > 0) ? float("
        f"(ROOT::Math::VectorUtil::DeltaR(lep1_p4, bjet1_p4) + ROOT::Math::VectorUtil::DeltaR(lep2_p4, bjet2_p4)) <= "
        f"(ROOT::Math::VectorUtil::DeltaR(lep1_p4, bjet2_p4) + ROOT::Math::VectorUtil::DeltaR(lep2_p4, bjet1_p4)) ? "
        f"analysis::Calculate_MT2_func(l1b1_p4, l2b2_p4, PuppiMET_p4, 0.0, 0.0) : "
        f"analysis::Calculate_MT2_func(l1b2_p4, l2b1_p4, PuppiMET_p4, 0.0, 0.0)"
        f") : -100.",
    )

    # Extras
    df = df.Define(
        "bb_CosTheta",
        f"(centralJet_pt.size() > 1) ? analysis::Calculate_CosDTheta(bjet1_p4, bjet2_p4) : -100.",
    )
    df = df.Define(
        "Lep1Lep2Jet1Jet2_p4",
        "(bjet1_isValid && bjet2_isValid) ? (ll_p4+bjet1_p4+bjet2_p4) : LorentzVectorM()",
    )
    df = df.Define(
        "Lep1Jet1Jet2_p4",
        "(bjet1_isValid && bjet2_isValid) ? (lep1_p4+bjet1_p4+bjet2_p4) : LorentzVectorM()",
    )
    df = df.Define(
        "Lep1Lep2Jet1Jet2_mass",
        f"(lep1_legType > 0 && lep2_legType > 0) ? Lep1Lep2Jet1Jet2_p4.mass() : 0.0",
    )
    df = df.Define(
        "Lep1Jet1Jet2_mass", f"(lep1_legType > 0) ? Lep1Jet1Jet2_p4.mass() : 0.0"
    )

    # fixed PT values for mT_fix (decorrelated from lepton pt)
    # 35 GeV for muons, 30 GeV for electrons
    df = df.Define(
        "pT_fix", "(lep1_legType == static_cast<int>(Leg::mu) ? 35.0 : 30.0)"
    )
    # dphi between lepton and MET using VectorUtil
    df = df.Define(
        "dphi_fix", "abs(ROOT::Math::VectorUtil::DeltaPhi(lep1_p4, PuppiMET_p4))"
    )
    # dphi between lepton and MET using VectorUtil
    df = df.Define(
        "dphi", "abs(ROOT::Math::VectorUtil::DeltaPhi(lep1_p4, PuppiMET_p4))"
    )
    # fixed transverse mass
    df = df.Define("mT_fix", "sqrt(2.0 * pT_fix * PuppiMET_pt * (1.0 - cos(dphi_fix)))")
    # transverse mass SL
    df = df.Define("mT_SL", "sqrt(2.0 * lep1_pt * PuppiMET_pt * (1.0 - cos(dphi)))")
    df = df.Define("nExtraLeps", "nExtraMuon + nExtraElectron")

    df = df.Define(
        "min_lb_dR",
        f"min("
        f"min(ROOT::Math::VectorUtil::DeltaR(lep1_p4, bjet1_p4), ROOT::Math::VectorUtil::DeltaR(lep1_p4, bjet2_p4)),"
        f"min(ROOT::Math::VectorUtil::DeltaR(lep2_p4, bjet1_p4), ROOT::Math::VectorUtil::DeltaR(lep2_p4, bjet2_p4))"
        ")",
    )

    df = df.Define(
        "lb_centrality",
        "(bjet1_pt + bjet2_pt + PuppiMET_pt) > 0.0 ? "
        "(lep1_pt + lep2_pt) / (bjet1_pt + bjet2_pt + PuppiMET_pt) : 0.0",
    )

    df = df.Define("HT_total", "HT + PuppiMET_pt + lep1_pt + lep2_pt")

    df = df.Define("ll_bb_pt_ratio", "bb_pt > 0.0 ? ll_pt / bb_pt : 0.0")

    df = df.Define(
        "min_lmet_mt",
        f"min("
        "pow(2 * lep1_pt * PuppiMET_pt * (1 - cos(lep1_phi - PuppiMET_phi)), 0.5),"
        "pow(2 * lep2_pt * PuppiMET_pt * (1 - cos(lep2_phi - PuppiMET_phi)), 0.5)"
        ")",
    )

    df = df.Define("ll_cos_theta_star", "tanh((lep1_eta - lep2_eta) / 2.0)")

    df = df.Define(
        "min_lmet_dphi",
        f"min("
        "abs(ROOT::Math::VectorUtil::DeltaPhi(lep1_p4, PuppiMET_p4)),"
        "abs(ROOT::Math::VectorUtil::DeltaPhi(lep2_p4, PuppiMET_p4))"
        ")",
    )

    df = df.Define(
        "ll_met_MT_WithMass",
        "static_cast<float>(Calculate_MT_WithMass(lep1_p4 + lep2_p4, PuppiMET_p4))",
    )

    return df


def defineJetSelections(df, isData, period="Run3_2023BPix"):
    # Define vars to save
    jet_vars = [
        "p4",
        "pt",
        "phi",
        "eta",
        "mass",
        "btagPNetB",
        "idbtagPNetB",
        "btagUParTAK4B",
        "idbtagUParTAK4B",
        "rawFactor",
        "PNetRegPtRawCorr",
        "PNetRegPtRawCorrNeutrino",
    ]
    jet_mc_vars = ["hadronFlavour", "partonFlavour"]
    fatjet_vars = [
        "p4",
        "pt",
        "phi",
        "eta",
        "mass",
        "particleNetWithMass_HbbvsQCD",
        "particleNet_XqqVsQCD",
        "particleNetWithMass_WvsQCD",
        "particleNet_massCorr",
        "msoftdrop",
        "nConstituents",
        "tau1",
        "tau2",
        "tau3",
        "tau4",
    ]
    fatjet_mc_vars = ["hadronFlavour"]  # SelectedFatJet does not have partonFlavour
    if not isData:
        fatjet_vars = fatjet_vars + fatjet_mc_vars
        jet_vars = jet_vars + jet_mc_vars
    existing_cols = {str(c) for c in df.GetColumnNames()}
    jet_vars = [v for v in jet_vars if v == "p4" or f"centralJet_{v}" in existing_cols]
    fatjet_vars = [
        v for v in fatjet_vars if v == "p4" or f"SelectedFatJet_{v}" in existing_cols
    ]

    # First step is to decide Hbb boosted
    # Take FatJets, mask by BTag and msoftdrop, sort by BTag Score
    # If one exists, category is Hbb_Boosted

    df = df.Define(
        "FatBJet_Sel",
        "SelectedFatJet_particleNetWithMass_HbbvsQCD > 0.92 && SelectedFatJet_msoftdrop > 30",
    )
    df = df.Define("FatBJet_idx", "CreateIndexes(Sum(FatBJet_Sel))")
    df = df.Define(
        "FatBJet_idxSorted",
        "Take(ReorderObjects(SelectedFatJet_particleNetWithMass_HbbvsQCD[FatBJet_Sel], FatBJet_idx), min((int)FatBJet_idx.size(), 1))",
    )
    for var in fatjet_vars:
        df = df.Define(
            f"FatBJet_{var}",
            f"Take(SelectedFatJet_{var}[FatBJet_Sel], FatBJet_idxSorted)",
        )

    # Do not need a selection, we should just take the top 2 score Jets whether they pass the cut
    # df = df.Define("BJet_Sel", "centralJet_idbtagPNetB >= 1")
    if (
        "centralJet_idbtagPNetB" in existing_cols
        or "centralJet_idbtagUParTAK4B" in existing_cols
    ):
        id_sel = (
            "centralJet_idbtagUParTAK4B"
            if "centralJet_idbtagUParTAK4B" in existing_cols
            else "centralJet_idbtagPNetB"
        )
        df = df.Define("BJet_Sel", f"{id_sel} >= -1")
    else:
        df = df.Define("BJet_Sel", "centralJet_pt >= 0.f")
    use_upart = "centralJet_btagUParTAK4B" in existing_cols
    sort_score = "centralJet_btagUParTAK4B" if use_upart else "centralJet_btagPNetB"
    df = df.Define("BJet_idx", "CreateIndexes(Sum(BJet_Sel))")
    df = df.Define(
        "BJet_idxSorted",
        f"Take(ReorderObjects({sort_score}[BJet_Sel], BJet_idx), min((int)BJet_idx.size(), 2))",
    )
    for var in jet_vars:
        df = df.Define(
            f"BJet_{var}", f"Take(centralJet_{var}[BJet_Sel], BJet_idxSorted)"
        )

    df = df.Define("Nbjets", "BJet_pt.size()")
    df = df.Define("bjet1_isValid", "(Nbjets > 0)")
    df = df.Define("bjet2_isValid", "(Nbjets > 1)")

    if "idbtagUParTAK4B" in jet_vars:
        id_branch = "idbtagUParTAK4B"
        score_branch = None
        bTagWP = None
    elif "idbtagPNetB" in jet_vars:
        id_branch = "idbtagPNetB"
        score_branch = None
        bTagWP = None
    elif use_upart:
        id_branch = None
        score_branch = "btagUParTAK4B"
        bTagWP = WorkingPointsUParTAK4.get(period, WorkingPointsParticleNet[period])[
            "Medium"
        ]
    else:
        id_branch = None
        score_branch = "btagPNetB"
        bTagWP = WorkingPointsParticleNet[period]["Medium"]
    if id_branch is not None:
        df = df.Define(
            "bjet1_isBTagged", f"bjet1_isValid ? BJet_{id_branch}[0] >= 1 : 0"
        )
        df = df.Define(
            "bjet2_isBTagged", f"bjet2_isValid ? BJet_{id_branch}[1] >= 1 : 0"
        )
    else:
        df = df.Define(
            "bjet1_isBTagged",
            f"bjet1_isValid ? BJet_{score_branch}[0] >= {bTagWP}f : 0",
        )
        df = df.Define(
            "bjet2_isBTagged",
            f"bjet2_isValid ? BJet_{score_branch}[1] >= {bTagWP}f : 0",
        )
    df = df.Define(
        f"nBTaggedJets", "int(bjet1_isBTagged) + int(bjet2_isBTagged)"
    )  # Used in bbtautau DY reweight, name configured in global.yaml

    df = df.Define("Nfatbjets", "FatBJet_pt.size()")
    df = df.Define("fatbjet_isValid", "(Nfatbjets > 0)")

    df = df.Define("Hbb_Boosted", "fatbjet_isValid")

    for var in jet_vars:
        df = df.Define(
            f"bjet1_{var}",
            f"bjet1_isValid ? BJet_{var}[0] : std::decay_t<decltype(BJet_{var})>::value_type()",
        )
        df = df.Define(
            f"bjet2_{var}",
            f"bjet2_isValid ? BJet_{var}[1] : std::decay_t<decltype(BJet_{var})>::value_type()",
        )

    for var in fatjet_vars:
        df = df.Define(
            f"fatbjet_{var}",
            f"fatbjet_isValid ? FatBJet_{var}[0] : std::decay_t<decltype(FatBJet_{var})>::value_type()",
        )

    df = df.Define(
        f"fatbjet_mass_PNetCorr",
        "fatbjet_isValid ? FatBJet_mass[0] * FatBJet_particleNet_massCorr[0] : std::decay_t<decltype(FatBJet_mass)>::value_type()",
    )

    df = df.Define("Nfatjets", "SelectedFatJet_pt.size()")
    df = df.Define(
        "hadWcand_FatJet",
        "SelectedFatJet_pt > 0.0 && SelectedFatJet_msoftdrop > 20.0 && SelectedFatJet_particleNetWithMass_WvsQCD > 0.1",
    )
    # Create a mask removing FatJets that are chosen as the FatBJet
    df = df.Define(
        "FatWJet_HbbBoostedSel",
        "RemoveOverlaps(SelectedFatJet_p4, hadWcand_FatJet, FatBJet_p4, 0.1)",
    )
    # Create a mask removing FatJets that overlap within 0.8 dR of the Jets chosen as BJets
    df = df.Define(
        "FatWJet_HbbResolvedSel",
        "RemoveOverlaps(SelectedFatJet_p4, hadWcand_FatJet, BJet_p4, 0.8)",
    )

    df = df.Define(
        "FatWJet_Sel",
        "Hbb_Boosted ? FatWJet_HbbBoostedSel : FatWJet_HbbResolvedSel",
    )

    df = df.Define(
        "FatWJet_idx",
        "CreateIndexes(Sum(FatWJet_Sel))",
    )
    df = df.Define(
        "FatWJet_idxSorted",
        "ReorderObjects(SelectedFatJet_pt[FatWJet_Sel], FatWJet_idx)",
    )
    for var in fatjet_vars:
        df = df.Define(
            f"FatWJet_{var}",
            f"Take(SelectedFatJet_{var}[FatWJet_Sel], FatWJet_idxSorted)",
        )

    df = df.Define("Nfatwjets", "FatWJet_pt.size()")
    df = df.Define("fatwjet_isValid", "(Nfatwjets > 0) && !DL")
    df = df.Define(
        "fatwjet",
        "fatwjet_isValid ? FatWJet_p4[0] : LorentzVectorM()",
    )

    df = df.Define(
        f"fatwjet_mass_PNetCorr",
        "fatwjet_isValid ? FatWJet_mass[0] * FatWJet_particleNet_massCorr[0] : std::decay_t<decltype(FatWJet_mass)>::value_type()",
    )

    df = df.Define("Njets", "centralJet_pt.size()")
    df = df.Define("AllTrue_Jet", "centralJet_pt > 0.0")
    # Create a mask removing Jets that are chosen as the BJets
    df = df.Define(
        "WJet_HbbResolvedSel",
        "RemoveOverlaps(centralJet_p4, AllTrue_Jet, BJet_p4, 0.1)",
    )

    df = df.Define(
        "WJet_HbbBoostedSel",
        "RemoveOverlaps(centralJet_p4, AllTrue_Jet, FatBJet_p4, 0.8)",
    )

    df = df.Define(
        "WJet_Sel",
        "Hbb_Boosted ? WJet_HbbBoostedSel : WJet_HbbResolvedSel",
    )

    df = df.Define(
        "WJet_idx",
        "CreateIndexes(Sum(WJet_Sel))",
    )
    df = df.Define(
        "WJet_idxSorted",
        "ReorderObjects(centralJet_pt[WJet_Sel], WJet_idx)",
    )
    for var in jet_vars:
        df = df.Define(
            f"WJet_{var}",
            f"Take(centralJet_{var}[WJet_Sel], WJet_idxSorted)",
        )

    df = df.Define("wjet1_isValid", "(WJet_p4.size() > 0) && !DL")
    df = df.Define("wjet2_isValid", "(WJet_p4.size() > 1) && !DL")
    df = df.Define(
        "wjet1",
        "wjet1_isValid ? WJet_p4[0] : LorentzVectorM()",
    )
    df = df.Define(
        "wjet2",
        "wjet2_isValid ? WJet_p4[1] : LorentzVectorM()",
    )
    df = df.Define(
        "resolvedW_pt",
        "(wjet1_isValid && wjet2_isValid) ? (wjet1 + wjet2).Pt() : -1.0",
    )
    df = df.Define(
        "WJets_Boosted",
        "fatwjet_isValid && (resolvedW_pt < 0.0 || fatwjet.Pt() > resolvedW_pt)",
    )

    # Finally save the simple-name variables
    # bjet1, bjet2, fatbjet, wjet1, wjet2, fatwjet

    for var in jet_vars:
        df = df.Define(
            f"wjet1_{var}",
            f"wjet1_isValid ? WJet_{var}[0] : std::decay_t<decltype(WJet_{var})>::value_type()",
        )
        df = df.Define(
            f"wjet2_{var}",
            f"wjet2_isValid ? WJet_{var}[1] : std::decay_t<decltype(WJet_{var})>::value_type()",
        )

    for var in fatjet_vars:
        df = df.Define(
            f"fatwjet_{var}",
            f"fatwjet_isValid ? FatWJet_{var}[0] : std::decay_t<decltype(FatWJet_{var})>::value_type()",
        )

    # Lastly set mbb
    # PNet Corrections are currently incorrect
    # Using 1.0-rawFactor doesn't work since Jet corrections are already applied
    for bjet_idx in [1, 2]:
        df = df.Define(
            f"bjet{bjet_idx}_PNetRegPtRawCorr_p4",
            f"ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>(bjet{bjet_idx}_pt*(1.0-bjet{bjet_idx}_rawFactor)*bjet{bjet_idx}_PNetRegPtRawCorr, bjet{bjet_idx}_eta, bjet{bjet_idx}_phi, bjet{bjet_idx}_mass)",
        )
        df = df.Define(
            f"bjet{bjet_idx}_PNetRegPtRawCorr_PNetRegPtRawCorrNeutrino_p4",
            f"ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>(bjet{bjet_idx}_pt*(1.0-bjet{bjet_idx}_rawFactor)*bjet{bjet_idx}_PNetRegPtRawCorr*bjet{bjet_idx}_PNetRegPtRawCorrNeutrino, bjet{bjet_idx}_eta, bjet{bjet_idx}_phi, bjet{bjet_idx}_mass)",
        )

    df = df.Define(
        f"bb_mass",
        """
            if (fatbjet_isValid)
                return static_cast<float>(fatbjet_p4.M());
            else if (bjet1_isValid && bjet2_isValid)
                return static_cast<float>((bjet1_p4 + bjet2_p4).M());
            return static_cast<float>(std::decay_t<decltype(BJet_mass)>::value_type());
            """,
    )
    df = df.Define(
        f"bb_mass_PNetRegPtRawCorr",
        "bjet1_isValid && bjet2_isValid ? (bjet1_PNetRegPtRawCorr_p4+bjet2_PNetRegPtRawCorr_p4).mass() : std::decay_t<decltype(BJet_mass)>::value_type()",
    )
    df = df.Define(
        f"bb_mass_PNetRegPtRawCorr_PNetRegPtRawCorrNeutrino",
        "bjet1_isValid && bjet2_isValid ? (bjet1_PNetRegPtRawCorr_PNetRegPtRawCorrNeutrino_p4+bjet2_PNetRegPtRawCorr_PNetRegPtRawCorrNeutrino_p4).mass() : std::decay_t<decltype(BJet_mass)>::value_type()",
    )

    # define p4 of hadronic W candidate
    df = df.Define(
        "hadW_p4",
        """
            if (wjet1_isValid && wjet2_isValid && !WJets_Boosted)
                return wjet1_p4 + wjet2_p4;
            else if (fatwjet_isValid)
                return fatwjet_p4;
            else if (wjet1_isValid && wjet2_isValid)
                return wjet1_p4 + wjet2_p4;
            return LorentzVectorM();
        """,
    )

    # define p4 of leading b jet
    df = df.Define(
        "leadbjet_p4",
        """
            if (fatbjet_isValid)
                return fatbjet_p4;
            else if (bjet1_isValid && bjet2_isValid)
                return bjet1_p4.Pt() > bjet2_p4.Pt() ? bjet1_p4 : bjet2_p4;
            else
                return LorentzVectorM();
        """,
    )

    # define p4 of H->bb candidate
    df = df.Define(
        "Hbb_p4",
        """
            if (fatbjet_isValid)
                return fatbjet_p4;
            else if (bjet1_isValid && bjet2_isValid)
                return bjet1_p4 + bjet2_p4;
            else
                return LorentzVectorM();
        """,
    )

    return df


def defineTopCandP4(df):
    """
    Function computing top candidate's p4 in SL channel.
    Requires categories to be defined.
    """
    # compute leptonic W candidate assuming MET = (nu_px, nu_py) and onshell W from t->bW decays
    # first prepare auxilliary p4s of two possible leptonic W candidates
    # there are two because pz is determined by quadratic equation
    # in case of two solutions use both labelled pos and neg
    # when discriminant is < 0, use real part for both pos and neg
    df = df.Define(
        "lambda_top",
        """
            float mw = 80.1f;
            float lep_pt = lep1_p4.Pt();
            float lep_E = lep1_p4.E();
            float lep_pz = lep1_p4.Pz();
            float lambda_top = mw*mw/2 + PuppiMET_p4.Px()*lep1_p4.Px() + PuppiMET_p4.Py()*lep1_p4.Py();
            return lambda_top;
        """,
    )
    df = df.Define(
        "disc_top_sqr",
        """
            float mw = 80.1f;
            float lep_pt = lep1_p4.Pt();
            float lep_E = lep1_p4.E();
            float lep_pz = lep1_p4.Pz();
            float disc_top_sqr = lambda_top*lambda_top*lep_pz*lep_pz/(lep_pt*lep_pt*lep_pt*lep_pt) - (lep_E*lep_E*PuppiMET_pt*PuppiMET_pt - lambda_top*lambda_top)/(lep_pt*lep_pt);
            return disc_top_sqr;
        """,
    )

    df = df.Define(
        "nuFromT_pz_poz",
        """
            float lep_pt = lep1_p4.Pt();
            float lep_E = lep1_p4.E();
            float lep_pz = lep1_p4.Pz();
            if (disc_top_sqr > 0)
                return static_cast<float>(lambda_top*lep_pz/(lep_pt*lep_pt) + std::sqrt(disc_top_sqr));
            else
                return static_cast<float>(lambda_top*lep_pz/(lep_pt*lep_pt));
        """,
    )

    df = df.Define(
        "nuFromT_pz_neg",
        """
            float lep_pt = lep1_p4.Pt();
            float lep_E = lep1_p4.E();
            float lep_pz = lep1_p4.Pz();
            if (disc_top_sqr > 0)
                return static_cast<float>(lambda_top*lep_pz/(lep_pt*lep_pt) - std::sqrt(disc_top_sqr));
            else
                return static_cast<float>(lambda_top*lep_pz/(lep_pt*lep_pt));
        """,
    )

    df = df.Define(
        "nuFromT_E_pos",
        "return std::sqrt(PuppiMET_pt*PuppiMET_pt + nuFromT_pz_poz*nuFromT_pz_poz);",
    )
    df = df.Define(
        "nuFromT_E_neg",
        "return std::sqrt(PuppiMET_pt*PuppiMET_pt + nuFromT_pz_neg*nuFromT_pz_neg);",
    )

    df = df.Define(
        "nuFromT_pos_p4",
        "return LorentzVectorXYZ(PuppiMET_p4.Px(), PuppiMET_p4.Py(), nuFromT_pz_poz, nuFromT_E_pos);",
    )
    df = df.Define(
        "nuFromT_neg_p4",
        "return LorentzVectorXYZ(PuppiMET_p4.Px(), PuppiMET_p4.Py(), nuFromT_pz_neg, nuFromT_E_neg);",
    )

    df = df.Define("lepWFromT_pos_p4", "return lep1_p4 + nuFromT_pos_p4;")
    df = df.Define("lepWFromT_neg_p4", "return lep1_p4 + nuFromT_neg_p4;")

    # define collection of p4s of top quark candidates
    # will have 3 elements:
    #   0. hadronic top candidate from t->bW->bqq decay
    #   1. leptonic top candidate with lepWFromT_pos_p4
    #   2. leptonic top candidate with lepWFromT_neg_p4
    # each element 0-3 is a collection itself and it contains
    # constituents of top candidate, i.e.
    #   0. b-tagged object (jet or fatjet)
    #   1. leptonic W (in case of t->bW->blv) or jet (in case t->bW->bqq)
    #   2. jet (in case t->bW->bqq) or nothing (in case of t->bW->blv)
    df = df.Define(
        "tops",
        f"""
            RVecVec<LorentzVectorM> tops(3, RVecLV{{}});
            if ((resolved || res2b) && !DL)
            {{
                float lep_bjet1_dr = ROOT::Math::VectorUtil::DeltaR(lep1_p4, bjet1_p4);
                float lep_bjet2_dr = ROOT::Math::VectorUtil::DeltaR(lep1_p4, bjet2_p4);
                if (lep_bjet1_dr < lep_bjet2_dr)
                {{
                    tops[0] = {{bjet2_p4, wjet1_p4, wjet2_p4}};
                    tops[1] = {{bjet1_p4, lepWFromT_pos_p4}};
                    tops[2] = {{bjet1_p4, lepWFromT_neg_p4}};
                }}
                else
                {{
                    tops[0] = {{bjet1_p4, wjet1_p4, wjet2_p4}};
                    tops[1] = {{bjet2_p4, lepWFromT_pos_p4}};
                    tops[2] = {{bjet2_p4, lepWFromT_neg_p4}};
                }}
            }}
            else if (boosted)
            {{
                if (fatwjet_isValid && fatbjet_isValid)
                {{

                    if (bjet1_isValid && bjet2_isValid)
                    {{
                        std::vector<LorentzVectorM> bcands = {{bjet1_p4, bjet2_p4, fatbjet_p4}};
                        auto dr_cmp_lep = [&lep1_p4](LorentzVectorM const& v1, LorentzVectorM const& v2){{
                            return ROOT::Math::VectorUtil::DeltaR(v1, lep1_p4) < ROOT::Math::VectorUtil::DeltaR(v2, lep1_p4);
                        }};
                        auto it = std::min_element(bcands.begin(), bcands.end(), dr_cmp_lep);
                        size_t bcand_from_lep_top_idx = it - bcands.begin();

                        LorentzVectorM hadW_p4 = (wjet1_isValid && wjet2_isValid && !WJets_Boosted) ? (wjet1_p4 + wjet2_p4) : fatwjet_p4;
                        size_t bcand_from_had_top_idx = (bcand_from_lep_top_idx == 0) ? 1 : 0;
                        float min_dr = ROOT::Math::VectorUtil::DeltaR(bcands[bcand_from_had_top_idx], hadW_p4);

                        for (size_t i = 0; i < 3; ++i)
                        {{
                            float dr = ROOT::Math::VectorUtil::DeltaR(bcands[i], hadW_p4);
                            if (i != bcand_from_lep_top_idx && dr < min_dr)
                            {{
                                min_dr = dr;
                                bcand_from_had_top_idx = i;
                            }}
                        }}

                        if (wjet1_isValid && wjet2_isValid && !WJets_Boosted)
                            tops[0] = {{bcands[bcand_from_had_top_idx], wjet1_p4, wjet2_p4}};
                        else
                            tops[0] = {{bcands[bcand_from_had_top_idx], fatwjet_p4}};
                        tops[1] = {{bcands[bcand_from_lep_top_idx], lepWFromT_pos_p4}};
                        tops[2] = {{bcands[bcand_from_lep_top_idx], lepWFromT_neg_p4}};
                    }}
                    else if (bjet1_isValid || bjet2_isValid)
                    {{
                        auto bjet_p4 = bjet1_isValid ? bjet1_p4 : bjet2_p4;
                        float lep_bjet_dr = ROOT::Math::VectorUtil::DeltaR(lep1_p4, bjet_p4);
                        float lep_fatbjet_dr = ROOT::Math::VectorUtil::DeltaR(lep1_p4, fatbjet_p4);
                        if (lep_bjet_dr < lep_fatbjet_dr)
                        {{
                            if (wjet1_isValid && wjet2_isValid && !WJets_Boosted)
                                tops[0] = {{fatbjet_p4, wjet1_p4, wjet2_p4}};
                            else
                                tops[0] = {{fatbjet_p4, fatwjet_p4 }};
                            tops[1] = {{bjet_p4, lepWFromT_pos_p4}};
                            tops[2] = {{bjet_p4, lepWFromT_neg_p4}};
                        }}
                        else
                        {{
                            if (wjet1_isValid && wjet2_isValid && !WJets_Boosted)
                                tops[0] = {{bjet_p4, wjet1_p4, wjet2_p4}};
                            else
                                tops[0] = {{bjet_p4, fatwjet_p4}};
                            tops[1] = {{fatbjet_p4, lepWFromT_pos_p4}};
                            tops[2] = {{fatbjet_p4, lepWFromT_neg_p4}};
                        }}
                    }}
                }}
                else if (!fatwjet_isValid && fatbjet_isValid)
                {{
                    if (bjet1_isValid && bjet2_isValid)
                    {{
                        if (!wjet1_isValid || !wjet2_isValid)
                        {{
                            return tops;
                        }}

                        LorentzVectorM hadW_p4 = wjet1_p4 + wjet2_p4;

                        std::vector<LorentzVectorM> bcands = {{bjet1_p4, bjet2_p4, fatbjet_p4}};
                        auto dr_cmp_lep = [&lep1_p4](LorentzVectorM const& v1, LorentzVectorM const& v2){{
                            return ROOT::Math::VectorUtil::DeltaR(v1, lep1_p4) < ROOT::Math::VectorUtil::DeltaR(v2, lep1_p4);
                        }};
                        auto it = std::min_element(bcands.begin(), bcands.end(), dr_cmp_lep);
                        size_t bcand_from_lep_top_idx = it - bcands.begin();

                        size_t bcand_from_had_top_idx = (bcand_from_lep_top_idx == 0) ? 1 : 0;
                        float min_dr = ROOT::Math::VectorUtil::DeltaR(bcands[bcand_from_had_top_idx], hadW_p4);

                        for (size_t i = 0; i < 3; ++i)
                        {{
                            float dr = ROOT::Math::VectorUtil::DeltaR(bcands[i], hadW_p4);
                            if (i != bcand_from_lep_top_idx && dr < min_dr)
                            {{
                                min_dr = dr;
                                bcand_from_had_top_idx = i;
                            }}
                        }}

                        tops[0] = {{bcands[bcand_from_had_top_idx], wjet1_p4, wjet2_p4}};
                        tops[1] = {{bcands[bcand_from_lep_top_idx], lepWFromT_pos_p4}};
                        tops[2] = {{bcands[bcand_from_lep_top_idx], lepWFromT_neg_p4}};
                    }}
                    else if (bjet1_isValid || bjet2_isValid)
                    {{
                        auto bjet_p4 = bjet1_isValid ? bjet1_p4 : bjet2_p4;
                        if (wjet1_isValid && wjet2_isValid)
                        {{
                            float lep_bjet_dr = ROOT::Math::VectorUtil::DeltaR(lep1_p4, bjet_p4);
                            float lep_fatbjet_dr = ROOT::Math::VectorUtil::DeltaR(lep1_p4, fatbjet_p4);
                            if (lep_bjet_dr < lep_fatbjet_dr)
                            {{
                                tops[0] = {{fatbjet_p4, wjet1_p4, wjet2_p4}};
                                tops[1] = {{bjet_p4, lepWFromT_pos_p4}};
                                tops[2] = {{bjet_p4, lepWFromT_neg_p4}};
                            }}
                            else
                            {{
                                tops[0] = {{bjet_p4, wjet1_p4, wjet2_p4}};
                                tops[1] = {{fatbjet_p4, lepWFromT_pos_p4}};
                                tops[2] = {{fatbjet_p4, lepWFromT_neg_p4}};
                            }}
                        }}
                    }}
                }}
                else if (fatwjet_isValid && !fatbjet_isValid)
                {{
                    if (bjet1_isValid && bjet2_isValid)
                    {{
                        float lep_bjet1_dr = ROOT::Math::VectorUtil::DeltaR(lep1_p4, bjet1_p4);
                        float lep_bjet2_dr = ROOT::Math::VectorUtil::DeltaR(lep1_p4, bjet2_p4);
                        if (lep_bjet1_dr < lep_bjet2_dr)
                        {{
                            tops[0] = {{bjet2_p4, fatwjet_p4}};
                            tops[1] = {{bjet1_p4, lepWFromT_pos_p4}};
                            tops[2] = {{bjet1_p4, lepWFromT_neg_p4}};
                        }}
                        else
                        {{
                            tops[0] = {{bjet1_p4, fatwjet_p4}};
                            tops[1] = {{bjet2_p4, lepWFromT_pos_p4}};
                            tops[2] = {{bjet2_p4, lepWFromT_neg_p4}};
                        }}
                    }}
                }}
            }}
            return tops;
        """,
    )

    # now define p4 of each top
    df = df.Define(
        "hadT_p4",
        """
            LorentzVectorM res;
            for (auto const& v: tops[0])
                res += v;
            return res;
        """,
    )
    df = df.Define(
        "lepT_pos_p4",
        """
            LorentzVectorM res;
            for (auto const& v: tops[1])
                res += v;
            return res;
        """,
    )
    df = df.Define(
        "lepT_neg_p4",
        """
            LorentzVectorM res;
            for (auto const& v: tops[2])
                res += v;
            return res;
        """,
    )

    # boolean indicating which solution (positive or negative) was picked for leptonic top candidate
    df = df.Define(
        "top_solution_tag",
        """
            float dphi_pos = std::abs(ROOT::Math::VectorUtil::DeltaPhi(hadT_p4, lepT_pos_p4));
            float dphi_neg = std::abs(ROOT::Math::VectorUtil::DeltaPhi(hadT_p4, lepT_neg_p4));
            return dphi_pos > dphi_neg;
        """,
    )

    df = df.Define("lepT_p4", "return top_solution_tag ? lepT_pos_p4 : lepT_neg_p4;")
    df = df.Define(
        "lepWfromT_p4", "return top_solution_tag ? lepWFromT_pos_p4 : lepWFromT_neg_p4;"
    )

    return df


def defineTopVariables(df):
    """
    Function computing top candidate related variables for the SL channel.
    Requires top candidate's p4 to be computed and defined.
    """
    df = df.Define("hadT_mass", "return hadT_p4.M();")
    df = df.Define("lepT_mass", "return lepT_p4.M();")

    df = df.Define("hadT_pt", "return hadT_p4.Pt();")
    df = df.Define("lepT_pt", "return lepT_p4.Pt();")

    df = df.Define(
        "hadT_constituentPtFrac",
        """
            if (tops[0].empty())
                return 0.0f;
            float sum_pt = 0.0f;
            for (auto const& p: tops[0])
                sum_pt += p.Pt();
            return static_cast<float>(hadT_p4.Pt()/sum_pt);
        """,
    )

    df = df.Define(
        "lepT_constituentPtFrac",
        """
            if (top_solution_tag)
                return static_cast<float>(tops[1].empty() ? 0.0f : lepT_p4.Pt()/(tops[1][0].Pt() + PuppiMET_pt + lep1_pt));
            else
                return static_cast<float>(tops[2].empty() ? 0.0f : lepT_p4.Pt()/(tops[2][0].Pt() + PuppiMET_pt + lep1_pt));
        """,
    )

    df = df.Define(
        "TT_mass",
        "return static_cast<float>((lepT_p4.Pt() > 0.0f && hadT_p4.Pt() > 0.0f) ? (lepT_p4 + hadT_p4).M() : 0.0f);",
    )

    df = df.Define(
        "lepT_mT",
        f"""
            VectorXY<float> nu_t;
            LorentzVectorM bjet_p4;
            if (top_solution_tag)
            {{
                nu_t = VectorXY<float>(nuFromT_pos_p4.Px(), nuFromT_pos_p4.Py());
                bjet_p4 = tops[1].empty() ? LorentzVectorM() : tops[1][0];
            }}
            else
            {{
                nu_t = VectorXY<float>(nuFromT_neg_p4.Px(), nuFromT_neg_p4.Py());
                bjet_p4 = tops[2].empty() ? LorentzVectorM() : tops[2][0];
            }}
            VectorXY<float> lep1_t = VectorXY<float>(lep1_p4.Px(), lep1_p4.Py());
            VectorXY<float> bjet_t = VectorXY<float>(bjet_p4.Px(), bjet_p4.Py());
            VectorXY<float> total_transverse_momentum = nu_t + lep1_t + bjet_t;

            float total_transverse_energy = std::sqrt(nu_t.Mag2())
                                          + std::sqrt(lep1_t.Mag2() + lep1_p4.M2())
                                          + std::sqrt(bjet_t.Mag2() + bjet_p4.M2());

            float mt_square = total_transverse_energy*total_transverse_energy - total_transverse_momentum.Mag2();
            return static_cast<float>(mt_square > 0.0f ? std::sqrt(mt_square) : -1.0f);
        """,
    )

    df = df.Define(
        "hadT_hadW_dphi", "return ROOT::Math::VectorUtil::DeltaPhi(hadT_p4, hadW_p4);"
    )
    df = df.Define("hadT_hadW_deta", "return hadT_p4.Eta() - hadW_p4.Eta();")
    df = df.Define(
        "hadT_hadW_dR", "return ROOT::Math::VectorUtil::DeltaR(hadT_p4, hadW_p4);"
    )

    df = df.Define(
        "hadW_lepWfromT_dphi",
        "return ROOT::Math::VectorUtil::DeltaPhi(lepWfromT_p4, hadW_p4);",
    )
    df = df.Define("hadW_lepWfromT_deta", "return lepWfromT_p4.Eta() - hadW_p4.Eta();")
    df = df.Define(
        "hadW_lepWfromT_dR",
        "return ROOT::Math::VectorUtil::DeltaR(lepWfromT_p4, hadW_p4);",
    )

    # df = df.Define("Hbb_lepWfromT_dphi", "return ROOT::Math::VectorUtil::DeltaPhi(lepWfromT_p4, Hbb_p4);")
    # df = df.Define("Hbb_lepWfromT_deta", "return lepWfromT_p4.Eta() - Hbb_p4.Eta();")
    # df = df.Define("Hbb_lepWfromT_dR", "return ROOT::Math::VectorUtil::DeltaR(lepWfromT_p4, Hbb_p4);")

    df = df.Define(
        "bjet_lepWfromT_minDphi",
        """
            RVecF dphis;
            if (fatbjet_isValid)
                dphis.push_back(ROOT::Math::VectorUtil::DeltaPhi(lepWfromT_p4, fatbjet_p4));
            if (bjet1_isValid)
                dphis.push_back(ROOT::Math::VectorUtil::DeltaPhi(lepWfromT_p4, bjet1_p4));
            if (bjet2_isValid)
                dphis.push_back(ROOT::Math::VectorUtil::DeltaPhi(lepWfromT_p4, bjet2_p4));

            auto it = std::min_element(dphis.begin(), dphis.end());
            if (it != dphis.end())
                return static_cast<float>(*it);
            return 5.0f;
        """,
    )

    df = df.Define(
        "bjet_lep_mass",
        """
            if (top_solution_tag)
                return static_cast<float>(tops[1].empty() ? -1.0f : (tops[1][0] + lep1_p4).M());
            else
                return static_cast<float>(tops[2].empty() ? -1.0f : (tops[2][0] + lep1_p4).M());
            return -1.0f;
        """,
    )

    return df


def defineFeatureValidityFlags(df):
    """
    Function computing computing masks indicating in which event certain
    high-level features are valid. Requires features to be already defined.
    Is needed for proper feature normalization.
    """
    df = df.Define("hadT_mass_valid", "return static_cast<int>(hadT_p4.M() > 0.0f);")
    df = df.Define("lepT_mass_valid", "return static_cast<int>(lepT_p4.M() > 0.0f);")
    df = df.Define("hadT_pt_valid", "return static_cast<int>(hadT_p4.Pt() > 0.0f);")
    df = df.Define("lepT_pt_valid", "return static_cast<int>(lepT_p4.Pt() > 0.0f);")
    df = df.Define(
        "hadT_constituentPtFrac_valid",
        "return static_cast<int>(hadT_constituentPtFrac > 0.0f);",
    )
    df = df.Define(
        "lepT_constituentPtFrac_valid",
        "return static_cast<int>(lepT_constituentPtFrac > 0.0f);",
    )
    df = df.Define("TT_mass_valid", "return static_cast<int>(TT_mass > 0.0f);")
    df = df.Define("lepT_mT_valid", "return static_cast<int>(lepT_mT > 0.0f);")

    df = df.Define(
        "hadT_hadW_dphi_valid",
        "return static_cast<int>(WhadCand_isValid && hadT_pt_valid);",
    )
    df = df.Define(
        "hadT_hadW_deta_valid",
        "return static_cast<int>(WhadCand_isValid && hadT_pt_valid);",
    )
    df = df.Define(
        "hadT_hadW_dR_valid",
        "return static_cast<int>(WhadCand_isValid && hadT_pt_valid);",
    )

    df = df.Define(
        "hadW_leadbjet_dphi_valid",
        "return static_cast<int>(WhadCand_isValid && (leadbjet_p4.Pt() > 0.0f));",
    )
    df = df.Define(
        "hadW_leadbjet_deta_valid",
        "return static_cast<int>(WhadCand_isValid && (leadbjet_p4.Pt() > 0.0f));",
    )
    df = df.Define(
        "hadW_leadbjet_dR_valid",
        "return static_cast<int>(WhadCand_isValid && (leadbjet_p4.Pt() > 0.0f));",
    )

    df = df.Define("hadW_mass_valid", "return static_cast<int>(hadW_mass > 0.0f);")

    df = df.Define(
        "bb_dphi_valid", "return static_cast<int>(bjet1_isValid && bjet2_isValid);"
    )
    df = df.Define(
        "bb_deta_valid", "return static_cast<int>(bjet1_isValid && bjet2_isValid);"
    )
    df = df.Define("bb_pt_valid", "return static_cast<int>(HbbCand_isValid);")

    df = df.Define("hadW_lep_dphi_valid", "return static_cast<int>(WhadCand_isValid);")
    df = df.Define("hadW_lep_dR_valid", "return static_cast<int>(WhadCand_isValid);")
    df = df.Define("hadW_lep_deta_valid", "return static_cast<int>(WhadCand_isValid);")

    df = df.Define(
        "hadW_lepWfromT_dphi_valid", "return static_cast<int>(WhadCand_isValid);"
    )
    df = df.Define(
        "hadW_lepWfromT_deta_valid", "return static_cast<int>(WhadCand_isValid);"
    )
    df = df.Define(
        "hadW_lepWfromT_dR_valid", "return static_cast<int>(WhadCand_isValid);"
    )
    df = df.Define(
        "hadW_lepWfromH_dphi_valid", "return static_cast<int>(WhadCand_isValid);"
    )
    df = df.Define(
        "hadW_lepWfromH_deta_valid", "return static_cast<int>(WhadCand_isValid);"
    )
    df = df.Define(
        "hadW_lepWfromH_dR_valid", "return static_cast<int>(WhadCand_isValid);"
    )

    # df = df.Define("Hbb_lepWfromT_dphi_valid", "return static_cast<int>(HbbCand_isValid);")
    # df = df.Define("Hbb_lepWfromT_deta_valid", "return static_cast<int>(HbbCand_isValid);")
    # df = df.Define("Hbb_lepWfromT_dR_valid", "return static_cast<int>(HbbCand_isValid);")
    df = df.Define(
        "Hbb_lepWfromH_dphi_valid", "return static_cast<int>(HbbCand_isValid);"
    )
    df = df.Define(
        "Hbb_lepWfromH_deta_valid", "return static_cast<int>(HbbCand_isValid);"
    )
    df = df.Define(
        "Hbb_lepWfromH_dR_valid", "return static_cast<int>(HbbCand_isValid);"
    )

    df = df.Define(
        "bjet_lep_minDr_valid", "return static_cast<int>(bjet_lep_minDr > 0.0f);"
    )
    df = df.Define(
        "bjet_lep_minDphi_valid",
        "return static_cast<int>(bjet1_isValid || bjet2_isValid || fatbjet_isValid);",
    )

    df = df.Define(
        "bjet_lepWfromT_minDphi_valid",
        "return static_cast<int>(bjet1_isValid || bjet2_isValid || fatbjet_isValid);",
    )
    df = df.Define(
        "bjet_lepWfromH_minDphi_valid",
        "return static_cast<int>(bjet1_isValid || bjet2_isValid || fatbjet_isValid);",
    )

    df = df.Define("WW_pt_valid", "return static_cast<int>(WhadCand_isValid);")
    df = df.Define("WW_ptToE_valid", "return static_cast<int>(WhadCand_isValid);")
    df = df.Define("bb_ptToE_valid", "return static_cast<int>(bb_ptToE > 0.0f);")

    df = df.Define(
        "fatbjet_2prong_valid",
        "return static_cast<int>(fatbjet_isValid && fatbjet_tau1 > 0.0)",
    )
    df = df.Define(
        "fatbjet_3prong_valid",
        "return static_cast<int>(fatbjet_isValid && fatbjet_tau2 > 0.0)",
    )

    return df


def defineLepWCandP4(df):
    "Function computing p4 of leptonic W from H->WW decay assuming higgs mass constraint"

    df = df.Define("mWhadLep_p4", "return hadW_p4 + lep1_p4;")

    df = df.Define(
        "lambda_higgs",
        """
            float mh = 125.0f;
            float mWhadLep = mWhadLep_p4.M();
            float lambda_higgs = (mh*mh - mWhadLep*mWhadLep)/2 + PuppiMET_p4.Px()*mWhadLep_p4.Px() + PuppiMET_p4.Py()*mWhadLep_p4.Py();
            return lambda_higgs;
        """,
    )
    df = df.Define(
        "a",
        "return mWhadLep_p4.Pz()*mWhadLep_p4.Pz() - mWhadLep_p4.E()*mWhadLep_p4.E();",
    )
    df = df.Define("b", "return 2*lambda_higgs*mWhadLep_p4.Pz();")
    df = df.Define(
        "c",
        "return lambda_higgs*lambda_higgs - mWhadLep_p4.E()*mWhadLep_p4.E()*PuppiMET_pt*PuppiMET_pt;",
    )
    df = df.Define("disc_higgs_sqr", "return b*b - 4*a*c;")

    df = df.Define(
        "nuFromH_pz_poz",
        """
            float const eps = 1e-6f;
            if (std::abs(a) < eps)
                return (std::abs(b) > eps) ? static_cast<float>(-c/b) : 0.0f;
            if (disc_higgs_sqr > 0)
                return static_cast<float>(-b/(2*a) + std::sqrt(disc_higgs_sqr)/(2*a));
            else
                return static_cast<float>(-b/(2*a));
        """,
    )

    df = df.Define(
        "nuFromH_pz_neg",
        """
            float const eps = 1e-6f;
            if (std::abs(a) < eps)
                return (std::abs(b) > eps) ? static_cast<float>(-c/b) : 0.0f;
            if (disc_higgs_sqr > 0)
                return static_cast<float>(-b/(2*a) - std::sqrt(disc_higgs_sqr)/(2*a));
            else
                return static_cast<float>(-b/(2*a));
        """,
    )

    df = df.Define(
        "nuFromH_E_pos",
        "return std::sqrt(PuppiMET_pt*PuppiMET_pt + nuFromH_pz_poz*nuFromH_pz_poz);",
    )
    df = df.Define(
        "nuFromH_E_neg",
        "return std::sqrt(PuppiMET_pt*PuppiMET_pt + nuFromH_pz_neg*nuFromH_pz_neg);",
    )

    df = df.Define(
        "nuFromH_pos_p4",
        "return LorentzVectorXYZ(PuppiMET_p4.Px(), PuppiMET_p4.Py(), nuFromH_pz_poz, nuFromH_E_pos);",
    )
    df = df.Define(
        "nuFromH_neg_p4",
        "return LorentzVectorXYZ(PuppiMET_p4.Px(), PuppiMET_p4.Py(), nuFromH_pz_neg, nuFromH_E_neg);",
    )

    df = df.Define("lepWFromH_pos_p4", "return lep1_p4 + nuFromH_pos_p4;")
    df = df.Define("lepWFromH_neg_p4", "return lep1_p4 + nuFromH_neg_p4;")

    df = df.Define("Hww_pos_p4", "return lepWFromH_pos_p4 + hadW_p4;")
    df = df.Define("Hww_neg_p4", "return lepWFromH_neg_p4 + hadW_p4;")

    df = df.Define(
        "higgs_solution_tag",
        """
            float dphi_pos = std::abs(ROOT::Math::VectorUtil::DeltaPhi(Hbb_p4, Hww_pos_p4));
            float dphi_neg = std::abs(ROOT::Math::VectorUtil::DeltaPhi(Hbb_p4, Hww_neg_p4));
            return dphi_pos > dphi_neg;
        """,
    )

    df = df.Define(
        "lepWfromH_p4",
        "return higgs_solution_tag ? lepWFromH_pos_p4 : lepWFromH_neg_p4;",
    )
    df = df.Define("Hww_p4", "return higgs_solution_tag ? Hww_pos_p4 : Hww_neg_p4;")

    return df


def AddDNNVariablesSL(df, isData=False):
    # single lepton features
    df = df.Define(
        "hadW_leadbjet_dphi",
        "return ROOT::Math::VectorUtil::DeltaPhi(hadW_p4, leadbjet_p4);",
    )
    df = df.Define("hadW_leadbjet_deta", "return hadW_p4.Eta() - leadbjet_p4.Eta();")
    df = df.Define(
        "hadW_leadbjet_dR",
        "return ROOT::Math::VectorUtil::DeltaR(hadW_p4, leadbjet_p4);",
    )
    df = df.Define("hadW_mass", "return hadW_p4.M();")
    df = df.Define(
        "hadW_lep_dphi", "return ROOT::Math::VectorUtil::DeltaPhi(hadW_p4, lep1_p4);"
    )
    df = df.Define(
        "hadW_lep_dR", "return ROOT::Math::VectorUtil::DeltaR(hadW_p4, lep1_p4);"
    )
    df = df.Define("hadW_lep_deta", "return hadW_p4.Eta() - lep1_p4.Eta();")

    df = df.Define(
        "hadW_lepWfromH_dphi",
        "return ROOT::Math::VectorUtil::DeltaPhi(hadW_p4, lepWfromH_p4);",
    )
    df = df.Define(
        "hadW_lepWfromH_dR",
        "return ROOT::Math::VectorUtil::DeltaR(hadW_p4, lepWfromH_p4);",
    )
    df = df.Define("hadW_lepWfromH_deta", "hadW_p4.Eta() - lepWfromH_p4.Eta();")

    df = df.Define(
        "Hbb_lepWfromH_dphi",
        "return ROOT::Math::VectorUtil::DeltaPhi(lepWfromH_p4, Hbb_p4);",
    )
    df = df.Define("Hbb_lepWfromH_deta", "return lepWfromH_p4.Eta() - Hbb_p4.Eta();")
    df = df.Define(
        "Hbb_lepWfromH_dR",
        "return ROOT::Math::VectorUtil::DeltaR(lepWfromH_p4, Hbb_p4);",
    )

    df = df.Define(
        "lep1_bjets_minDr",
        """
            RVecF drs;
            if (fatbjet_isValid)
                drs.push_back(ROOT::Math::VectorUtil::DeltaR(lep1_p4, fatbjet_p4));
            if (bjet1_isValid)
                drs.push_back(ROOT::Math::VectorUtil::DeltaR(lep1_p4, bjet1_p4));
            if (bjet2_isValid)
                drs.push_back(ROOT::Math::VectorUtil::DeltaR(lep1_p4, bjet2_p4));

            auto it = std::min_element(drs.begin(), drs.end());
            if (it != drs.end())
                return static_cast<float>(*it);
            return -1.0f;
        """,
    )

    df = df.Define(
        "lep1_bjets_minDphi",
        """
            RVecF dphis;
            if (fatbjet_isValid)
                dphis.push_back(ROOT::Math::VectorUtil::DeltaPhi(lep1_p4, fatbjet_p4));
            if (bjet1_isValid)
                dphis.push_back(ROOT::Math::VectorUtil::DeltaPhi(lep1_p4, bjet1_p4));
            if (bjet2_isValid)
                dphis.push_back(ROOT::Math::VectorUtil::DeltaPhi(lep1_p4, bjet2_p4));

            auto it = std::min_element(dphis.begin(), dphis.end());
            if (it != dphis.end())
                return static_cast<float>(*it);
            return 5.0f;
        """,
    )

    df = df.Define(
        "bjet_lepWfromH_minDphi",
        """
            RVecF dphis;
            if (fatbjet_isValid)
                dphis.push_back(ROOT::Math::VectorUtil::DeltaPhi(lepWfromH_p4, fatbjet_p4));
            if (bjet1_isValid)
                dphis.push_back(ROOT::Math::VectorUtil::DeltaPhi(lepWfromH_p4, bjet1_p4));
            if (bjet2_isValid)
                dphis.push_back(ROOT::Math::VectorUtil::DeltaPhi(lepWfromH_p4, bjet2_p4));

            auto it = std::min_element(dphis.begin(), dphis.end());
            if (it != dphis.end())
                return static_cast<float>(*it);
            return 5.0f;
        """,
    )

    df = df.Define("WW_pt", "return static_cast<float>(Hww_p4.Pt());")
    df = df.Define(
        "WW_ptToE",
        "return static_cast<float>(Hww_p4.E() > 0.0 ? WW_pt/Hww_p4.E() : -1.0f);",
    )
    df = df.Define(
        "bb_ptToE",
        """
            if (fatbjet_isValid)
                return static_cast<float>(bb_pt/fatbjet_p4.E());
            else if (bjet1_isValid && bjet2_isValid)
                return static_cast<float>(bb_pt/(bjet1_p4 + bjet2_p4).E());
            return -1.0f;
        """,
    )

    df = df.Define(
        "fatbjet_2prong",
        """
            if (fatbjet_isValid)
                return fatbjet_tau1 > 0.0 ? fatbjet_tau2/fatbjet_tau1 : -1.0;
            return -1.0;
        """,
    )

    df = df.Define(
        "fatbjet_3prong",
        """
            if (fatbjet_isValid)
                return fatbjet_tau2 > 0.0 ? fatbjet_tau3/fatbjet_tau2 : -1.0;
            return -1.0;
        """,
    )
    return df


def AddDNNVariablesCommon(df, isData=False):
    df = df.Define("HT", f"Sum(centralJet_pt)")

    df = df.Define(
        "bb_dR",
        f"ROOT::Math::VectorUtil::DeltaR(bjet1_p4, bjet2_p4)",
    )

    df = df.Define(
        "met_bb_dphi",
        f"ROOT::Math::VectorUtil::DeltaPhi(PuppiMET_p4,(bjet1_p4+bjet2_p4))",
    )
    df = df.Define("min_dR_lep0_jets", f"MinDeltaR(lep1_p4, centralJet_p4)")
    df = df.Define("min_dR_lep1_jets", f"MinDeltaR(lep2_p4, centralJet_p4)")

    df = df.Define(
        "MT",
        """
            if (lep1_legType > 0 && lep2_legType > 0)
                return static_cast<float>(Calculate_TotalMT(lep1_p4, lep2_p4, PuppiMET_p4));
            else if (lep1_legType > 0 && lep2_legType < 1)
                return static_cast<float>(Calculate_MT(lep1_p4, PuppiMET_p4));
            return -1.0f;
        """,
    )

    df = df.Define(
        "bb_dphi", "return ROOT::Math::VectorUtil::DeltaPhi(bjet1_p4, bjet2_p4);"
    )
    df = df.Define("bb_deta", "return bjet1_p4.Eta() - bjet2_p4.Eta();")
    df = df.Define(
        "bb_pt",
        """
            if (fatbjet_isValid)
                return static_cast<float>(fatbjet_p4.Pt());
            else if (bjet1_isValid && bjet2_isValid)
                return static_cast<float>((bjet1_p4 + bjet2_p4).Pt());
            return 0.0f;
        """,
    )

    df = df.Define(
        "MET_over_sqrt_HT",
        "HT > 0.0 ? PuppiMET_pt / sqrt(HT) : 0.0",
    )

    df = df.Define("bb_btag_product", "bjet1_btagPNetB * bjet2_btagPNetB")

    return df


def addDeepHMERelErr(df):
    # this function can only be called if DeepHME is loaded to dataframe
    # it is a placeholder for know
    # calling without DeepHME will result in a crash
    df = df.Define(
        "DeepHME_mass_rel_error",
        "DeepHME_mass > 0.0 ? static_cast<float>(DeepHME_mass_error)/static_cast<float>(DeepHME_mass) : -1.0f;",
    )
    return df


def PrepareDfForHistograms(dfForHistograms, isData, stage):
    dfForHistograms.defineLeptonChannel()
    dfForHistograms.df = defineAllP4(dfForHistograms.df)
    dfForHistograms.calculateMT()
    dfForHistograms.df = defineJetSelections(
        dfForHistograms.df,
        isData,
        getattr(dfForHistograms, "period", "Run3_2023BPix"),
    )
    dfForHistograms.df = AddDNNVariablesCommon(dfForHistograms.df, isData)
    dfForHistograms.df = AddDNNVariablesDL(dfForHistograms.df, isData)
    dfForHistograms.defineTriggers()
    dfForHistograms.defineLeptonPreselection()
    dfForHistograms.defineQCDRegions()
    dfForHistograms.defineControlRegions(stage)
    dfForHistograms.defineCategories()
    dfForHistograms.df = defineTopCandP4(dfForHistograms.df)
    dfForHistograms.df = defineLepWCandP4(dfForHistograms.df)
    dfForHistograms.df = AddDNNVariablesSL(dfForHistograms.df, isData)
    dfForHistograms.df = defineTopVariables(dfForHistograms.df)
    # I comment this out now to save computation time
    # dfForHistograms.df = defineFeatureValidityFlags(dfForHistograms.df)
    # this is a placeholder, calling without DeepHME loaded will result in a crash
    # also needs to be disabled for CI
    # dfForHistograms.df = addDeepHMERelErr(dfForHistograms.df)
    dfForHistograms.defineCutFlow()
    return dfForHistograms
