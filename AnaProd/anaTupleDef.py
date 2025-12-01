import AnaProd.baseline as AnaBaseline
import FLAF.Common.BaselineSelection as CommonBaseline
from Corrections.Corrections import Corrections

loadTF = False
loadHHBtag = False
lepton_legs = ["lep1", "lep2"]
offline_legs = ["lep1", "lep2"]


Muon_int_observables = [
    "Muon_mediumId",
    "Muon_tightId",
    "Muon_highPtId",
    "Muon_pfIsoId",
    "Muon_mediumPromptId",
    "Muon_miniIsoId",
    "Muon_mvaMuID_WP",
]
Muon_float_observables = [
    "Muon_tkRelIso",
    "Muon_pfRelIso04_all",
    "Muon_pfRelIso03_all",
    "Muon_miniPFRelIso_all",
]
Muon_observables = Muon_int_observables + Muon_float_observables
Electron_int_observables = ["Electron_mvaNoIso_WP80", "Electron_mvaIso_WP80"]
Electron_float_observables = [
    "Electron_pfRelIso03_all",
    "Electron_mvaIso",
    "Electron_mvaNoIso",
    "Electron_miniPFRelIso_all",
]
Electron_observables = Electron_int_observables + Electron_float_observables
JetObservables = [
    "PNetRegPtRawCorr",
    "PNetRegPtRawCorrNeutrino",
    "PNetRegPtRawRes",
    "rawFactor",
    "btagPNetB",
    "btagPNetCvB",
    "btagPNetCvL",
    "btagPNetCvNotB",
    "btagPNetQvG",
    "btagPNetTauVJet",
    "chEmEF",
    "chHEF",
    "chMultiplicity",
    "hfEmEF",
    "hfHEF",
    "hfadjacentEtaStripsSize",
    "hfcentralEtaStripSize",
    "hfsigmaEtaEta",
    "hfsigmaPhiPhi",
    "jetId",
    "muEF",
    "muonSubtrFactor",
    "nConstituents",
    "nElectrons",
    "nMuons",
    "nSVs",
    "neEmEF",
    "neHEF",
    "neMultiplicity",
    "ptRes",
    "idbtagPNetB",
    "area",
]  # 2024

JetObservablesMC = ["hadronFlavour", "partonFlavour"]

FatJetObservables = [
    "area",
    "chEmEF",
    "chHEF",
    "chMultiplicity",
    "globalParT2_QCD0HF",
    "globalParT2_QCD1HF",
    "globalParT2_QCD2HF",
    "globalParT2_TopW",
    "globalParT2_TopbW",
    "globalParT2_TopbWev",
    "globalParT2_TopbWmv",
    "globalParT2_TopbWq",
    "globalParT2_TopbWqq",
    "globalParT2_TopbWtauhv",
    "globalParT2_Xbb",
    "globalParT2_XbbVsQCD",
    "globalParT2_Xcc",
    "globalParT2_Xcs",
    "globalParT2_Xgg",
    "globalParT2_Xqq",
    "globalParT2_Xtauhtaue",
    "globalParT2_Xtauhtauh",
    "globalParT2_Xtauhtaum",
    "globalParT_massRes",
    "globalParT_massVis",
    "jetId",
    "lsf3",
    "msoftdrop",
    "muEF",
    "n2b1",
    "n3b1",
    "nConstituents",
    "neEmEF",
    "neHEF",
    "neMultiplicity",
    "particleNetLegacy_QCD",
    "particleNetLegacy_QCDb",
    "particleNetLegacy_QCDbb",
    "particleNetLegacy_QCDc",
    "particleNetLegacy_QCDcc",
    "particleNetLegacy_QCDothers",
    "particleNetLegacy_Xbb",
    "particleNetLegacy_Xcc",
    "particleNetLegacy_Xqq",
    "particleNetLegacy_mass",
    "particleNetWithMass_H4qvsQCD",
    "particleNetWithMass_HbbvsQCD",
    "particleNetWithMass_HccvsQCD",
    "particleNetWithMass_QCD",
    "particleNetWithMass_TvsQCD",
    "particleNetWithMass_WvsQCD",
    "particleNetWithMass_ZvsQCD",
    "particleNet_QCD",
    "particleNet_QCD0HF",
    "particleNet_QCD1HF",
    "particleNet_QCD2HF",
    "particleNet_Xbb",
    "particleNet_XbbVsQCD",
    "particleNet_XccVsQCD",
    "particleNet_XggVsQCD",
    "particleNet_XqqVsQCD",
    "particleNet_XteVsQCD",
    "particleNet_XtmVsQCD",
    "particleNet_XttVsQCD",
    "particleNet_massCorr",
    "rawFactor",
    "tau1",
    "tau2",
    "tau3",
    "tau4",
]

FatJetObservablesMC = ["hadronFlavour"]

SubJetObservables = ["btagDeepB", "eta", "mass", "phi", "pt", "rawFactor"]
SubJetObservablesMC = ["hadronFlavour", "partonFlavour"]

defaultColToSave = [
    "FullEventId",
    "luminosityBlock",
    "run",
    "event",
    "period",
    "X_mass",
    "X_spin",
    "isData",
    "PuppiMET_covXX",
    "PuppiMET_covXY",
    "PuppiMET_covYY",
    "PuppiMET_significance",
    "PuppiMET_sumEt",
    "nJet",
    "PV_npvs",
]

# Add this functionality eventually
MCObservables = [
    "LHEPdfWeight",
    "LHEReweightingWeight",
    "nLHEPdfWeight",
    "nLHEReweightingWeight",
    "LHEScaleWeight",
    "nLHEScaleWeight",
    "PSWeight",
    "nPSWeight",
]


def getDefaultColumnsToSave(isData):
    colToSave = defaultColToSave.copy()
    if not isData:
        colToSave.extend(["Pileup_nTrueInt"])
    return colToSave


def addAllVariables(
    dfw,
    syst_name,
    isData,
    trigger_class,
    lepton_legs,
    isSignal,
    applyTriggerFilter,
    global_params,
    channels,
):
    print(f"Adding variables for {syst_name}")
    # dfw.Apply(CommonBaseline.SelectRecoP4, syst_name, global_params["nano_version"])
    dfw.Apply(AnaBaseline.RecoHWWCandidateSelection)
    dfw.Apply(AnaBaseline.RecoHWWJetSelection)
    dfw.Apply(Corrections.getGlobal().jet.getEnergyResolution)
    dfw.Apply(Corrections.getGlobal().btag.getWPid)
    dfw.Apply(
        Corrections.getGlobal().JetVetoMap.GetJetVetoMap
    )  # Must init JetVetoMap before applying
    dfw.Apply(CommonBaseline.ApplyJetVetoMap)

    dfw.Define("Jet_isForward", "abs(v_ops::eta(Jet_p4)) > 2.5")
    for var in ["pt", "eta", "phi", "mass"]:
        dfw.DefineAndAppend(f"ForwardJet_{var}", f"v_ops::{var}(Jet_p4[Jet_isForward])")
    for var in [
        "jetId",
        "puIdDisc",
    ]:  # These are not part of the v_ops namespace due to not being part of p4 vec
        dfw.DefineAndAppend(f"ForwardJet_{var}", f"Jet_{var}[Jet_isForward]")

    PtEtaPhiM = ["pt", "eta", "phi", "mass"]
    # save reco lepton from HWWcandidate
    dfw.DefineAndAppend(f"nSelMu", f"Muon_pt[Muon_sel].size()")
    dfw.DefineAndAppend(f"nSelEle", f"Electron_pt[Electron_sel].size()")
    n_legs = 2
    for leg_idx in range(n_legs):

        def LegVar(var_name, var_expr, var_type=None, var_cond=None, default=0):
            cond = f"HwwCandidate.leg_type.size() > {leg_idx}"
            if var_cond:
                cond = f"{cond} && ({var_cond})"
            define_expr = (
                f"static_cast<{var_type}>({var_expr})" if var_type else var_expr
            )
            full_define_expr = f"{cond} ? ({define_expr}) : {default}"
            dfw.DefineAndAppend(f"lep{leg_idx+1}_{var_name}", full_define_expr)

        for var in PtEtaPhiM:
            LegVar(
                var,
                f"HwwCandidate.leg_p4.at({leg_idx}).{var}()",
                var_type="float",
                default="0.f",
            )
        LegVar(
            "legType", f"HwwCandidate.leg_type.at({leg_idx})", default="Leg::none"
        )  # Save the two for now, type is used in many corrections
        LegVar(
            "charge",
            f"HwwCandidate.leg_charge.at({leg_idx})",
            var_type="int",
            default="0",
        )
        LegVar(
            "iso",
            f"HwwCandidate.leg_rawIso.at({leg_idx})",
            var_type="float",
            default="0",
        )
        for muon_obs in Muon_observables:
            LegVar(
                muon_obs,
                f"{muon_obs}.at(HwwCandidate.leg_index.at({leg_idx}))",
                var_cond=f"HwwCandidate.leg_type.at({leg_idx}) == Leg::mu",
                default="-1",
            )
        for ele_obs in Electron_observables:
            LegVar(
                ele_obs,
                f"{ele_obs}.at(HwwCandidate.leg_index.at({leg_idx}))",
                var_cond=f"HwwCandidate.leg_type.at({leg_idx}) == Leg::e",
                default="-1",
            )
        # Save the lep* p4 and index directly to avoid using HwwCandidate in SF LUTs
        dfw.Define(
            f"lep{leg_idx+1}_p4",
            f"HwwCandidate.leg_type.size() > {leg_idx} ? HwwCandidate.leg_p4.at({leg_idx}) : LorentzVectorM()",
        )
        dfw.Define(
            f"lep{leg_idx+1}_index",
            f"HwwCandidate.leg_type.size() > {leg_idx} ? HwwCandidate.leg_index.at({leg_idx}) : -1",
        )

    # save information for fatjets
    fatjet_obs = []
    fatjet_obs.extend(FatJetObservables)
    if not isData:
        dfw.Define(
            f"tmp_FatJet_genJet_idx",
            f" FindMatching(FatJet_p4[FatJet_sel],GenJetAK8_p4,0.3)",
        )
        fatjet_obs.extend(FatJetObservablesMC)
    dfw.Define(f"tmp_SelectedFatJet_pt", f"v_ops::pt(FatJet_p4[FatJet_sel])")
    dfw.Define(f"tmp_SelectedFatJet_eta", f"v_ops::eta(FatJet_p4[FatJet_sel])")
    dfw.Define(f"tmp_SelectedFatJet_phi", f"v_ops::phi(FatJet_p4[FatJet_sel])")
    dfw.Define(f"tmp_SelectedFatJet_mass", f"v_ops::mass(FatJet_p4[FatJet_sel])")
    for fatjetVar in fatjet_obs:
        dfw.Define(f"tmp_SelectedFatJet_{fatjetVar}", f"FatJet_{fatjetVar}[FatJet_sel]")
    subjet_obs = []
    subjet_obs.extend(SubJetObservables)
    if not isData:
        dfw.Define(
            f"tmp_SubJet1_genJet_idx",
            f" FindMatching(SubJet_p4[FatJet_subJetIdx1],SubGenJetAK8_p4,0.3)",
        )
        dfw.Define(
            f"tmp_SubJet2_genJet_idx",
            f" FindMatching(SubJet_p4[FatJet_subJetIdx2],SubGenJetAK8_p4,0.3)",
        )
        fatjet_obs.extend(SubJetObservablesMC)
    for subJetIdx in [1, 2]:
        dfw.Define(
            f"tmp_SelectedFatJet_subJetIdx{subJetIdx}",
            f"FatJet_subJetIdx{subJetIdx}[FatJet_sel]",
        )
        dfw.Define(
            f"tmp_FatJet_SubJet{subJetIdx}_isValid",
            f" FatJet_subJetIdx{subJetIdx} >=0 && FatJet_subJetIdx{subJetIdx} < nSubJet",
        )
        dfw.Define(
            f"tmp_SelectedFatJet_SubJet{subJetIdx}_isValid",
            f"tmp_FatJet_SubJet{subJetIdx}_isValid[FatJet_sel]",
        )
        for subJetVar in subjet_obs:
            dfw.Define(
                f"tmp_SelectedFatJet_SubJet{subJetIdx}_{subJetVar}",
                f"""
                                RVecF subjet_var(tmp_SelectedFatJet_pt.size(), 0.f);
                                for(size_t fj_idx = 0; fj_idx<tmp_SelectedFatJet_pt.size(); fj_idx++) {{
                                    auto sj_idx = tmp_SelectedFatJet_subJetIdx{subJetIdx}.at(fj_idx);
                                    if(sj_idx >= 0 && sj_idx < SubJet_{subJetVar}.size()){{
                                        subjet_var[fj_idx] = SubJet_{subJetVar}.at(sj_idx);
                                    }}
                                }}
                                return subjet_var;
                                """,
            )

    dfw.Define(
        "SelectedFatJet_idx",
        "CreateIndexes(tmp_SelectedFatJet_particleNet_XbbVsQCD.size())",
    )
    dfw.Define(
        "SelectedFatJet_idxSorted",
        "ReorderObjects(tmp_SelectedFatJet_particleNet_XbbVsQCD, SelectedFatJet_idx)",
    )

    fatjet_obs = []
    fatjet_obs.extend(FatJetObservables)
    if not isData:
        dfw.Define(
            f"FatJet_genJet_idx",
            f" Take(tmp_FatJet_genJet_idx, SelectedFatJet_idxSorted)",
        )
        fatjet_obs.extend(FatJetObservablesMC)
    for var in PtEtaPhiM:
        name = f"SelectedFatJet_{var}"
        dfw.DefineAndAppend(
            name, f"Take(tmp_SelectedFatJet_{var}, SelectedFatJet_idxSorted)"
        )
    for fatjetVar in fatjet_obs:
        dfw.DefineAndAppend(
            f"SelectedFatJet_{fatjetVar}",
            f"Take(tmp_SelectedFatJet_{fatjetVar}, SelectedFatJet_idxSorted)",
        )
    subjet_obs = []
    subjet_obs.extend(SubJetObservables)
    if not isData:
        dfw.Define(
            f"SubJet1_genJet_idx",
            f" Take(tmp_SubJet1_genJet_idx, SelectedFatJet_idxSorted)",
        )
        dfw.Define(
            f"SubJet2_genJet_idx",
            f" Take(tmp_SubJet2_genJet_idx, SelectedFatJet_idxSorted)",
        )
        fatjet_obs.extend(SubJetObservablesMC)
    for subJetIdx in [1, 2]:
        dfw.Define(
            f"SelectedFatJet_subJetIdx{subJetIdx}",
            f"Take(tmp_SelectedFatJet_subJetIdx{subJetIdx}, SelectedFatJet_idxSorted)",
        )
        dfw.Define(
            f"FatJet_SubJet{subJetIdx}_isValid",
            f" Take(tmp_FatJet_SubJet{subJetIdx}_isValid, SelectedFatJet_idxSorted)",
        )
        dfw.DefineAndAppend(
            f"SelectedFatJet_SubJet{subJetIdx}_isValid",
            f"Take(tmp_SelectedFatJet_SubJet{subJetIdx}_isValid, SelectedFatJet_idxSorted)",
        )
        for subJetVar in subjet_obs:
            dfw.DefineAndAppend(
                f"SelectedFatJet_SubJet{subJetIdx}_{subJetVar}",
                f"Take(tmp_SelectedFatJet_SubJet{subJetIdx}_{subJetVar}, SelectedFatJet_idxSorted)",
            )

    met_type = global_params["met_type"]
    dfw.DefineAndAppend(
        f"{met_type}_pt_nano", f"static_cast<float>({met_type}_p4_nano.pt())"
    )
    dfw.DefineAndAppend(
        f"{met_type}_phi_nano", f"static_cast<float>({met_type}_p4_nano.phi())"
    )

    dfw.RedefineAndAppend(f"{met_type}_pt", f"static_cast<float>({met_type}_p4.pt())")
    dfw.RedefineAndAppend(f"{met_type}_phi", f"static_cast<float>({met_type}_p4.phi())")

    if trigger_class is not None:
        hltBranches = dfw.Apply(
            trigger_class.ApplyTriggers, lepton_legs, isData, applyTriggerFilter
        )
        dfw.colToSave.extend(hltBranches)

    dfw.DefineAndAppend("channelId", "static_cast<int>(HwwCandidate.channel())")
    channel_to_select = " || ".join(
        f"HwwCandidate.channel()==Channel::{ch}" for ch in channels
    )  # global_params["channelSelection"])
    dfw.Filter(channel_to_select, "select channels")
    # save all selected reco jets
    dfw.Define("centralJet_idx", "CreateIndexes(Jet_btagPNetB[Jet_sel].size())")
    dfw.Define(
        "centralJet_idxSorted", "ReorderObjects(Jet_btagPNetB[Jet_sel], centralJet_idx)"
    )
    for var in PtEtaPhiM:
        name = f"centralJet_{var}"
        dfw.DefineAndAppend(
            name, f"Take(v_ops::{var}(Jet_p4[Jet_sel]), centralJet_idxSorted)"
        )

    # save gen jets matched to selected reco jets
    if not isData:
        dfw.Define(
            "centralJet_matchedGenJetIdx",
            f"Take(Jet_genJetIdx[Jet_sel], centralJet_idxSorted)",
        )
        for var in PtEtaPhiM:
            name = f"centralJet_matchedGenJet_{var}"
            dfw.DefineAndAppend(
                name,
                f"""RVecF res;
                                        for (auto idx: centralJet_matchedGenJetIdx)
                                        {{
                                            res.push_back(idx == -1 ? 0.0 : GenJet_p4[idx].{var}());
                                        }}
                                        return res;""",
            )

        for var in JetObservablesMC:
            name = f"centralJet_matchedGenJet_{var}"
            dfw.DefineAndAppend(
                name,
                f"""RVecF res;
                                        for (auto idx: centralJet_matchedGenJetIdx)
                                        {{
                                            res.push_back(idx == -1 ? 0.0 : GenJet_{var}[idx]);
                                        }}
                                        return res;""",
            )

        dfw.Define("IsTrueBjet", "GenJet_hadronFlavour == 5")
        dfw.Define(
            "GenJet_TrueBjetTag",
            "FindTwoJetsClosestToMPV(125.0, GenJet_p4, IsTrueBjet)",
        )
        dfw.DefineAndAppend(
            "centralJet_TrueBjetTag",
            """RVecI res;
                                        for (auto idx: centralJet_matchedGenJetIdx)
                                        {
                                            res.push_back(idx == -1 ? 0 : static_cast<int>(GenJet_TrueBjetTag[idx]));
                                        }
                                        return res;""",
        )
    reco_jet_obs = []
    reco_jet_obs.extend(JetObservables)
    if not isData:
        reco_jet_obs.extend(JetObservablesMC)
    for jet_obs in reco_jet_obs:
        name = f"centralJet_{jet_obs}"
        dfw.DefineAndAppend(name, f"Take(Jet_{jet_obs}[Jet_sel], centralJet_idxSorted)")
    if isSignal:
        # save gen H->VV
        dfw.Define(
            "H_to_VV",
            """GetGenHVVCandidate(event, genLeptons, GenPart_pdgId, GenPart_daughters, GenPart_statusFlags, GenPart_pt, GenPart_eta, GenPart_phi, GenPart_mass, GenJet_p4, true)""",
        )
        for var in PtEtaPhiM:
            dfw.DefineAndAppend(
                f"genHVV_{var}", f"static_cast<float>(H_to_VV.cand_p4.{var}())"
            )

        # save gen level vector bosons from H->VV
        for boson in [1, 2]:
            name = f"genV{boson}"
            dfw.DefineAndAppend(
                f"{name}_pdgId", f"GenPart_pdgId[ H_to_VV.legs[{boson - 1}].index ]"
            )
            for var in PtEtaPhiM:
                dfw.DefineAndAppend(
                    f"{name}_{var}",
                    f"static_cast<float>(H_to_VV.legs[{boson - 1}].cand_p4.{var}())",
                )

        # save gen level products of vector boson decays (prod - index of product (quark, leptons or neutrinos))
        for boson in [1, 2]:
            for prod in [1, 2]:
                name = f"genV{boson}prod{prod}"
                for var in PtEtaPhiM:
                    dfw.DefineAndAppend(
                        f"{name}_{var}",
                        f"static_cast<float>(H_to_VV.legs[{boson - 1}].leg_p4[{prod - 1}].{var}())",
                    )
                    dfw.DefineAndAppend(
                        f"{name}_vis_{var}",
                        f"static_cast<float>(H_to_VV.legs[{boson - 1}].leg_vis_p4[{prod - 1}].{var}())",
                    )
                dfw.DefineAndAppend(
                    f"{name}_legType",
                    f"static_cast<int>(H_to_VV.legs[{boson - 1}].leg_kind[{prod - 1}])",
                )
                dfw.DefineAndAppend(
                    f"{name}_pdgId",
                    f"GenPart_pdgId[ H_to_VV.legs.at({boson - 1}).leg_index.at({prod - 1}) ]",
                )

        # save gen level H->bb
        dfw.Define(
            "H_to_bb",
            """GetGenHBBCandidate(event, GenPart_pdgId, GenPart_daughters, GenPart_statusFlags, GenPart_pt, GenPart_eta, GenPart_phi, GenPart_mass, GenJet_p4, false)""",
        )
        for var in PtEtaPhiM:
            dfw.DefineAndAppend(
                f"genHbb_{var}", f"static_cast<float>(H_to_bb.cand_p4.{var}())"
            )

        # save gen level b quarks
        for b_quark in [1, 2]:
            name = f"genb{b_quark}"
            for var in PtEtaPhiM:
                dfw.DefineAndAppend(
                    f"{name}_{var}",
                    f"static_cast<float>(H_to_bb.leg_p4[{b_quark - 1}].{var}())",
                )
                dfw.DefineAndAppend(
                    f"{name}_vis_{var}",
                    f"static_cast<float>(H_to_bb.leg_vis_p4[{b_quark - 1}].{var}())",
                )

    if not isData:
        # save gen leptons matched to reco leptons
        for lep in [1, 2]:
            name = f"lep{lep}_gen"
            # MatchGenLepton returns index of in genLetpons collection if match exists
            dfw.Define(
                f"{name}_idx",
                f" HwwCandidate.leg_type.size() >= {lep} ? MatchGenLepton(HwwCandidate.leg_p4.at({lep - 1}), genLeptons, 0.4) : -1",
            )
            dfw.Define(
                f"{name}_p4",
                f"return {name}_idx == -1 ? LorentzVectorM() : LorentzVectorM(genLeptons.at({name}_idx).visibleP4());",
            )
            dfw.Define(
                f"{name}_mother_p4",
                f"return {name}_idx == -1 ? LorentzVectorM() : (*genLeptons.at({name}_idx).mothers().begin())->p4;",
            )

            dfw.DefineAndAppend(
                f"{name}_kind",
                f"return {name}_idx == -1 ? -1 : static_cast<int>(genLeptons.at({name}_idx).kind());",
            )
            dfw.DefineAndAppend(
                f"{name}_motherPdgId",
                f"return {name}_idx == -1 ? -1 : (*genLeptons.at({name}_idx).mothers().begin())->pdgId",
            )
            for var in PtEtaPhiM:
                dfw.DefineAndAppend(
                    f"{name}_mother_{var}",
                    f"static_cast<float>({name}_mother_p4.{var}())",
                )
                dfw.DefineAndAppend(
                    f"{name}_{var}", f"static_cast<float>({name}_p4.{var}())"
                )
