# yumeng:
# This script is for DILEPTON efficiency study. 
# The basic steps are: checking input files (NANOAOD), processing events, and applying selection criteria.  
# The main tasks include generating histograms using functions like 
# Report_cutflow_hist,
# EventsDistribution1D_hist,  
# EventsDistribution2D_hist, 
# comparecut_hist.
# The run options can be defined at the end of script 

import array

import datetime
import os
import sys
import ROOT
import shutil
import zlib

import argparse
import traceback

ROOT.gROOT.ProcessLine(".include " + os.environ['ANALYSIS_PATH'])
header_path_RootExt = "include/RootExt.h"
header_path_GenLepton = "include/GenLepton.h"
header_path_Gen = "include/BaselineGenSelection.h"
header_path_Reco = "include/BaselineRecoSelection.h"
header_path_HHbTag = "include/HHbTagScores.h"
header_path_AnalysisTools = "include/AnalysisTools.h"
ROOT.gInterpreter.Declare(f'#include "{header_path_RootExt}"')
ROOT.gInterpreter.Declare(f'#include "{header_path_GenLepton}"')
ROOT.gInterpreter.Declare(f'#include "{header_path_Gen}"')
ROOT.gInterpreter.Declare(f'#include "{header_path_Reco}"')
ROOT.gInterpreter.Declare(f'#include "{header_path_AnalysisTools}"') 

parser = argparse.ArgumentParser()
parser.add_argument('--particleFile', type=str,
                    default=f"{os.environ['ANALYSIS_PATH']}/config/pdg_name_type_charge.txt")

args = parser.parse_args()

ROOT.gInterpreter.ProcessLine(f"ParticleDB::Initialize(\"{args.particleFile}\");")

header_path_studyYumengAnalysisTools = "./yumenginclude/studyYumengAnalysisTools.h"
ROOT.gInterpreter.Declare(f'#include "{header_path_studyYumengAnalysisTools}"') 

#The functions I have added to studyYumengAnalysisTools.h for this study are as follows:
"""
ROOT::VecOps::RVec<LorentzVectorM> yumengGetP4(
    const ROOT::VecOps::RVec<double>& pt,
    const ROOT::VecOps::RVec<double>& eta,
    const ROOT::VecOps::RVec<double>& phi,
    const ROOT::VecOps::RVec<double>& mass) {

    ROOT::VecOps::RVec<LorentzVectorM> p4(pt.size());
    
    for (size_t i = 0; i < pt.size(); i++) {
        p4[i] = LorentzVectorM(
            static_cast<double>(pt[i]),
            static_cast<double>(eta[i]),
            static_cast<double>(phi[i]),
            static_cast<double>(mass[i])
        );
    }   
    return p4;
}

float FindMatchingdR(const LorentzVectorM& target_p4, const RVecLV& ref_p4,const float deltaR_thr){
  double deltaR_min = deltaR_thr;
  int current_idx = -1;
  for(int refIdx =0; refIdx<ref_p4.size(); refIdx++){
    auto dR_targetRef= ROOT::Math::VectorUtil::DeltaR(target_p4, ref_p4.at(refIdx));
    if ( dR_targetRef < deltaR_min ) {
      deltaR_min = dR_targetRef ;
      current_idx = refIdx;
    }
  }
  return deltaR_min;
}

std::set<size_t> FindMatchingSet(const LorentzVectorM& target_p4, const RVecLV& ref_p4, const float deltaR_thr){
  std::set<size_t> matched_idx;
  for(size_t refIdx =0; refIdx<ref_p4.size(); ++refIdx){
    auto dR_targetRef= ROOT::Math::VectorUtil::DeltaR(target_p4, ref_p4.at(refIdx));
    if ( dR_targetRef < deltaR_thr ) {
      matched_idx.insert(refIdx);
    }
  }
  return matched_idx;
}
"""

def check_root_file(file_path):
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        return False

    root_file = ROOT.TFile.Open(file_path)
    if not root_file or root_file.IsZombie():
        print(f"Error: The file '{file_path}' can not be opened.")
        return False

    print(f"Success: The file '{file_path}' is valid")
    root_file.Close()
    return True

def get_event_counts(df):
    try:
        return df.Count().GetValue()
    except Exception as e:
        print(f"Error counting events: {e}")
        #traceback.print_exc()

def process_tree_Gen(file_path, tree_name):
    df = ROOT.RDataFrame(tree_name, file_path)

    df = df.Define("GenJet_p4", "RVecLV()")
    df = df.Define("GenPart_daughters", "GetDaughters(GenPart_genPartIdxMother)")
    
    if tree_name == "Events":
        df = df.Define("genLeptons", """reco_tau::gen_truth::GenLepton::fromNanoAOD(GenPart_pt, GenPart_eta,
                                    GenPart_phi, GenPart_mass, GenPart_genPartIdxMother, GenPart_pdgId,
                                    GenPart_statusFlags, event)""")
        df = df.Define("H_to_VV", """GetGenHVVCandidate(event, genLeptons, GenPart_pdgId, GenPart_daughters, 
                                  GenPart_statusFlags, GenPart_pt, GenPart_eta, GenPart_phi, GenPart_mass, GenJet_p4, true)""")
        df = df.Define("H_to_BB", """GetGenHBBCandidate(event, GenPart_pdgId, GenPart_daughters, GenPart_statusFlags, GenPart_pt, GenPart_eta, GenPart_phi, GenPart_mass, GenJet_p4, true)""")

        df = df.Define(f"{mychfull}_p4", f"yumengGetP4({mychfull}_pt, {mychfull}_eta, {mychfull}_phi, {mychfull}_mass)")
        if mychannel == "dle" or mychannel == "dlmuon":
            for i in range(2):
                df=df.Define(f"RecoLepIdx{i}", f"FindMatching(H_to_VV.legs.at({i}).leg_vis_p4.at(0), {mychfull}_p4, {matchThresh})")
        if mychannel == "dlmu_e":
                df = df.Define(f"{mychfull1}_p4", f"yumengGetP4({mychfull1}_pt, {mychfull1}_eta, {mychfull1}_phi, {mychfull1}_mass)")

    else:
        df = df.Define("genLeptons", """reco_tau::gen_truth::GenLepton::fromNanoAOD(GenPart_pt, GenPart_eta,
                                    GenPart_phi, GenPart_mass, GenPart_genPartIdxMother, GenPart_pdgId,
                                    GenPart_statusFlags,-1)""")
        df = df.Define("H_to_VV", """GetGenHVVCandidate(-1,genLeptons, GenPart_pdgId, GenPart_daughters, 
                                  GenPart_statusFlags, GenPart_pt, GenPart_eta, GenPart_phi, GenPart_mass, GenJet_p4, true)""")
        df = df.Define("H_to_BB", """GetGenHBBCandidate(-1, GenPart_pdgId, GenPart_daughters, GenPart_statusFlags, GenPart_pt, GenPart_eta, GenPart_phi, GenPart_mass, GenJet_p4, true)""")
    
    for i in range(2):
        df = df.Define(f"Leg{i}_Kind", f"H_to_VV.legs.at({i}).leg_kind.at(0)")
        df= df.Define(f"GenLep{i}_p4",f"H_to_VV.legs.at({i}).leg_vis_p4.at(0)")
        df= df.Define(f"Genb{i}_p4",f"H_to_BB.leg_p4.at({i})")

    if mychannel == "dle":
        df_GenDL = df.Filter("Leg0_Kind == Vleg::PromptElectron && Leg1_Kind == Vleg::PromptElectron", "")
        genDL_count = get_event_counts(df_GenDL)

        df_Gen_Accept = df_GenDL.Filter("GenLep0_p4.pt() > 5 && abs(GenLep0_p4.eta()) < 2.5 && GenLep1_p4.pt() > 5 && abs(GenLep1_p4.eta()) < 2.5", "")
        gen_accept_count = get_event_counts(df_Gen_Accept)
        
    if mychannel == "dlmuon":
        df_GenDL=df.Filter("Leg0_Kind==Vleg::PromptMuon && Leg1_Kind==Vleg::PromptMuon", "")
        genDL_count = get_event_counts(df_GenDL)

        df_Gen_Accept = df_GenDL.Filter("GenLep0_p4.pt()>5 && abs (GenLep0_p4.eta()) <2.5 && GenLep1_p4.pt()>5 && abs (GenLep1_p4.eta()) <2.5", "")
        gen_accept_count = get_event_counts(df_Gen_Accept)
    
    if mychannel == "dlmu_e":
        df_GenDL = df.Filter(
        "(Leg0_Kind == Vleg::PromptMuon && Leg1_Kind == Vleg::PromptElectron) || "
        "(Leg0_Kind == Vleg::PromptElectron && Leg1_Kind == Vleg::PromptMuon)", 
        "")
        genDL_count = get_event_counts(df_GenDL)

        df_GenDL = df_GenDL.Define("GenMu_p4", "Leg0_Kind == Vleg::PromptMuon ? GenLep0_p4 : GenLep1_p4") \
                   .Define("Gene_p4", "Leg0_Kind == Vleg::PromptElectron ? GenLep0_p4 : GenLep1_p4")
        
        df_GenDL= df_GenDL.Define("RecoLepIdx0", f"FindMatching(GenMu_p4, Muon_p4, {matchThresh})") \
                    .Define("RecoLepIdx1", f"FindMatching(Gene_p4, Electron_p4, {matchThresh1})")

        df_Gen_Accept = df_GenDL.Filter("GenMu_p4.pt() > 5 && abs(GenMu_p4.eta()) < 2.5 && Gene_p4.pt() > 5 && abs(Gene_p4.eta()) < 2.5", "")
        gen_accept_count = get_event_counts(df_Gen_Accept)
    
    df_Genb_Accept = df_Gen_Accept.Filter("Genb0_p4.pt()>20 && Genb1_p4.pt()>20 && abs (Genb0_p4.eta()) <2.5 && abs (Genb1_p4.eta()) <2.5", "")
    genb_accept_count = get_event_counts(df_Genb_Accept)

    return df_GenDL, df_Gen_Accept, df_Genb_Accept,genDL_count, gen_accept_count, genb_accept_count

def RecoHWWCandidateSelection(df):
    def add_filter(df, base_cut, and_cut, label):
        full_cut = f"Sum({base_cut} && {and_cut}) > 0"
        df = df.Filter(full_cut, label)
        return df, f"{base_cut} && {and_cut}"
    
    if mychannel == "dle":
        df=df.Filter("nElectron>1", "RecoElectron>=2")
        df=df.Filter("RecoLepIdx0>=0 && RecoLepIdx1 >=0 && RecoLepIdx0!=RecoLepIdx1", "Reco_ee_MatchGen")
        df = df.Filter("Electron_pt.at(RecoLepIdx0)>5 && Electron_pt.at(RecoLepIdx1)>5" , "Reco_ee_pt>5")#10
        df = df.Filter("Electron_eta.at(RecoLepIdx0)<2.5 && Electron_eta.at(RecoLepIdx1)<2.5", "Reco_ee_eta<2.5")
        df = df.Filter("Electron_dz.at(RecoLepIdx0)<0.1 && Electron_dz.at(RecoLepIdx1)<0.1", "Reco_ee_dz<0.1")
        df = df.Filter("Electron_dxy.at(RecoLepIdx0)<0.05 && Electron_dxy.at(RecoLepIdx1)<0.05", "Reco_ee_dxy<0.05")
        df = df.Filter("Electron_sip3d.at(RecoLepIdx0) <= 8 && Electron_sip3d.at(RecoLepIdx1) <= 8", "Reco_ee_sip3d <= 8")

    if mychannel== "dlmuon":
        df=df.Filter("nMuon>1", "RecoMuon>=2")
        # data = df.AsNumpy(["RecoLepIdx0", "RecoLepIdx1"])
        # print("debug", data["RecoLepIdx0"][:10], data["RecoLepIdx1"][:10])
        df=df.Filter("RecoLepIdx0>=0 && RecoLepIdx1>=0 && RecoLepIdx0!= RecoLepIdx1", "Reco_mumu_MatchGen")
        df = df.Filter("Muon_pt.at(RecoLepIdx0)>5 && Muon_pt.at(RecoLepIdx1)>5", "Reco_mumu_pt>5")  #15
        df = df.Filter("Muon_eta.at(RecoLepIdx0)<2.5 && Muon_eta.at(RecoLepIdx1)<2.5", "Reco_mumu_eta<2.5") #2.4
        df = df.Filter("Muon_dz.at(RecoLepIdx0)<0.1 && Muon_dz.at(RecoLepIdx1)<0.1", "Reco_mumu_dz<0.1")
        df = df.Filter("Muon_dxy.at(RecoLepIdx0)<0.05 && Muon_dxy.at(RecoLepIdx1)<0.05", "Reco_mumu_dxy<0.05")
        df = df.Filter("Muon_sip3d.at(RecoLepIdx0) <= 8 && Muon_sip3d.at(RecoLepIdx1) <= 8", "Reco_mumu_sip3d <= 8")

    if mychannel== "dlmu_e":
        df=df.Filter("nMuon>0 && nElectron>0", "At least 1 mu and 1 e")
        df=df.Filter("RecoLepIdx0>=0 && RecoLepIdx1>=0", "Reco_emu_MatchGen")
        df = df.Filter("Muon_pt.at(RecoLepIdx0)>5 && Electron_pt.at(RecoLepIdx1)>5", "Reco_emu_pt>5") 
        df = df.Filter("Muon_eta.at(RecoLepIdx0)<2.5 && Electron_eta.at(RecoLepIdx1)<2.5", "Reco_emu_eta<2.5")
        df = df.Filter("Muon_dz.at(RecoLepIdx0)<0.1 && Electron_dz.at(RecoLepIdx1)<0.1", "Reco_emu_dz<0.1")
        df = df.Filter("Muon_dxy.at(RecoLepIdx0)<0.05 && Electron_dxy.at(RecoLepIdx1)<0.05", "Reco_emu_dxy<0.05")
        df = df.Filter("Muon_sip3d.at(RecoLepIdx0) <= 8 && Electron_sip3d.at(RecoLepIdx1) <= 8", "Reco_emu_sip3d <= 8")
    return df

def Report_cutflow_hist(canvas, mass_point, report, initial_count, genDL_count,  gen_accept_count, genb_accept_count, skim_count, reportName="Cutflow", printOut=False):
    cuts = [c for c in report]
    num_cuts = len(cuts)

    hist_eff = ROOT.TH1D(reportName + f"__M-{mass_point}" + "_Efficiency", reportName+f" {ctype}"+" Efficiency" + "_for Masspoints", num_cuts + 4, 0, num_cuts + 4)
    hist_eff.GetXaxis().SetBinLabel(1, "Initial_NANOAOD")
    hist_eff.SetBinContent(1, 1.0)
    hist_eff.SetBinError(1, 0.0) 
    hist_eff.SetMinimum(0)

    if ctype == "Relative":
        hist_eff.GetXaxis().SetBinLabel(2, f"GenDL{mychbrief}")
        hist_eff.SetBinContent(2, genDL_count/initial_count)

        hist_eff.GetXaxis().SetBinLabel(3, f"Gen_{mychbrief}_Lep_Accept")
        hist_eff.SetBinContent(3, gen_accept_count/genDL_count)

        hist_eff.GetXaxis().SetBinLabel(4, f"Gen_b_Accept")
        hist_eff.SetBinContent(4, genb_accept_count/gen_accept_count)

        for c_id, cut in enumerate(cuts):
                p = cut.GetEff() / 100
                n = cut.GetAll()
                binomial_error = ROOT.TMath.Sqrt(p * (1 - p) / n)
                hist_eff.SetBinContent(c_id + 5, cut.GetEff()/100)
                hist_eff.GetXaxis().SetBinLabel(c_id + 5, cut.GetName())
                hist_eff.SetBinError(c_id + 5, binomial_error)
    
    if ctype == "Cumulative":
        hist_eff.GetXaxis().SetBinLabel(2, f"GenDL{mychbrief}")
        hist_eff.SetBinContent(2, genDL_count/initial_count)

        hist_eff.GetXaxis().SetBinLabel(3, f"Gen_{mychbrief}_Lep_Accept")
        hist_eff.SetBinContent(3, gen_accept_count/initial_count)

        hist_eff.GetXaxis().SetBinLabel(4, f"Gen_b_Accept")
        hist_eff.SetBinContent(4, genb_accept_count/initial_count)

        for c_id, cut in enumerate(cuts):
                p = cut.GetPass()/initial_count
                n = initial_count
                binomial_error = ROOT.TMath.Sqrt(p * (1 - p) / n)
                hist_eff.SetBinContent(c_id + 5, cut.GetPass()/initial_count)
                hist_eff.GetXaxis().SetBinLabel(c_id + 5, cut.GetName())
                hist_eff.SetBinError(c_id + 5, binomial_error)
    
    hist_eff.GetXaxis().SetLabelSize(0.03);  
    
    #hist_eff.GetXaxis().LabelsOption("v");
    canvas.SetBottomMargin(0.25);

    return hist_eff

def EventsDistribution1D_hist(mass_point,df):  
    if mychannel == "dle" or mychannel == "dlmuon":
            for i in range(2):
                df = df.Define(f"RecolepdRSize{i}", f"FindMatchingSet(H_to_VV.legs.at({i}).leg_vis_p4.at(0), {mychfull}_p4, 0.02).size()")
    if mychannel == "dlmu_e":
                df= df.Define(f"RecolepdRSize0", f"FindMatchingSet(GenMu_p4, Muon_p4, 0.02).size()") \
                    .Define(f"RecolepdRSize1", f"FindMatchingSet(Gene_p4, Electron_p4, 0.02).size()")

    yrange=1#0.25

    hist_pt = df.Histo1D((f"recolepdRSize-{mass_point}", f"Count of Recoleps within dR of Genlep({mychbrief})", 10, 0, 10), "RecolepdRSize0")
    hist_pt.GetXaxis().SetTitle(f"Count, {mychbrief}")
    fname="matcbdROne" 
   
    hist_pt = hist_pt.GetValue()

    return hist_pt, yrange, fname

def EventsDistribution2D(mass_point,df):
    if mychannel == "dle" or mychannel == "dlmuon":
        for i in range(2):
            df = df.Define(f"dR_GenRecoLep{i}", f"ROOT::Math::VectorUtil::DeltaR(GenLep{i}_p4, {mychfull}_p4.at(RecoLepIdx{i}))");
            df = df.Define(f"PtRatio_GenRecoLep{i}", f"GenLep{i}_p4.pt() / {mychfull}_pt.at(RecoLepIdx{i})");
    if mychannel == "dlmu_e":
        df = df.Define(f"dR_GenRecoLep0", f"ROOT::Math::VectorUtil::DeltaR(GenMu_p4, Muon_p4.at(RecoLepIdx0))") \
                .Define(f"dR_GenRecoLep1", f"ROOT::Math::VectorUtil::DeltaR(Gene_p4, Electron_p4.at(RecoLepIdx1))");
        df = df.Define(f"PtRatio_GenRecoLep0", f"GenMu_p4.pt() / Muon_pt.at(RecoLepIdx0)") \
                .Define(f"PtRatio_GenRecoLep1", f"Gene_p4.pt() / Electron_pt.at(RecoLepIdx1)");

    if mychannel == "dle" or mychannel == "dlmuon":
        df=df.Filter("RecoLepIdx0>=0 && RecoLepIdx1 >=0 && RecoLepIdx0!=RecoLepIdx1", "")
    if mychannel == "dlmu_e":
         df=df.Filter("RecoLepIdx0>=0 && RecoLepIdx1>=0", "")

    # particle= f"PtRatio_GenRecoLep0"
    # particle2= f"dR_GenRecoLep0"
    particle= f"PtRatio_GenRecoLep1"
    particle2= f"dR_GenRecoLep1"
    xmax=1.5
    ymax=0.05
    yrange=0.9
    bin_wid=0.001

    pt_min = 0

    binsx=int((xmax - pt_min)/bin_wid)
    binsy=int((ymax - pt_min)/bin_wid)
    model_2d = ROOT.RDF.TH2DModel(f"{particle2}_vs_{particle}_M-{mass_point}", f"{particle2} vs {particle} M{mass_point}({mychbrief})", 
                                    binsx, pt_min, xmax,
                                    binsy, pt_min, ymax)

    hist_pt= df.Histo2D(model_2d, f"{particle}", f"{particle2}")


    hist_pt.GetXaxis().SetTitle(f"{particle}(GeV)({mychbrief} channel)")
    hist_pt.GetYaxis().SetTitle(f"{particle2}(GeV)({mychbrief} channel)")

    return hist_pt, yrange, particle

def comparecut_hist(mass_point, df, initial_count):
    if mychannel == "dle":
        ##################################################Ele: 
        cuts = [
        # f"{mychfull}_cutBased.at(RecoLepIdx) >= 1",  
        # f"{mychfull}_cutBased.at(RecoLepIdx) >= 2",
        # f"{mychfull}_cutBased.at(RecoLepIdx) >= 3",
        # f"{mychfull}_cutBased.at(RecoLepIdx) >= 4",
        f"{mychfull}_mvaIso_WP80.at(RecoLepIdx0) && {mychfull}_mvaIso_WP80.at(RecoLepIdx1)",
        f"{mychfull}_mvaIso_WP90.at(RecoLepIdx0) && {mychfull}_mvaIso_WP90.at(RecoLepIdx1) ",
        #f"{mychfull}_mvaIso_WPHZZ.at(RecoLepIdx)",
        # f"{mychfull}_mvaNoIso_WP80.at(RecoLepIdx) && {mychfull}_pfRelIso03_all.at(RecoLepIdx)<0.4",
        # f"{mychfull}_mvaNoIso_WP80.at(RecoLepIdx) && {mychfull}_pfRelIso03_all.at(RecoLepIdx)<0.25",
        # f"{mychfull}_mvaNoIso_WP80.at(RecoLepIdx) && {mychfull}_pfRelIso03_all.at(RecoLepIdx)<0.2",
        # f"{mychfull}_mvaNoIso_WP80.at(RecoLepIdx) && {mychfull}_pfRelIso03_all.at(RecoLepIdx)<0.15",
        f"{mychfull}_mvaNoIso_WP80.at(RecoLepIdx0) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx0)<0.4 && {mychfull}_mvaNoIso_WP80.at(RecoLepIdx1) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx1)<0.4 ",
        f"{mychfull}_mvaNoIso_WP80.at(RecoLepIdx0) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx0)<0.25 && {mychfull}_mvaNoIso_WP80.at(RecoLepIdx1) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx1)<0.25 " ,
        f"{mychfull}_mvaNoIso_WP80.at(RecoLepIdx0) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx0)<0.2 && {mychfull}_mvaNoIso_WP80.at(RecoLepIdx1) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx1)<0.2",
        f"{mychfull}_mvaNoIso_WP80.at(RecoLepIdx0) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx0)<0.15 && {mychfull}_mvaNoIso_WP80.at(RecoLepIdx1) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx1)<0.15",
        # f"{mychfull}_mvaNoIso_WP90.at(RecoLepIdx) && {mychfull}_pfRelIso03_all.at(RecoLepIdx)<0.4",
        # f"{mychfull}_mvaNoIso_WP90.at(RecoLepIdx) && {mychfull}_pfRelIso03_all.at(RecoLepIdx)<0.25",
        # f"{mychfull}_mvaNoIso_WP90.at(RecoLepIdx) && {mychfull}_pfRelIso03_all.at(RecoLepIdx)<0.2",
        # f"{mychfull}_mvaNoIso_WP90.at(RecoLepIdx) && {mychfull}_pfRelIso03_all.at(RecoLepIdx)<0.15",
        f"{mychfull}_mvaNoIso_WP90.at(RecoLepIdx0) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx0)<0.4 && {mychfull}_mvaNoIso_WP90.at(RecoLepIdx1) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx1)<0.4",
        f"{mychfull}_mvaNoIso_WP90.at(RecoLepIdx0) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx0)<0.25 && {mychfull}_mvaNoIso_WP90.at(RecoLepIdx1) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx1)<0.25",
        f"{mychfull}_mvaNoIso_WP90.at(RecoLepIdx0) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx0)<0.2 && {mychfull}_mvaNoIso_WP90.at(RecoLepIdx1) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx1)<0.2",
        f"{mychfull}_mvaNoIso_WP90.at(RecoLepIdx0) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx0)<0.15 && {mychfull}_mvaNoIso_WP90.at(RecoLepIdx1) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx1)<0.15",
    ]

        labels = [
        # "cutBased>=veto",
        # "cutBased>=Loose",
        # "cutBased>=Medium",
        # "cutBased>=Tight",
        "MVA Iso WP80",
        "MVA Iso WP90",
        #"MVA Iso WPHZZ",
        # "mvaNoIso_80 && pfIso03_all<0.4",
        # "mvaNoIso_80 && pfIso03_all<0.25 ",
        # "mvaNoIso_80 && pfIso03_all<0.2 ",
        # "mvaNoIso_80 && pfIso03_all<0.15 ",
        "mvaNoIso_80 && miniPF_all<0.4",
        "mvaNoIso_80 && miniPF_all<0.25 ",
        "mvaNoIso_80 && miniPF_all<0.2 ",
        "mvaNoIso_80 && miniPF_all<0.15 ",
        # "mvaNoIso_90 && pfIso03_all<0.4",
        # "mvaNoIso_90 && pfIso03_all<0.25 ",
        # "mvaNoIso_90 && pfIso03_all<0.2 ",
        # "mvaNoIso_90 && pfIso03_all<0.15 ",
        "mvaNoIso_90 && miniPF_all<0.4",
        "mvaNoIso_90 && miniPF_all<0.25 ",
        "mvaNoIso_90 && miniPF_all<0.2 ",
        "mvaNoIso_90 && miniPF_all<0.15 ",   
    ]

    if mychannel == "dlmuon":
        ##################################################Muon ID&&miniPFIso:
        cuts = [
        # Loose ID && PFRelIso cuts
        f"{mychfull}_looseId.at(RecoLepIdx0) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx0) < 0.4 && {mychfull}_looseId.at(RecoLepIdx1) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx1) < 0.4",  # Loose ID, Isolation < 0.4
        f"{mychfull}_looseId.at(RecoLepIdx0) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx0) < 0.25 &&{mychfull}_looseId.at(RecoLepIdx1) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx1) < 0.25 ", # Loose ID, Isolation < 0.25
        f"{mychfull}_looseId.at(RecoLepIdx0) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx0) < 0.2 && {mychfull}_looseId.at(RecoLepIdx1) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx1) < 0.2",  # Loose ID, Isolation < 0.2
        f"{mychfull}_looseId.at(RecoLepIdx0) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx0) < 0.15 && {mychfull}_looseId.at(RecoLepIdx1) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx1) < 0.15", # Loose ID, Isolation < 0.15

        # Medium ID && PFRelIso cuts
        f"{mychfull}_mediumId.at(RecoLepIdx0) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx0) < 0.4 && {mychfull}_mediumId.at(RecoLepIdx1) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx1) < 0.4",  # Medium ID, Isolation < 0.4
        f"{mychfull}_mediumId.at(RecoLepIdx0) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx0) < 0.25 && {mychfull}_mediumId.at(RecoLepIdx1) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx1) < 0.25", # Medium ID, Isolation < 0.25
        f"{mychfull}_mediumId.at(RecoLepIdx0) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx0) < 0.2 && {mychfull}_mediumId.at(RecoLepIdx1) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx1) < 0.2",  # Medium ID, Isolation < 0.2
        f"{mychfull}_mediumId.at(RecoLepIdx0) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx0) < 0.15 && {mychfull}_mediumId.at(RecoLepIdx1) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx1) < 0.15", # Medium ID, Isolation < 0.15

        # Tight ID && PFRelIso cuts
        f"{mychfull}_tightId.at(RecoLepIdx0) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx0) < 0.4 && {mychfull}_tightId.at(RecoLepIdx1) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx1) < 0.4",  # Tight ID, Isolation < 0.4
        f"{mychfull}_tightId.at(RecoLepIdx0) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx0) < 0.25 && {mychfull}_tightId.at(RecoLepIdx1) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx1) < 0.25", # Tight ID, Isolation < 0.25
        f"{mychfull}_tightId.at(RecoLepIdx0) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx0) < 0.2 && {mychfull}_tightId.at(RecoLepIdx1) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx1) < 0.2 ",  # Tight ID, Isolation < 0.2
        f"{mychfull}_tightId.at(RecoLepIdx0) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx0) < 0.15 && {mychfull}_tightId.at(RecoLepIdx1) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx1) < 0.15", # Tight ID, Isolation < 0.15

        # tracker highPt ID && PFRelIso cuts
        f"{mychfull}_highPtId.at(RecoLepIdx0)>=1 && {mychfull}_miniPFRelIso_all.at(RecoLepIdx0) < 0.4 && {mychfull}_highPtId.at(RecoLepIdx1)>=1 && {mychfull}_miniPFRelIso_all.at(RecoLepIdx1) < 0.4",  # Loose ID, Isolation < 0.4
        f"{mychfull}_highPtId.at(RecoLepIdx0)>=1 && {mychfull}_miniPFRelIso_all.at(RecoLepIdx0) < 0.25 &&{mychfull}_highPtId.at(RecoLepIdx1)>=1 && {mychfull}_miniPFRelIso_all.at(RecoLepIdx1) < 0.25 ", # Loose ID, Isolation < 0.25
        f"{mychfull}_highPtId.at(RecoLepIdx0)>=1 && {mychfull}_miniPFRelIso_all.at(RecoLepIdx0) < 0.2 && {mychfull}_highPtId.at(RecoLepIdx1)>=1 && {mychfull}_miniPFRelIso_all.at(RecoLepIdx1) < 0.2",  # Loose ID, Isolation < 0.2
        f"{mychfull}_highPtId.at(RecoLepIdx0)>=1 && {mychfull}_miniPFRelIso_all.at(RecoLepIdx0) < 0.15 && {mychfull}_highPtId.at(RecoLepIdx1)>=1 && {mychfull}_miniPFRelIso_all.at(RecoLepIdx1) < 0.15", # Loose ID, Isolation < 0.15

        # tracker highPt ID && tkIsoId cuts
        f"{mychfull}_highPtId.at(RecoLepIdx0)>=1 && {mychfull}_tkIsoId.at(RecoLepIdx0) >= 1 && {mychfull}_highPtId.at(RecoLepIdx1)>=1 && {mychfull}_tkIsoId.at(RecoLepIdx1) >= 1", 
        f"{mychfull}_highPtId.at(RecoLepIdx0)>=1 && {mychfull}_tkIsoId.at(RecoLepIdx0) >= 2 && {mychfull}_highPtId.at(RecoLepIdx1)>=1 && {mychfull}_tkIsoId.at(RecoLepIdx1) >= 2",

        # global highPt ID && PFRelIso cuts
        f"{mychfull}_highPtId.at(RecoLepIdx0)>=2 && {mychfull}_miniPFRelIso_all.at(RecoLepIdx0) < 0.4 && {mychfull}_highPtId.at(RecoLepIdx1)>=2 && {mychfull}_miniPFRelIso_all.at(RecoLepIdx1) < 0.4",  # Loose ID, Isolation < 0.4
        f"{mychfull}_highPtId.at(RecoLepIdx0)>=2 && {mychfull}_miniPFRelIso_all.at(RecoLepIdx0) < 0.25 &&{mychfull}_highPtId.at(RecoLepIdx1)>=2 && {mychfull}_miniPFRelIso_all.at(RecoLepIdx1) < 0.25 ", # Loose ID, Isolation < 0.25
        f"{mychfull}_highPtId.at(RecoLepIdx0)>=2 && {mychfull}_miniPFRelIso_all.at(RecoLepIdx0) < 0.2 && {mychfull}_highPtId.at(RecoLepIdx1)>=2 && {mychfull}_miniPFRelIso_all.at(RecoLepIdx1) < 0.2",  # Loose ID, Isolation < 0.2
        f"{mychfull}_highPtId.at(RecoLepIdx0)>=2 && {mychfull}_miniPFRelIso_all.at(RecoLepIdx0) < 0.15 && {mychfull}_highPtId.at(RecoLepIdx1)>=2 && {mychfull}_miniPFRelIso_all.at(RecoLepIdx1) < 0.15", # Loose ID, Isolation < 0.15
        ]

        labels = [
        # Loose ID labels
        "LooseID && miniPFIso_all< 0.4",
        "LooseID && miniPFIso< 0.25",
        "LooseID && miniPFIso< 0.2",
        "LooseID && miniPFIso< 0.15",

        # Medium ID labels
        "MediumID && miniPFIso< 0.4",
        "MediumID && miniPFIso< 0.25",
        "MediumID && miniPFIso< 0.2",
        "MediumID && miniPFIso< 0.15",

        # Tight ID labels
        "TightID && miniPFIso< 0.4",
        "TightID && miniPFIso< 0.25",
        "TightID && miniPFIso< 0.2",
        "TightID && miniPFIso< 0.15",

        # Tracker highPt ID labels
        "Tracker highPt && miniPFIso< 0.4",
        "Tracker highPt && miniPFIso< 0.25",
        "Tracker highPt && miniPFIso< 0.2",
        "Tracker highPt && miniPFIso< 0.15",

        # tracker highPt ID && tkIsoId cuts
        "Tracker highPt && TrackerIso Loose",
        "Tracker highPt && TrackerIso Tight",

         # global highPt ID labels
        "Global highPt && miniPFIso< 0.4",
        "Global highPt && miniPFIso< 0.25",
        "Global highPt && miniPFIso< 0.2",
        "Global highPt && miniPFIso< 0.15",
        ]

    if mychannel == "dlmu_e":
        cuts = [
        f"{mychfull}_tightId.at(RecoLepIdx0) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx0) < 0.15 && {mychfull1}_mvaIso_WP80.at(RecoLepIdx1)",
        f"{mychfull}_highPtId.at(RecoLepIdx0)>=1 && {mychfull}_miniPFRelIso_all.at(RecoLepIdx0) < 0.15 &&{mychfull1}_mvaIso_WP80.at(RecoLepIdx1)",
        f"{mychfull}_tightId.at(RecoLepIdx0) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx0) < 0.15 && {mychfull1}_mvaNoIso_WP80.at(RecoLepIdx1) && {mychfull1}_miniPFRelIso_all.at(RecoLepIdx1)<0.15", 
        f"{mychfull}_highPtId.at(RecoLepIdx0)>=1 && {mychfull}_miniPFRelIso_all.at(RecoLepIdx0) < 0.15 && {mychfull1}_mvaNoIso_WP80.at(RecoLepIdx1) && {mychfull1}_miniPFRelIso_all.at(RecoLepIdx1)<0.15", 
        ]

        labels = [ 
        "mu:Tight& mini0.15; e:mva80",
        "mu:highPt& mini0.15; e:mva80",
        "mu:Tight& mini0.15; e:mvaNo80& mini0.15",
        "mu:highPt& mini0.15 e:mvaNo80& mini0.15",
        ]

    #change for relative/culmulative
    #total_events = df.Count().GetValue()
    total_events = initial_count
    efficiencies = []
    for cut, label in zip(cuts, labels):
        filtered_df = df.Filter(cut, label)
        passed_events = filtered_df.Count().GetValue()
        efficiency = passed_events / total_events if total_events > 0 else 0
        efficiencies.append(efficiency)
        #print(f"M-{mass_point}, {label} cut relative eff:{efficiency}")

    #hist = ROOT.TH1F(f"Relativre Eff for diff cuts{mass_point}", "Relative Eff for diff cuts", len(labels), 0, len(labels))
    hist = ROOT.TH1F(f"Cumulative Eff for diff cuts{mass_point}", "Cumulative Eff for diff cuts", len(labels), 0, len(labels))

    for i, (label, efficiency) in enumerate(zip(labels, efficiencies), start=1):
        hist.SetBinContent(i, efficiency)
        hist.GetXaxis().SetBinLabel(i, label)

    hist.GetXaxis().SetTitle("Cuts")
    #hist.GetYaxis().SetTitle("Relative Efficiency")
    hist.GetYaxis().SetTitle("Culmulative Efficiency")
    hist.GetXaxis().SetLabelSize(0.03)
    hist.GetXaxis().LabelsOption("v");
    #hist.GetXaxis().LabelsOption("a");
    hist.SetMinimum(0)
    #hist.SetMaximum(1)
    hist.SetMaximum(0.1)
    return hist

######################################################################def of main functions starts:
def main(mychannel, hist_type):
    mass_points = [250, 450, 650, 1000, 3000, 5000]
    colors = [ROOT.kRed, ROOT.kBlue, ROOT.kGreen, ROOT.kMagenta, ROOT.kCyan, ROOT.kOrange]
    #mass_points = [5000]

    histograms = []
    canvas = ROOT.TCanvas("canvas", "Mass Points Comparison", 800, 600)
    #SetBottomMargin for comparecut_hist
    canvas.SetBottomMargin(0.4)
    canvas.SetGrid()
    yrange=1
    legend = ROOT.TLegend(0.75, 0.9, 1, 1)
    legend.SetNColumns(2)

    for i, mass_point in enumerate(mass_points):
        file_path = f"/eos/user/y/yumeng/ana/flaf/2022run3nanoaod/GluGlutoRadiontoHHto2B2Vto2B2L2Nu_M_hadd/hadd{mass_point}.root"
        if check_root_file(file_path):
            df_eventstree = ROOT.RDataFrame("Events", file_path)
            eventscount=get_event_counts(df_eventstree)
        #initial_count = eventscount + notselectedcount
        initial_count = eventscount

        df_GenDL_Eventstree, df_GenAccept_Eventstree, df_GenbAccept_Eventstree, genDL_count_eventstree, gen_accept_count_eventstree, genb_accept_count_eventstree = process_tree_Gen(file_path, "Events")

        if hist_type =="cutflow":
            hist=process_cutflow_hist(canvas, mass_point, initial_count, genDL_count_eventstree, gen_accept_count_eventstree, genb_accept_count_eventstree, df_GenbAccept_Eventstree)
        if  hist_type == "EventsDistribution1D":
            hist, yrange, fname= EventsDistribution1D_hist(mass_point, df_GenbAccept_Eventstree)
        if hist_type =="EventsDistribution2D": 
            hist, yrange, particle=EventsDistribution2D(mass_point, df_GenDL_Eventstree)
            ROOT.gStyle.SetOptStat(0)
            hist.Draw("COLZ")
            canvas.SaveAs(f"plots/{particle}_{mass_point}{mychannel}.png")
            continue
        if  hist_type == "comparecut":
            df=RecoHWWCandidateSelection(df_GenbAccept_Eventstree)
            hist= comparecut_hist(mass_point, df, initial_count) 
        
        if hist_type not in ("cutflow", "comparecut"):
            #including the overflow bin nbinsx+1 (in 2 places)
            #integral = hist.Integral()
            integral = hist.Integral(1, hist.GetNbinsX() + 1)
            if integral != 0:
                for bin_idx in range(1, hist.GetNbinsX() + 1):
                    bin_content = hist.GetBinContent(bin_idx)
                    if bin_content > 0:
                        error = ROOT.TMath.Sqrt(bin_content)
                    else:
                        error = 0  # No error for empty bins
                    hist.SetBinError(bin_idx, error)    
                hist.Scale(1.0 / integral)
            else:
                print("Warning: Histogram has zero integral, cannot normalize.")
            hist.GetYaxis().SetTitle("Normalized Distribution")
            hist.GetYaxis().SetRangeUser(0, yrange)
        
        overflow_bin_content = hist.GetBinContent(hist.GetNbinsX() + 1)
        print("Integral of overflow bin:", overflow_bin_content)

        legend.AddEntry(hist, f"M-{mass_point}", "l")
        hist.SetLineColor(colors[i])
        histograms.append(hist)

        if i == 0:
            #hist.Draw("HIST E")
            hist.Draw("HIST")
            ROOT.gStyle.SetOptStat(0)
        else:
            #hist.Draw("HIST E SAME")
            hist.Draw("HIST SAME")
            ROOT.gStyle.SetOptStat(0)

    legend.Draw()

    if hist_type =="cutflow":
            canvas.SaveAs(f"plots/1{mychannel}MatcNano{ctype}.png")
    if hist_type =="EventsDistribution1D": 
            canvas.SaveAs(f"plots/1{fname}_{mychbrief}_Combine.png")
    if hist_type =="comparecut": 
            canvas.SaveAs(f"plots/1{mychannel}comparecut_Combine.png")

######################################################################def of processing cutflow hist
def process_cutflow_hist(canvas, mass_point, initial_count, genDL_count_eventstree, gen_accept_count_eventstree, genb_accept_count_eventstree,df_GenbAccept_Eventstree):
    genDL_count= genDL_count_eventstree
    gen_accept_count= gen_accept_count_eventstree
    genb_accept_count= genb_accept_count_eventstree
    skim_count = genb_accept_count_eventstree

    df_cutflow=RecoHWWCandidateSelection(df_GenbAccept_Eventstree)
    #df_cutflow=RecoHWWJetSelection(df_cutflow)

    report=df_cutflow.Report()

    hist = Report_cutflow_hist(canvas, mass_point, report, initial_count, genDL_count, gen_accept_count, genb_accept_count, skim_count)
    return hist

######################################################################run options!
#mychannel = "dlmuon"
#mychannel = "dle"
mychannel = "dlmu_e"

if mychannel == "dlmuon":
    mychbrief="mumu"
    mychfull="Muon" 
    matchThresh=0.01
if mychannel == "dle":
    mychbrief="ee"
    mychfull="Electron"
    matchThresh=0.02
if mychannel == "dlmu_e":
    mychbrief="emu"
    mychfull="Muon"
    mychfull1="Electron"
    matchThresh=0.01
    matchThresh1=0.02

hist_type ="cutflow"
ctype = "Relative"
#ctype = "Cumulative"
#hist_type = "EventsDistribution1D"
#hist_type ="EventsDistribution2D"
#hist_type = "comparecut"

if hist_type =="EventsDistribution2D": 
    matchThresh=matchThresh1=1

main(mychannel, hist_type)



    

    
    


    






