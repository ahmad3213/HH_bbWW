# yumeng:
# This script is for SINGLE LEPTON efficiency study. 
# The basic steps are: checking input files (NANOAOD), processing events, and applying selection criteria.  
# The main tasks include generating histograms using functions like 
# Report_cutflow_hist,
# EventsDistribution1D_hist,  
# EventsDistribution2D_hist, 
# TEff_RecoGenMatch_hist,
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
        df=df.Define("Close0RecoLepIdx", f"FindMatching(H_to_VV.legs.at(0).leg_vis_p4.at(0), {mychfull}_p4, 1)")


    else:
        df = df.Define("genLeptons", """reco_tau::gen_truth::GenLepton::fromNanoAOD(GenPart_pt, GenPart_eta,
                                    GenPart_phi, GenPart_mass, GenPart_genPartIdxMother, GenPart_pdgId,
                                    GenPart_statusFlags,-1)""")
        df = df.Define("H_to_VV", """GetGenHVVCandidate(-1,genLeptons, GenPart_pdgId, GenPart_daughters, 
                                  GenPart_statusFlags, GenPart_pt, GenPart_eta, GenPart_phi, GenPart_mass, GenJet_p4, true)""")
        df = df.Define("H_to_BB", """GetGenHBBCandidate(-1, GenPart_pdgId, GenPart_daughters, GenPart_statusFlags, GenPart_pt, GenPart_eta, GenPart_phi, GenPart_mass, GenJet_p4, true)""")
    
    df = df.Define("Leg0_Kind", "H_to_VV.legs.at(0).leg_kind.at(0)")
    df= df.Define("GenLep0_p4","H_to_VV.legs.at(0).leg_vis_p4.at(0)")
    
    df= df.Define("Genb0_p4","H_to_BB.leg_p4.at(0)")
    df= df.Define("Genb1_p4","H_to_BB.leg_p4.at(1)")

    if mychannel == "sle":
        #df_GenSL=df.Filter("Leg0_Kind==Vleg::PromptElectron|| Leg0_Kind==Vleg::TauDecayedToElectron", "")
        df_GenSL=df.Filter("Leg0_Kind==Vleg::PromptElectron", "")
        genSL_count = get_event_counts(df_GenSL)

        #df_Gen_Accept = df_GenSL.Filter("GenLep0_p4.pt()>10 && abs (GenLep0_p4.eta()) <2.5", "")
        df_Gen_Accept = df_GenSL.Filter("GenLep0_p4.pt()>5 && abs (GenLep0_p4.eta()) <2.5", "")
        gen_accept_count = get_event_counts(df_Gen_Accept)
        
    if mychannel == "slmuon":
        #df_GenSL=df.Filter("Leg0_Kind==Vleg::PromptMuon|| Leg0_Kind==Vleg::TauDecayedToMuon", "")
        df_GenSL=df.Filter("Leg0_Kind==Vleg::PromptMuon", "")
        genSL_count = get_event_counts(df_GenSL)

        #df_Gen_Accept = df_GenSL.Filter("GenLep0_p4.pt()>15 && abs (GenLep0_p4.eta()) <2.4", "")
        df_Gen_Accept = df_GenSL.Filter("GenLep0_p4.pt()>5 && abs (GenLep0_p4.eta()) <2.5", "")
        gen_accept_count = get_event_counts(df_Gen_Accept)
    
    df_Genb_Accept = df_Gen_Accept.Filter("Genb0_p4.pt()>20 && Genb1_p4.pt()>20 && abs (Genb0_p4.eta()) <2.5 && abs (Genb1_p4.eta()) <2.5", "")
    genb_accept_count = get_event_counts(df_Genb_Accept)

    df_GenWq_Accept = df_Genb_Accept.Filter("H_to_VV.legs.at(1).leg_p4.at(0).pt()>20 && H_to_VV.legs.at(1).leg_p4.at(1).pt()>20 && abs (H_to_VV.legs.at(1).leg_p4.at(0).eta()) <5 && abs (H_to_VV.legs.at(1).leg_p4.at(1).eta()) <5", "")
    genWq_accept_count = get_event_counts(df_GenWq_Accept)

    return df_GenSL, df_Gen_Accept, df_Genb_Accept, df_GenWq_Accept, genSL_count, gen_accept_count, genb_accept_count, genWq_accept_count

def RecoHWWCandidateSelection(df):
    def add_filter(df, base_cut, and_cut, label):
        full_cut = f"Sum({base_cut} && {and_cut}) > 0"
        df = df.Filter(full_cut, label)
        return df, f"{base_cut} && {and_cut}"
    
    if mychannel == "sle":
        df=df.Filter("nElectron>0", "HasRecoElectron")
        df=df.Define("RecoLepIdx", "FindMatching(GenLep0_p4, Electron_p4, 0.2)")
        df=df.Filter("RecoLepIdx>=0", "Reco_e_MatchGen")
        df = df.Filter("Electron_pt.at(RecoLepIdx)>5", "Reco_e_pt>5")#10
        df = df.Filter("Electron_eta.at(RecoLepIdx)<2.5", "Reco_e_eta<2.5")
        df = df.Filter("Electron_dz.at(RecoLepIdx)<0.1", "Reco_e_dz<0.1")
        df = df.Filter("Electron_dxy.at(RecoLepIdx)<0.05", "Reco_e_dxy<0.05")
        df = df.Filter("Electron_sip3d.at(RecoLepIdx) <= 8", "Reco_e_sip3d <= 8")

    if mychannel== "slmuon":
        df=df.Filter("nMuon>0", "HasRecoMuon")
        df=df.Define("RecoLepIdx", "FindMatching(GenLep0_p4, Muon_p4, 0.01)")
        df=df.Filter("RecoLepIdx>=0", "Reco_mu_MatchGen")
        df = df.Filter("Muon_pt.at(RecoLepIdx)>5", "Reco_mu_pt>5")  #15
        df = df.Filter("Muon_eta.at(RecoLepIdx)<2.5", "Reco_mu_eta<2.5") #2.4
        df = df.Filter("Muon_dz.at(RecoLepIdx)<0.1", "Reco_mu_dz<0.1")
        df = df.Filter("Muon_dxy.at(RecoLepIdx)<0.05", "Reco_mu_dxy<0.05")
        df = df.Filter("Muon_sip3d.at(RecoLepIdx) <= 8", "Reco_mu_sip3d <= 8")
    return df

def Report_cutflow_hist(canvas, mass_point, report, initial_count, genSL_count,  gen_accept_count, genb_accept_count, genWq_accept_count, skim_count, reportName="Cutflow", printOut=False):
    cuts = [c for c in report]
    num_cuts = len(cuts)

    hist_eff = ROOT.TH1D(reportName + f"__M-{mass_point}" + "_Efficiency", reportName+f" {ctype}"+" Efficiency" + "_for Masspoints", num_cuts + 4, 0, num_cuts + 4)
    hist_eff.GetXaxis().SetBinLabel(1, "Initial_NANOAOD")
    hist_eff.SetBinContent(1, 1.0)
    hist_eff.SetBinError(1, 0.0) 
    hist_eff.SetMinimum(0)

    if ctype == "Relative":
        hist_eff.GetXaxis().SetBinLabel(2, f"GenSL{mychbrief}")
        hist_eff.SetBinContent(2, genSL_count/initial_count)

        hist_eff.GetXaxis().SetBinLabel(3, f"Gen_{mychbrief}_Lep_Accept")
        hist_eff.SetBinContent(3, gen_accept_count/genSL_count)

        hist_eff.GetXaxis().SetBinLabel(4, f"Gen_b_Accept")
        hist_eff.SetBinContent(4, genb_accept_count/gen_accept_count)

        hist_eff.GetXaxis().SetBinLabel(5, f"Gen_Wq_Accept")
        hist_eff.SetBinContent(5, genWq_accept_count/genb_accept_count)

        for c_id, cut in enumerate(cuts):
                p = cut.GetEff() / 100
                n = cut.GetAll()
                binomial_error = ROOT.TMath.Sqrt(p * (1 - p) / n)
                hist_eff.SetBinContent(c_id + 6, cut.GetEff()/100)
                hist_eff.GetXaxis().SetBinLabel(c_id + 6, cut.GetName())
                hist_eff.SetBinError(c_id + 6, binomial_error)
    
    if ctype == "Cumulative":
        hist_eff.GetXaxis().SetBinLabel(2, f"GenSL{mychbrief}")
        hist_eff.SetBinContent(2, genSL_count/initial_count)

        hist_eff.GetXaxis().SetBinLabel(3, f"Gen_{mychbrief}_Lep_Accept")
        hist_eff.SetBinContent(3, gen_accept_count/initial_count)

        hist_eff.GetXaxis().SetBinLabel(4, f"Gen_b_Accept")
        hist_eff.SetBinContent(4, genb_accept_count/initial_count)

        hist_eff.GetXaxis().SetBinLabel(5, f"Gen_Wq_Accept")
        hist_eff.SetBinContent(5, genWq_accept_count/initial_count)

        for c_id, cut in enumerate(cuts):
                p = cut.GetPass()/initial_count
                n = initial_count
                binomial_error = ROOT.TMath.Sqrt(p * (1 - p) / n)
                hist_eff.SetBinContent(c_id + 6, cut.GetPass()/initial_count)
                hist_eff.GetXaxis().SetBinLabel(c_id + 6, cut.GetName())
                hist_eff.SetBinError(c_id + 6, binomial_error)
    
    hist_eff.GetXaxis().SetLabelSize(0.03);  
    
    #hist_eff.GetXaxis().LabelsOption("v");
    canvas.SetBottomMargin(0.25);

    return hist_eff

def EventsDistribution1D_hist(mass_point,df):  
    df = df.Define("HVVCandidatept", "H_to_VV.cand_p4.Pt()")
    df = df.Define("FatJet_p4", "yumengGetP4(FatJet_pt, FatJet_eta, FatJet_phi, FatJet_mass)")
    df = df.Define("mindRGenlepFatjet","FindMatchingdR(H_to_VV.legs.at(0).leg_vis_p4.at(0), FatJet_p4, 100)")
    df = df.Define("dRGenlepVh","ROOT::Math::VectorUtil::DeltaR(H_to_VV.legs.at(0).leg_vis_p4.at(0), H_to_VV.legs.at(1).cand_p4)")

    #variable changes:
    pt_min = 0
    pt_max = 1
    bins = int((pt_max - pt_min)/10)

    #variable changes:
    ###df = df.Filter("mindRGenlepFatjet<6")
    ###df=df.Filter("HVVCandidatept<700", "")
    yrange=0.25
    
    hist_pt = df.Histo1D((f"mindR(GenLep, Fatjet){mass_point}", f"mindR(GenLep, Fatjet), SL{mychbrief} channel", 50, 0, 0.8), "mindRGenlepFatjet")
    hist_pt.GetXaxis().SetTitle(f"mindR(GenLep, Fatjet), SL{mychbrief} channel)")
    fname="MtNanodRFat" 
   
    hist_pt = hist_pt.GetValue()

    return hist_pt, yrange, fname

def EventsDistribution2D_hist(mass_point,df):
    df = df.Define("HVVCandidatePt", "H_to_VV.cand_p4.Pt()") 
    
    df = df.Define("VCandidatePt", "H_to_VV.legs.at(0).cand_p4.Pt()")
    
    df = df.Define("VCandidate2Pt", "H_to_VV.legs.at(1).cand_p4.Pt()")

    df= df.Define("neutrino0_p4","H_to_VV.legs.at(0).leg_p4.at(1)")
    df= df.Define("GenLep0Pt","GenLep0_p4.pt()")
    df= df.Define("neutrino0Pt","neutrino0_p4.pt()")
    df= df.Define("GenLepPlusNuPt","(GenLep0_p4+neutrino0_p4).pt()")
    
    df= df.Define("GenLep0E","GenLep0_p4.E()")
    df= df.Define("neutrino0E","neutrino0_p4.E()")    

    df = df.Define("HBBCandidatePt", "H_to_BB.cand_p4.Pt()")

    df = df.Define("dR_GenRecoLep", f"ROOT::Math::VectorUtil::DeltaR(GenLep0_p4, {mychfull}_p4.at(Close0RecoLepIdx))");
    df = df.Define("PtRatio_GenReco0Lep", f"GenLep0Pt / {mychfull}_pt.at(Close0RecoLepIdx)");
    #df.Display(["dR_GenRecoLep", "PtRatio_GenRecoLep"]).Print()

    # #particle name should be consistent with df
    # particle= "HVVCandidatePt"
    # pt_max =3000
    # yrange=0.62
    # bin_wid=30

    # particle= "VCandidatePt"
    # pt_max=2000
    # #yrange=0.82
    # yrange=0.9
    # bin_wid=50

    # particle= "HBBCandidatePt"
    # pt_max =3000
    # yrange=0.62
    # bin_wid=30

    #particle= "VCandidatePt"
    #particle2= "GenLepPlusNuPt"
    # particle= "neutrino0E"
    # particle2= "GenLep0E"
    # particle= "neutrino0Pt"
    # particle2= "GenLep0Pt"
    # pt_max=2000
    # yrange=0.9
    # bin_wid=50

    df=df.Filter("Close0RecoLepIdx>=0", "")
    particle= "PtRatio_GenReco0Lep"
    particle2= "dR_GenRecoLep"
    xmax=1.5
    ymax=0.05
    yrange=0.9
    bin_wid=0.001
   
    pt_min = 0
    
    binsx=int((xmax - pt_min)/bin_wid)
    binsy=int((ymax - pt_min)/bin_wid)
    model_2d = ROOT.RDF.TH2DModel(f"{particle2}_vs_{particle}_M-{mass_point}", f"{particle2} vs {particle} M-{mass_point}({mychannel})", 
                                  binsx, pt_min, xmax,
                                  binsy, pt_min, ymax)

    hist_pt= df.Histo2D(model_2d, f"{particle}", f"{particle2}")

    hist_pt.GetXaxis().SetTitle(f"{particle}(GeV)({mychannel} channel)")
    hist_pt.GetYaxis().SetTitle(f"{particle2}(GeV)({mychannel} channel)")

    return hist_pt, yrange, particle

def TEff_RecoGenMatch_hist(mass_point,df):
    df = df.Define("GenLep0pt", "GenLep0_p4.pt()") 
    df = df.Define("VCandidatept", "H_to_VV.legs.at(0).cand_p4.Pt()")    
    df = df.Define("HVVCandidatept", "H_to_VV.cand_p4.Pt()")
    df = df.Define("GenLep0eta", "GenLep0_p4.eta()") 
    df = df.Define("VCandidateeta", "H_to_VV.legs.at(0).cand_p4.eta()")    
    df = df.Define("HVVCandidateeta", "H_to_VV.cand_p4.eta()")
    # df = df.Define("FatJet_p4", "yumengGetP4(FatJet_pt, FatJet_eta, FatJet_phi, FatJet_mass)")
    # df = df.Define("mindRGenlepFatjet","FindMatchingdR(H_to_VV.legs.at(0).leg_vis_p4.at(0), FatJet_p4, 100)")
    df = df.Define("dRGenlepVh","ROOT::Math::VectorUtil::DeltaR(H_to_VV.legs.at(0).leg_vis_p4.at(0), H_to_VV.legs.at(1).cand_p4)")
    df = df.Define("Jet_p4", "yumengGetP4(Jet_pt, Jet_eta, Jet_phi, Jet_mass)")
    df = df.Define("mindRGenlepJet","FindMatchingdR(H_to_VV.legs.at(0).leg_vis_p4.at(0), Jet_p4, 100)")
    
    # particl="VCandidate"
    # particl="GenLep0"
    # pt_max=2000
    
    # particl="HVVCandidate"
    # pt_max=3000

    #particl= "mindRGenlepFatjet"
    #particl= "mindRGenlepJet"
    particl= "dRGenlepVh"
    pt_max=0.5
    
    pt_min = 0

    var_min = pt_min 
    var_max = pt_max

    #bins= int((var_max - var_min)/50)
    bins=50

    if particl== "mindRGenlepFatjet":
        var1="mindR(Genlep,Fatjets)"
    if particl== "mindRGenlepJet":
        var1="mindR(Genlep,Jets)"
    if particl== "mindRGenlepFatjet":
        var1="mindR(Genlep,Fatjets)"
    if particl== "dRGenlepVh":
        var1="dR(Genlep,Hadronic W)"

    df=df.Define("RecoLepIdx", f"FindMatching(GenLep0_p4, {mychfull}_p4, 0.02)")
    df_sz=df.Filter("nElectron>0", "")
    legend = ROOT.TLegend(0.7, 0.8, 0.9, 0.9)
    
    #variable changes, getvalue for entry
    h_before = df.Histo1D(("h_before", f"{var1} Distribution", bins, var_min, var_max), f"{particl}").GetValue()
    h_after = df_sz.Histo1D(("h_after", f"HasRecoE_Eff vs {var1} (M-{mass_point})", bins, var_min, var_max), f"{particl}").GetValue()
    hist_pt = h_before.Clone("h_ratio")
    hist_pt.SetTitle(f"Has_e_Eff vs {var1} M-{mass_point}")
    hist_pt.GetYaxis().SetTitle(f"Eff (after HasRecoElectron)")
    legend.AddEntry(h_before, "Before Has_e", "l")
    legend.AddEntry(h_after, "After Has_e", "l")
    
    h_before.SetBinContent(bins, h_before.GetBinContent(bins) + h_before.GetBinContent(bins + 1))
    h_after.SetBinContent(bins, h_after.GetBinContent(bins) + h_after.GetBinContent(bins + 1))

    hist_pt = ROOT.TEfficiency(h_after, h_before)

    c = ROOT.TCanvas("c", "canvas", 1200, 600)
    c.Divide(2,1)
    c.SetGrid()

    pad1 = c.cd(1)
    ROOT.gStyle.SetOptStat(0)
    pad1.SetPad(0.01, 0.01, 0.5, 0.99)
    pad1.SetGrid()
    h_before.SetLineColor(ROOT.kGreen)
    h_after.SetLineColor(ROOT.kRed)
    h_before.Draw()
    h_after.Draw("same")
    legend.Draw()
    h_before.GetXaxis().SetTitle(f"{var1}, {mychannel} Channel")
    h_before.GetYaxis().SetTitle(f"Counts")

    pad2 = c.cd(2)
    pad2.SetGrid()
    pad2.SetPad(0.5, 0.01, 0.99, 0.99)
    hist_pt.SetTitle(f"Has_e_Eff vs {var1} M-{mass_point};{var1}, SL{mychbrief} Channel;Eff(after HasRecoElectron)")
    hist_pt.SetLineColor(ROOT.kBlack)
    hist_pt.Draw("AP")
    
    yrange=1.1

    c.SaveAs(f"plots/1{mychbrief}{mass_point}{particl}{var1}_Eff_MtNano.png")

    return hist_pt,yrange

def comparecut_hist(mass_point, df):
    if mychannel == "sle":
        cuts = [
        f"{mychfull}_cutBased.at(RecoLepIdx) >= 1",  
        f"{mychfull}_cutBased.at(RecoLepIdx) >= 2",
        f"{mychfull}_cutBased.at(RecoLepIdx) >= 3",
        f"{mychfull}_cutBased.at(RecoLepIdx) >= 4",
        f"{mychfull}_mvaIso_WP80.at(RecoLepIdx)",
        f"{mychfull}_mvaIso_WP90.at(RecoLepIdx)",
        #f"{mychfull}_mvaIso_WPHZZ.at(RecoLepIdx)",
        f"{mychfull}_mvaNoIso_WP80.at(RecoLepIdx) && {mychfull}_pfRelIso03_all.at(RecoLepIdx)<0.4",
        f"{mychfull}_mvaNoIso_WP80.at(RecoLepIdx) && {mychfull}_pfRelIso03_all.at(RecoLepIdx)<0.25",
        f"{mychfull}_mvaNoIso_WP80.at(RecoLepIdx) && {mychfull}_pfRelIso03_all.at(RecoLepIdx)<0.2",
        f"{mychfull}_mvaNoIso_WP80.at(RecoLepIdx) && {mychfull}_pfRelIso03_all.at(RecoLepIdx)<0.15",
        f"{mychfull}_mvaNoIso_WP80.at(RecoLepIdx) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx)<0.4",
        f"{mychfull}_mvaNoIso_WP80.at(RecoLepIdx) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx)<0.25",
        f"{mychfull}_mvaNoIso_WP80.at(RecoLepIdx) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx)<0.2",
        f"{mychfull}_mvaNoIso_WP80.at(RecoLepIdx) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx)<0.15",
        f"{mychfull}_mvaNoIso_WP90.at(RecoLepIdx) && {mychfull}_pfRelIso03_all.at(RecoLepIdx)<0.4",
        f"{mychfull}_mvaNoIso_WP90.at(RecoLepIdx) && {mychfull}_pfRelIso03_all.at(RecoLepIdx)<0.25",
        f"{mychfull}_mvaNoIso_WP90.at(RecoLepIdx) && {mychfull}_pfRelIso03_all.at(RecoLepIdx)<0.2",
        f"{mychfull}_mvaNoIso_WP90.at(RecoLepIdx) && {mychfull}_pfRelIso03_all.at(RecoLepIdx)<0.15",
        f"{mychfull}_mvaNoIso_WP90.at(RecoLepIdx) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx)<0.4",
        f"{mychfull}_mvaNoIso_WP90.at(RecoLepIdx) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx)<0.25",
        f"{mychfull}_mvaNoIso_WP90.at(RecoLepIdx) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx)<0.2",
        f"{mychfull}_mvaNoIso_WP90.at(RecoLepIdx) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx)<0.15",
        ]

        labels = [
        "cutBased>=veto",
        "cutBased>=Loose",
        "cutBased>=Medium",
        "cutBased>=Tight",
        "MVA Iso WP80",
        "MVA Iso WP90",
        #"MVA Iso WPHZZ",
        "mvaNoIso_80 && pfIso03_all<0.4",
        "mvaNoIso_80 && pfIso03_all<0.25 ",
        "mvaNoIso_80 && pfIso03_all<0.2 ",
        "mvaNoIso_80 && pfIso03_all<0.15 ",
        "mvaNoIso_80 && miniPF_all<0.4",
        "mvaNoIso_80 && miniPF_all<0.25 ",
        "mvaNoIso_80 && miniPF_all<0.2 ",
        "mvaNoIso_80 && miniPF_all<0.15 ",
        "mvaNoIso_90 && pfIso03_all<0.4",
        "mvaNoIso_90 && pfIso03_all<0.25 ",
        "mvaNoIso_90 && pfIso03_all<0.2 ",
        "mvaNoIso_90 && pfIso03_all<0.15 ",
        "mvaNoIso_90 && miniPF_all<0.4",
        "mvaNoIso_90 && miniPF_all<0.25 ",
        "mvaNoIso_90 && miniPF_all<0.2 ",
        "mvaNoIso_90 && miniPF_all<0.15 ",   
        ]

    if mychannel == "slmuon":
        cuts = [
        # Loose ID && PFRelIso cuts
        f"{mychfull}_looseId.at(RecoLepIdx) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx) < 0.4 ",  # Loose ID, Isolation < 0.4
        f"{mychfull}_looseId.at(RecoLepIdx) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx) < 0.25 ", # Loose ID, Isolation < 0.25
        f"{mychfull}_looseId.at(RecoLepIdx) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx) < 0.2 ",  # Loose ID, Isolation < 0.2
        f"{mychfull}_looseId.at(RecoLepIdx) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx) < 0.15", # Loose ID, Isolation < 0.15

        # Medium ID && PFRelIso cuts
        f"{mychfull}_mediumId.at(RecoLepIdx) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx) < 0.4 ",  # Medium ID, Isolation < 0.4
        f"{mychfull}_mediumId.at(RecoLepIdx) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx) < 0.25", # Medium ID, Isolation < 0.25
        f"{mychfull}_mediumId.at(RecoLepIdx) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx) < 0.2 ",  # Medium ID, Isolation < 0.2
        f"{mychfull}_mediumId.at(RecoLepIdx) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx) < 0.15", # Medium ID, Isolation < 0.15

        # Tight ID && PFRelIso cuts
        f"{mychfull}_tightId.at(RecoLepIdx) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx) < 0.4 ",  # Tight ID, Isolation < 0.4
        f"{mychfull}_tightId.at(RecoLepIdx) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx) < 0.25", # Tight ID, Isolation < 0.25
        f"{mychfull}_tightId.at(RecoLepIdx) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx) < 0.2  ",  # Tight ID, Isolation < 0.2
        f"{mychfull}_tightId.at(RecoLepIdx) && {mychfull}_miniPFRelIso_all.at(RecoLepIdx) < 0.15", # Tight ID, Isolation < 0.15

        # tracker highPt ID && PFRelIso cuts
        f"{mychfull}_highPtId.at(RecoLepIdx)>=1 && {mychfull}_miniPFRelIso_all.at(RecoLepIdx) < 0.4 ", 
        f"{mychfull}_highPtId.at(RecoLepIdx)>=1 && {mychfull}_miniPFRelIso_all.at(RecoLepIdx) < 0.25 ",
        f"{mychfull}_highPtId.at(RecoLepIdx)>=1 && {mychfull}_miniPFRelIso_all.at(RecoLepIdx) < 0.2 ", 
        f"{mychfull}_highPtId.at(RecoLepIdx)>=1 && {mychfull}_miniPFRelIso_all.at(RecoLepIdx) < 0.15", 

        # tracker highPt ID && tkIsoId cuts
        f"{mychfull}_highPtId.at(RecoLepIdx)>=1 && {mychfull}_tkIsoId.at(RecoLepIdx) >= 1 ", 
        f"{mychfull}_highPtId.at(RecoLepIdx)>=1 && {mychfull}_tkIsoId.at(RecoLepIdx) >= 2 ",

        # global highPt ID && PFRelIso cuts
        f"{mychfull}_highPtId.at(RecoLepIdx)>=2 && {mychfull}_miniPFRelIso_all.at(RecoLepIdx) < 0.4 ", 
        f"{mychfull}_highPtId.at(RecoLepIdx)>=2 && {mychfull}_miniPFRelIso_all.at(RecoLepIdx) < 0.25 ",
        f"{mychfull}_highPtId.at(RecoLepIdx)>=2 && {mychfull}_miniPFRelIso_all.at(RecoLepIdx) < 0.2 ", 
        f"{mychfull}_highPtId.at(RecoLepIdx)>=2 && {mychfull}_miniPFRelIso_all.at(RecoLepIdx) < 0.15", 
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

    total_events = df.Count().GetValue()
    efficiencies = []
    for cut, label in zip(cuts, labels):
        filtered_df = df.Filter(cut, label)
        passed_events = filtered_df.Count().GetValue()
        efficiency = passed_events / total_events if total_events > 0 else 0
        efficiencies.append(efficiency)
        print(f"M-{mass_point}, {label} cut relative eff:{efficiency}")

    hist = ROOT.TH1F(f"Relativre Eff for diff cuts{mass_point}", "Relative Eff for diff cuts", len(labels), 0, len(labels))

    for i, (label, efficiency) in enumerate(zip(labels, efficiencies), start=1):
        hist.SetBinContent(i, efficiency)
        hist.GetXaxis().SetBinLabel(i, label)

    hist.GetXaxis().SetTitle("Cuts")
    hist.GetYaxis().SetTitle("Relative Efficiency")
    hist.GetXaxis().SetLabelSize(0.03)
    hist.GetXaxis().LabelsOption("v");
    #hist.GetXaxis().LabelsOption("a");
    hist.SetMinimum(0)
    hist.SetMaximum(1)
    return hist


######################################################################def of main functions starts:
def main(mychannel, hist_type):
    mass_points = [250, 450, 650, 1000, 3000, 5000]
    colors = [ROOT.kRed, ROOT.kBlue, ROOT.kGreen, ROOT.kMagenta, ROOT.kCyan, ROOT.kOrange]
    #mass_points = [5000]

    histograms = []
    canvas = ROOT.TCanvas("canvas", "Mass Points Comparison", 800, 600)
    canvas.SetBottomMargin(0.3)
    canvas.SetGrid()
    yrange=1
    legend = ROOT.TLegend(0.75, 0.9, 1, 1)
    legend.SetNColumns(2)

    for i, mass_point in enumerate(mass_points):
        file_path = f"/eos/user/y/yumeng/ana/flaf/2022run3nanoaod/GluGlutoRadiontoHHto2B2Vto2B2JLNu_M_hadd/hadd{mass_point}.root"
        if check_root_file(file_path):
            df_eventstree = ROOT.RDataFrame("Events", file_path)
            eventscount=get_event_counts(df_eventstree)
        #initial_count = eventscount + notselectedcount
        initial_count = eventscount

        df_GenSL_Eventstree, df_GenAccept_Eventstree, df_GenbAccept_Eventstree, df_GenWqAccept_Eventstree, genSL_count_eventstree, gen_accept_count_eventstree, genb_accept_count_eventstree, genWq_accept_count_eventstree = process_tree_Gen(file_path, "Events")

        if hist_type =="cutflow":
            hist=process_cutflow_hist(canvas, mass_point, initial_count, genSL_count_eventstree, gen_accept_count_eventstree, genb_accept_count_eventstree, genWq_accept_count_eventstree, df_GenWqAccept_Eventstree)
        if  hist_type == "EventsDistribution1D":
            hist, yrange, fname= EventsDistribution1D_hist(mass_point, df_GenWqAccept_Eventstree)
        if  hist_type == "EventsDistribution2D":
            hist, yrange, particle=EventsDistribution2D_hist(mass_point, df_GenSL_Eventstree)
            ROOT.gStyle.SetOptStat(0)
            hist.Draw("COLZ")
            canvas.SaveAs(f"plots/1{mychannel}_{particle}_{mass_point}.png")
            continue
        if  hist_type == "TEff_RecoGenMatch":
            hist, yrange= TEff_RecoGenMatch_hist(mass_point, df_GenWqAccept_Eventstree)
            continue
        if  hist_type == "comparecut":
            df=RecoHWWCandidateSelection(df_GenWqAccept_Eventstree)
            hist= comparecut_hist(mass_point, df)
        
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
            canvas.SaveAs(f"plots/1{mychbrief}{fname}_Combine.png")            
    if hist_type =="comparecut": 
            canvas.SaveAs(f"plots/1{mychbrief}comparecut_Combine.png")    

######################################################################def of processing cutflow hist
def process_cutflow_hist(canvas, mass_point, initial_count, genSL_count_eventstree, gen_accept_count_eventstree, genb_accept_count_eventstree, genWq_accept_count_eventstree, df_GenWqAccept_Eventstree):
    # genSL_count= genSL_count_eventstree + genSL_count_notselected
    # gen_accept_count= gen_accept_count_eventstree + gen_accept_count_notselected
    # genb_accept_count= genb_accept_count_eventstree + genb_accept_count_notselected
    # genWq_accept_count= genWq_accept_count_eventstree + genWq_accept_count_notselected
    genSL_count= genSL_count_eventstree
    gen_accept_count= gen_accept_count_eventstree
    genb_accept_count= genb_accept_count_eventstree
    genWq_accept_count= genWq_accept_count_eventstree
    skim_count = genWq_accept_count_eventstree

    df_cutflow=RecoHWWCandidateSelection(df_GenWqAccept_Eventstree)

    report=df_cutflow.Report()

    hist = Report_cutflow_hist(canvas, mass_point, report, initial_count, genSL_count, gen_accept_count, genb_accept_count, genWq_accept_count, skim_count)
    return hist
    
######################################################################run options!
#mychannel = "sle"
mychannel = "slmuon"

if mychannel == "sle":
    mychbrief="e"
    mychfull="Electron"
if mychannel == "slmuon":
    mychbrief="Mu"
    mychfull="Muon"

hist_type ="cutflow"
#ctype = "Relative"
ctype = "Cumulative"
#hist_type = "EventsDistribution1D"
#hist_type = "EventsDistribution2D"
hist_type = "TEff_RecoGenMatch"
#hist_type = "comparecut"

main(mychannel, hist_type)



    

    
    


    






