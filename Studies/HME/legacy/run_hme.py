import ROOT
import os
import sys
import tqdm
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

sys.argv.append( '-b' )
sys.argv.append( '-q' )

#sys.path.append(os.path.join(os.path.abspath('..'), 'python'))
from hme import *


iterations = 1000
stop_watch = ROOT.TStopwatch()
#real events from radion signal with M=4, narrow width
lep1_p4 = ROOT.TLorentzVector();
lep2_p4 = ROOT.TLorentzVector();
jet1_p4 = ROOT.TLorentzVector();
jet2_p4 = ROOT.TLorentzVector();
met_vec2 = ROOT.TVector2()



#f = ROOT.TFile("../GluGlutoRadiontoHHto2B2Vto2B2L2Nu_M450_Run3Sync.root")
#events = f.Get("Double_Tree")

f = ROOT.TFile("../DL_450_example.root")
events = f.Get("Events")
hme_mass_list = []

out = ROOT.TFile("hme_legacy_output.root", "recreate")
out.cd()
tree = events.CloneTree(0)

for i, ev in enumerate(tqdm.tqdm(events)):
    #print("Event number ", i, " out of ", events.GetEntries())
    #if ev.Double_Signal == 0: continue
    if ev.ncentralJet < 2: continue
    if not ((ev.lep1_pt > 0) and (ev.lep2_pt > 0) and (ev.centralJet_pt[0] > 0) and (ev.centralJet_pt[1] > 0)): continue

    lep1_p4.SetPtEtaPhiM(ev.lep1_pt, ev.lep1_eta, ev.lep1_phi, ev.lep1_mass);
    lep2_p4.SetPtEtaPhiM(ev.lep2_pt, ev.lep2_eta, ev.lep2_phi, ev.lep2_mass);
    jet1_p4.SetPtEtaPhiM(ev.centralJet_pt[0], ev.centralJet_eta[0], ev.centralJet_phi[0], ev.centralJet_mass[0]);
    jet2_p4.SetPtEtaPhiM(ev.centralJet_pt[1], ev.centralJet_eta[1], ev.centralJet_phi[1], ev.centralJet_mass[1]);
    #met_vec2.Set(ev.met_px, ev.met_py);
    met_vec2.SetMagPhi(ev.PuppiMET_pt, ev.PuppiMET_phi);

    """
    boosted region, signal mass = 1.6TeV
    also use no bjet correction + smear MET
        hme.setBjetCorrectionType(0)
        hme.setMETCorrectionType(6)
    lep1_p4.SetPxPyPzE(-92.187746,54.858173,321.941003,339.343497);
    lep2_p4.SetPxPyPzE(-13.944575,12.626226,101.495754,103.224322);
    jet1_p4.SetPxPyPzE(272.369183,-122.649673,-353.965748,465.031469);
    jet2_p4.SetPxPyPzE(0.0, 0.0, 0.0, 0.0);
    met_vec2.Set(-194.524071,9.588025)
    """

    runHME = True
    truemass = 450.0
    nEv = 1
    if runHME:
        hme = HeavyMassEstimator()
        hme.setKinematic(lep1_p4, lep2_p4, jet1_p4, jet2_p4, met_vec2, 0.0)
        if hme.debug: hme.showKinematic()
        hme.setIterations(iterations)
        #hme.setDebug(True)
        hme.runHME()
        #hme.hme_offshellWmass.SetName("hme_offshellWmass_TCha.d_genlTCha.e"%nEv)
        if hme.hme_h2Mass.GetEntries() <= 0:
            print("NO solution found!!!!! ")
        elif hme.hme_h2Mass.GetEntries() > 0 and  hme.hme_h2Mass.GetXaxis().GetBinCenter(hme.hme_h2Mass.GetMaximumBin()) < 250.0 :
            print("Num solutions ",hme.hme_h2Mass.GetEntries()," BUT the maximum is ",hme.hme_h2Mass.GetXaxis().GetBinCenter(hme.hme_h2Mass.GetMaximumBin()))
        #hme.hme_h2Mass.Print("ALL")

        if hme.hme_h2Mass.GetEntries()> 0 and hme.hme_h2Mass.GetXaxis().GetBinCenter(hme.hme_h2Mass.GetMaximumBin())>=250.0 :
            #print "Reco Level most probable reco mass ",hme.hme_h2Mass.GetXaxis().GetBinCenter(hme.hme_h2Mass.GetMaximumBin())," entries ",hme.hme_h2Mass.GetEntries()," stddev ",hme.hme_h2Mass.GetStdDev(1)
            hme.hme_h2Mass.SetName("hme_h2Mass_ev%d_recolevel"%nEv)
            hme.hme_h2MassWeight1.SetName("hme_h2MassWeight1_ev%d_recolevel"%nEv)
            hme.hme_offshellWmass.SetName("hme_offshellWmass_ev%d_recolevel"%nEv)

            hme_h2mass_reco = hme.hme_h2Mass.GetXaxis().GetBinCenter(hme.hme_h2Mass.GetMaximumBin())
            hme_mean_reco = hme.hme_h2Mass.GetMean()
            hme_stddev_reco = hme.hme_h2Mass.GetStdDev(1)
            hme_entries_reco = float(hme.hme_h2Mass.GetEntries())/iterations
            hme_entry_peak_reco = hme.hme_h2Mass.Integral(hme.hme_h2Mass.GetMaximumBin()-5, hme.hme_h2Mass.GetMaximumBin()+5)

            hme_h2mass_weight1_reco = hme.hme_h2MassWeight1.GetXaxis().GetBinCenter(hme.hme_h2MassWeight1.GetMaximumBin())
            hme_mean_weight1_reco = hme.hme_h2MassWeight1.GetMean()
            hme_stddev_weight1_reco = hme.hme_h2MassWeight1.GetStdDev(1)
            hme_entries_weight1_reco = float(hme.hme_h2MassWeight1.GetEntries())/iterations
            hme_entry_peak_weight1_reco = hme.hme_h2MassWeight1.Integral(hme.hme_h2MassWeight1.GetMaximumBin()-5, hme.hme_h2MassWeight1.GetMaximumBin()+5)

            hme_h2mass_weight2_reco = hme.hme_h2MassWeight2.GetXaxis().GetBinCenter(hme.hme_h2MassWeight2.GetMaximumBin())
            hme_mean_weight2_reco = hme.hme_h2MassWeight2.GetMean()
            hme_stddev_weight2_reco = hme.hme_h2MassWeight2.GetStdDev(1)
            hme_entries_weight2_reco = float(hme.hme_h2MassWeight2.GetEntries())/iterations
            hme_entry_peak_weight2_reco = hme.hme_h2MassWeight2.Integral(hme.hme_h2MassWeight2.GetMaximumBin()-5, hme.hme_h2MassWeight2.GetMaximumBin()+5)
            if hme.debug:
                print("True HH mass ",truemass,"; reconstructed HH mass ",hme_h2mass_reco, " +/- ",hme_stddev_reco)
                print("\t reconstructed HH mass with type 1 weight ", hme_h2mass_weight1_reco," +/- ",hme_stddev_weight1_reco)
                print("\t reconstructed HH mass with type 2 weight ", hme_h2mass_weight2_reco," +/- ",hme_stddev_weight2_reco)
            hme_mass_list.append(hme_h2mass_reco)
    tree.Fill()

tree.Write()
out.Close()

print("Cputime ",stop_watch.CpuTime()," realtime ",stop_watch.RealTime())
stop_watch.Stop()


print("*****************************************************************************************************************************************")
print("if you want to use this code, please cite:                                                                                               ")
print("T. Huang, J.M. No, L. Pernie,  M. Ramsey-Musolf, A. Safonov, M. Spannowsky, and P. Winslow                                               ")
print("\" Resonant di-Higgs boson production in the bbWW channel: Probing the electroweak phase transition at the LHC\"                         ")
print("Phys. Rev. D 96, 035007, Published 11 August 2017                                                                                        ")
print("*****************************************************************************************************************************************")