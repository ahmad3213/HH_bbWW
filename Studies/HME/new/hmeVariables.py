import ROOT
ROOT.gROOT.ProcessLine('#include "include/EstimatorLTWrapper.hpp"')


def GetHMEVariables(df):
    df = df.Define("jets", """HME::VecLVF_t res;
                           for (size_t i = 0; i < nJet; ++i)
                           {{
                                res.emplace_back(centralJet_pt[i], centralJet_eta[i], centralJet_phi[i], centralJet_mass[i]);  
                            }}
                            return res;""")

    df = df.Define("leptons", """HME::VecLVF_t res;
                              res.emplace_back(lep1_pt, lep1_eta, lep1_phi, lep1_mass);
                              res.emplace_back(lep2_pt, lep2_eta, lep2_phi, lep2_mass);
                              return res;""")

    df = df.Define("met", """HME::LorentzVectorF_t res(PuppiMET_pt, 0.0, PuppiMET_phi, 0.0);    
                          return res;""")
		
    df = df.Define("hme_mass", """auto hme = HME::EstimatorLTWrapper::Instance().GetEstimator().EstimateMass(jets, leptons, met, event, HME::Channel::DL);
                                Float_t mass = -1.0f;
                                if (hme.has_value())
                                {{
                                    auto const& result_array = hme.value();
                                    mass = result_array[static_cast<size_t>(HME::EstimOut::mass)];
                                }}
                                return mass;""")                      
    return df