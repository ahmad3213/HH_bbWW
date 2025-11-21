import numpy as np
import Analysis.hh_bbww as analysis
import FLAF.Common.Utilities as Utilities
import awkward as ak
import os
from DeepHME.src.DeepHME import DeepHME

higs_output_mapping = {
    "Hbb": {"px": 4, "py": 5, "pz": 6, "E": 7},
    "HVV": {"px": 0, "py": 1, "pz": 2, "E": 3},
}


class DeepHMEProducer:
    def __init__(self, cfg, payload_name):
        self.cfg = cfg
        self.payload_name = payload_name

        deepHME_folder = os.path.join(
            os.environ["ANALYSIS_PATH"], "config", "DeepHME", self.cfg["model_name"]
        )
        with open(os.path.join(deepHME_folder, "deepHME_config.txt"), "r") as file:
            self.vars_to_save = [line[:-1] for line in file.readlines()]

        self.estimator = DeepHME(
            model_name=cfg["model_name"],
            channel=cfg["channel"],
            return_errors=cfg["return_errors"],
        )

    def run(self, array, keep_all_columns=False):
        mass = None
        mass_errors = None

        pred = self.estimator.predict(
            event_id=array["event"],
            lep1_pt=array["lep1_pt"],
            lep1_eta=array["lep1_eta"],
            lep1_phi=array["lep1_phi"],
            lep1_mass=array["lep1_mass"],
            lep2_pt=array["lep2_pt"] if self.cfg["channel"] == "DL" else None,
            lep2_eta=array["lep2_eta"] if self.cfg["channel"] == "DL" else None,
            lep2_phi=array["lep2_phi"] if self.cfg["channel"] == "DL" else None,
            lep2_mass=array["lep2_mass"] if self.cfg["channel"] == "DL" else None,
            met_pt=array["PuppiMET_pt"],
            met_phi=array["PuppiMET_phi"],
            jet_pt=array["centralJet_pt"],
            jet_eta=array["centralJet_eta"],
            jet_phi=array["centralJet_phi"],
            jet_mass=array["centralJet_mass"],
            jet_btagPNetB=array["centralJet_btagPNetB"],
            jet_btagPNetCvB=array["centralJet_btagPNetCvB"],
            jet_btagPNetCvL=array["centralJet_btagPNetCvL"],
            jet_btagPNetCvNotB=array["centralJet_btagPNetCvNotB"],
            jet_btagPNetQvG=array["centralJet_btagPNetQvG"],
            jet_PNetRegPtRawCorr=array["centralJet_PNetRegPtRawCorr"],
            jet_PNetRegPtRawCorrNeutrino=array["centralJet_PNetRegPtRawCorrNeutrino"],
            jet_PNetRegPtRawRes=array["centralJet_PNetRegPtRawRes"],
            fatjet_pt=array["SelectedFatJet_pt"],
            fatjet_eta=array["SelectedFatJet_eta"],
            fatjet_phi=array["SelectedFatJet_phi"],
            fatjet_mass=array["SelectedFatJet_mass"],
            fatjet_particleNet_QCD=array["SelectedFatJet_particleNet_QCD"],
            fatjet_particleNet_XbbVsQCD=array["SelectedFatJet_particleNet_XbbVsQCD"],
            fatjet_particleNetWithMass_QCD=array[
                "SelectedFatJet_particleNetWithMass_QCD"
            ],
            fatjet_particleNetWithMass_HbbvsQCD=array[
                "SelectedFatJet_particleNetWithMass_HbbvsQCD"
            ],
            fatjet_particleNet_massCorr=array["SelectedFatJet_particleNet_massCorr"],
            output_format="mass",
        )

        if self.cfg["return_errors"]:
            mass, mass_errors = pred
        else:
            mass = pred

        # for now we decided to not return errors for components of p4
        pred = self.estimator.predict(
            event_id=array["event"],
            lep1_pt=array["lep1_pt"],
            lep1_eta=array["lep1_eta"],
            lep1_phi=array["lep1_phi"],
            lep1_mass=array["lep1_mass"],
            lep2_pt=array["lep2_pt"] if self.cfg["channel"] == "DL" else None,
            lep2_eta=array["lep2_eta"] if self.cfg["channel"] == "DL" else None,
            lep2_phi=array["lep2_phi"] if self.cfg["channel"] == "DL" else None,
            lep2_mass=array["lep2_mass"] if self.cfg["channel"] == "DL" else None,
            met_pt=array["PuppiMET_pt"],
            met_phi=array["PuppiMET_phi"],
            jet_pt=array["centralJet_pt"],
            jet_eta=array["centralJet_eta"],
            jet_phi=array["centralJet_phi"],
            jet_mass=array["centralJet_mass"],
            jet_btagPNetB=array["centralJet_btagPNetB"],
            jet_btagPNetCvB=array["centralJet_btagPNetCvB"],
            jet_btagPNetCvL=array["centralJet_btagPNetCvL"],
            jet_btagPNetCvNotB=array["centralJet_btagPNetCvNotB"],
            jet_btagPNetQvG=array["centralJet_btagPNetQvG"],
            jet_PNetRegPtRawCorr=array["centralJet_PNetRegPtRawCorr"],
            jet_PNetRegPtRawCorrNeutrino=array["centralJet_PNetRegPtRawCorrNeutrino"],
            jet_PNetRegPtRawRes=array["centralJet_PNetRegPtRawRes"],
            fatjet_pt=array["SelectedFatJet_pt"],
            fatjet_eta=array["SelectedFatJet_eta"],
            fatjet_phi=array["SelectedFatJet_phi"],
            fatjet_mass=array["SelectedFatJet_mass"],
            fatjet_particleNet_QCD=array["SelectedFatJet_particleNet_QCD"],
            fatjet_particleNet_XbbVsQCD=array["SelectedFatJet_particleNet_XbbVsQCD"],
            fatjet_particleNetWithMass_QCD=array[
                "SelectedFatJet_particleNetWithMass_QCD"
            ],
            fatjet_particleNetWithMass_HbbvsQCD=array[
                "SelectedFatJet_particleNetWithMass_HbbvsQCD"
            ],
            fatjet_particleNet_massCorr=array["SelectedFatJet_particleNet_massCorr"],
            output_format="p4",
        )

        p4 = None
        p4_errors = None
        if self.cfg["return_errors"]:
            p4, p4_errors = pred
        else:
            p4 = pred

        # Delete not-needed array
        for col in array.fields:
            if col not in self.cfg["columns"]:
                if col != "FullEventId":
                    if keep_all_columns:
                        continue
                    del array[col]

        array[f"{self.payload_name}_mass"] = mass
        if "mass_error" in self.cfg["columns"]:
            array[f"{self.payload_name}_mass_error"] = mass_errors

        # save energies and momenta of higgs bosons
        higgses = ["Hbb", "HVV"]
        for col in self.cfg["columns"]:
            var = col.split("_")[-1]
            for higgs in higgses:
                if higgs in col:
                    assert var in [
                        "px",
                        "py",
                        "pz",
                        "E",
                    ], "Attempting to save non-kinematic variable from predicted p4"
                    idx = higs_output_mapping[higgs][var]
                    array[f"{self.payload_name}_{higgs}_{var}"] = p4[:, idx]

        return array
