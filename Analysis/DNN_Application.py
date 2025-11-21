# python3 /afs/cern.ch/work/p/prsolank/private/FLAF_8thJan/AnaProd/NNInterface.py --inModelDir /afs/cern.ch/work/p/prsolank/private/FLAF_8thJan/config/HH_bbtautau/nn_models --inFile /tmp/prsolank/luigi-tmp-416131263.root --outFileName /tmp/prsolank/luigi-tmp-862152055.root --uncConfig /afs/cern.ch/work/p/prsolank/private/FLAF_8thJan/config/Run2_2018/weights.yaml --globalConfig /afs/cern.ch/work/p/prsolank/private/FLAF_8thJan/config/HH_bbtautau/global.yaml --EraName e2018 --Mass 400 --Spin 2 --PairType 2

from __future__ import annotations
import os, sys
import numpy as np
import awkward as ak
import onnxruntime as ort
import psutil
import yaml
import os
import ROOT
import FLAF.Common.Utilities as Utilities
import Analysis.hh_bbww as analysis


class DNNProducer:
    def __init__(self, cfg, payload_name):

        self.cfg = cfg
        self.payload_name = payload_name

        sys.path.append(os.environ["ANALYSIS_PATH"])
        ROOT.gROOT.ProcessLine(".include " + os.environ["ANALYSIS_PATH"])
        ROOT.gInterpreter.Declare(f'#include "FLAF/include/Utilities.h"')
        ROOT.gROOT.ProcessLine(f'#include "FLAF/include/HistHelper.h"')
        ROOT.gROOT.ProcessLine(f'#include "FLAF/include/AnalysisTools.h"')
        ROOT.gROOT.ProcessLine(f'#include "FLAF/include/AnalysisMath.h"')
        ROOT.gROOT.ProcessLine(f'#include "FLAF/include/MT2.h"')
        ROOT.gROOT.ProcessLine(f'#include "FLAF/include/Lester_mt2_bisect.cpp"')

        self.dnnConfig = {}
        dnnFolder = os.path.join(
            os.environ["ANALYSIS_PATH"], "config", "DNN", self.cfg["version"]
        )
        with open(os.path.join(dnnFolder, "dnn_config.yaml"), "r") as file:
            self.dnnConfig = yaml.safe_load(file)

        # Features to use for DNN application (single vals)
        features = self.dnnConfig["features"]
        # Features to use for DNN application (vectors and index)
        list_features = self.dnnConfig["listfeatures"]
        # Features to use for DNN application (high level names to create)
        highlevel_features = self.dnnConfig["highlevelfeatures"]
        # Features to use for DNN application (hme names to pull from cache)
        hme_features = (
            self.dnnConfig["hmefeatures"]
            if "hmefeatures" in self.dnnConfig.keys()
            else None
        )

        # Features to load from df to awkward
        load_features = set()
        if features != None:
            load_features.update(features)
        if list_features != None:
            for feature in list_features:
                load_features.update([feature[0]])
        if highlevel_features != None:
            load_features.update(highlevel_features)
        if hme_features != None:
            load_features.update(hme_features)

        load_features.update(["FullEventId"])
        load_features.update(["event"])

        # What to save in tmp file
        self.vars_to_save = load_features
        # What to save for final output
        self.cols_to_save = [
            f"{self.payload_name}_{col}" for col in self.cfg["columns"]
        ]

        modelname_parity = self.dnnConfig["modelname_parity"]

        self.models = [
            [ort.InferenceSession(f"{os.path.join(dnnFolder, x)}.onnx"), y]
            for x, y in modelname_parity
        ]

    def prepare_dfw(self, dfw):
        print("Running DNN preparer")

        dfw.df = analysis.defineAllP4(dfw.df)
        dfw.df = analysis.AddDNNVariables(dfw.df)

        return dfw

    def run(self, array):
        print("Running DNN producer")

        array = self.ApplyDNN(array)

        # Delete not-needed branches
        for col in array.fields:
            if col not in self.cfg["columns"]:
                if col != "FullEventId":
                    del array[col]

        # Rename the branches
        for col in self.cfg["columns"]:
            if col in array.fields:
                array[f"{self.payload_name}_{col}"] = array[f"{col}"]
                del array[f"{col}"]
            else:
                print(f"Expected column {col} not found in your payload array!")

        return array

    def ApplyDNN(self, branches):
        models = self.models
        dnnConfig = self.dnnConfig

        # Features to use for DNN application (single vals)
        features = dnnConfig["features"]
        # Features to use for DNN application (vectors and index)
        list_features = dnnConfig["listfeatures"]
        # Features to use for DNN application (high level names to create)
        highlevel_features = dnnConfig["highlevelfeatures"]
        # Features to use for DNN application (hme names to pull from cache)
        hme_features = (
            self.dnnConfig["hmefeatures"]
            if "hmefeatures" in self.dnnConfig.keys()
            else None
        )

        nClasses = dnnConfig["nClasses"] if "nClasses" in dnnConfig.keys() else 3
        nParity = dnnConfig["nParity"] if "nParity" in dnnConfig.keys() else 4

        use_parametric = dnnConfig["use_parametric"]
        # param_mass_list = [ 250, 260, 270, 280, 300, 350, 450, 550, 600, 650, 700, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000, 4000, 5000 ]
        param_mass_list = dnnConfig["parametric_list"]

        class_names_list = (
            dnnConfig["class_names"]
            if "class_names" in dnnConfig.keys()
            else ["Signal", "TT", "DY"]
        )

        if not use_parametric:
            param_mass_list = [0]

        nEvents = len(branches)
        print(f"Running DNN Over {nEvents} events")

        # Initialize the final predictions per parametric per event per class
        all_predictions = np.zeros((len(param_mass_list), nEvents, nClasses))

        event_number = branches.event

        array = np.array(
            [getattr(branches, feature_name) for feature_name in features]
        ).transpose()

        # Get vector value array
        default_value = 0.0
        if list_features != None:
            array_listfeatures = np.array(
                [
                    ak.fill_none(
                        ak.pad_none(getattr(branches, feature_name), index + 1),
                        default_value,
                    )[:, index]
                    for [feature_name, index] in list_features
                ]
            ).transpose()
            # Need to append the value features and the listfeatures together
            array = np.append(array, array_listfeatures, axis=1)

        # Need to append the high level features and the other features together
        if highlevel_features != None:
            array_highlevelfeatures = np.array(
                [getattr(branches, feature_name) for feature_name in highlevel_features]
            ).transpose()
            array = np.append(array, array_highlevelfeatures, axis=1)

        if hme_features != None:
            array_hmefeatures = np.array(
                [getattr(branches, feature_name) for feature_name in hme_features]
            ).transpose()
            array = np.append(array, array_hmefeatures, axis=1)

        # Initialize the local predictions including per parity, but this will be summed out later
        local_predictions = np.zeros(
            (len(param_mass_list), len(array), nParity, nClasses)
        )

        for parityIdx, [sess, parityfunc] in enumerate(models):
            # We want to only apply the 3 models that are NOT trained on this parity

            # Add parametric mass point to the array
            for param_idx, param_mass in enumerate(param_mass_list):
                param_array = np.array([[param_mass for x in array]]).transpose()
                if use_parametric:
                    final_array = np.append(array, param_array, axis=1)
                else:
                    final_array = array

                # prediction = model.predict(final_array)
                prediction = sess.run(
                    None, {"x": final_array}
                )  # Take only first entry, prediction is [ [Sig, TT, DY], [mBB_SR] ]

                class_prediction = prediction[0]
                # adv_prediction = prediction[1] # Sometimes we don't use an adv model

                # Now we need to set the trained parity to 0
                # But if there is only one model, then skip parity
                event_num = np.expand_dims(
                    event_number, axis=-1
                )  # We now get event_number from the FullEventId branch earlier
                parity_filter = np.repeat(event_num, nClasses, axis=-1)
                if nParity != 1:
                    class_prediction = np.where(
                        parity_filter % nParity != parityIdx, class_prediction, 0.0
                    )
                local_predictions[param_idx, :, parityIdx, :] = class_prediction

        # Reduce dimension by sum the parity axis from local to global
        all_predictions = np.sum(local_predictions, axis=2)

        if nParity != 1:
            all_predictions = all_predictions / (
                nParity - 1
            )  # So we want to divide by nParity-1 (4 parity -> train with 1, apply with remaining 3)

        # Last save the branches
        for param_idx, param_mass in enumerate(param_mass_list):
            this_param_prediction = all_predictions[
                param_idx, :, :
            ]  # Now we want to get the individual param masses predictions for filling

            for class_idx, class_name in enumerate(class_names_list):
                branches[f"M{param_mass}_{class_name}"] = (
                    this_param_prediction.transpose()[class_idx].astype(np.float32)
                )

        print("Finishing call, memory?")
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / 1024 / 1024
        print(f"Current memory usage: {mem_mb:.2f} MB")

        return branches
