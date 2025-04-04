import tensorflow as tf
import pandas as pd
import numpy as np
import uproot 
import awkward as ak 
import os
import matplotlib.pyplot as plt
import sklearn.metrics
import yaml
import vector
import tqdm
import ROOT
import scipy
import tf2onnx
import onnx
import onnxruntime as ort
import shutil
import copy




def PlotMetric(history, model, metric, output_folder):
    plt.plot(history.history[metric], label=f'train_{metric}')
    plt.plot(history.history[f'val_{metric}'], label=f'val_{metric}')
    plt.title(f'{model} {metric}')
    plt.ylabel(metric)
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, f"{metric}_{model}.pdf"), bbox_inches='tight')
    plt.clf()


#Need to get train_features and train_labels
class DataWrapper():
    def __init__(self):
        print("Init data wrapper")
        self.value_to_label = {'1': 0, '2': 0, '8': 1, '5': 2}
        self.label_names = ["Signal", "TT", "DY"]

        self.value_to_label_binary = {'1': 0, '2': 0, '8': 1, '5': 1} # Binary for now

        self.binary = False

        self.feature_names = None
        self.listfeature_names = None
        self.highlevelfeatures_names = None
        self.label_name = None
        self.mbb_name = None

        self.features_no_param = None
        self.features = None
        self.listfeatures = None
        self.hlv = None
        self.param_values = None
        self.labels = None
        self.labels_binary = None
        self.mbb = None
        self.mbb_region = None
        self.mbb_region_binary = None
        self.mbb_region_random = None

        self.class_weight = None
        self.adv_weight = None

        self.train_features = None
        self.train_labels = None
        self.train_labels_binary = None
        self.train_mbb = None
        self.train_class_weight = None
        self.train_adv_weight = None

        self.test_features = None
        self.test_labels = None
        self.test_labels_binary = None
        self.test_mbb = None
        self.test_class_weight = None
        self.test_adv_weight = None

        self.param_list = [250, 260, 270, 280, 300, 350, 450, 550, 600, 650, 700, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000, 4000, 5000 ]
        self.use_parametric = False

        self.features_paramSet = None

        self.output_folder = ""

    def SetBinary(self, use_binary):
        self.binary = use_binary

    def SetOutputFolder(self, foldername):
        self.output_folder = foldername
        os.makedirs(foldername, exist_ok = True)

    def UseParametric(self, use_parametric):
        self.use_parametric = use_parametric
        print(f"Parametric feature set to {use_parametric}")

    def SetParamList(self, param_list):
        self.param_list = param_list

    def SetPredictParamValue(self, param_value):
        #During predict, we want to use a truly random param value even for signal!
        if param_value not in self.param_list:
            print(f"This param value {param_value} is not an option!")
        new_params = np.array([[param_value for x in self.features]]).transpose()

        self.features_paramSet = np.append(self.features_no_param, new_params, axis=1)

    def AddInputFeatures(self, features):
        if self.feature_names == None:
            self.feature_names = features
        else:
            self.feature_names = self.feature_names + features

        print(f"Added features {features}")
        print(f"New feature list {self.feature_names}")

    def AddInputFeaturesList(self, features, index):
        if self.listfeature_names == None:
            self.listfeature_names = [[feature, index] for feature in features]
        else:
            self.listfeature_names = self.listfeature_names + [[feature, index] for feature in features]

        print(f"Added features {features} with index {index}")
        print(f"New feature list {self.listfeature_names}")

    def AddHighLevelFeatures(self, features):
        if self.highlevelfeatures_names == None:
            self.highlevelfeatures_names = features
        else:
            self.highlevelfeatures_names = self.highlevelfeatures_names + features

        print(f"Added high level features {features}")
        print(f"New feature list {self.highlevelfeatures_names}")


    def AddInputLabel(self, labels_name):
        if self.label_name != None:
            print("What are you doing? You already defined the input label branch")
        self.label_name = labels_name

    def SetMbbName(self, mbb_name):
        if self.mbb_name != None:
            print("What are you doing? You already defined the mbb branch")
        self.mbb_name = mbb_name
        
    def ReadFile(self, file_name, entry_start=None, entry_stop=None):
        if self.feature_names == None:
            print("Uknown branches to read! DefineInputFeatures first!")
            return
        if self.label_name == None:
            print("No label branch defined!")
            return

        print(f"Reading file {file_name}")

        features_to_load = []
        features_to_load = features_to_load + self.feature_names
        for listfeature in self.listfeature_names:
           if listfeature[0] not in features_to_load: features_to_load.append(listfeature[0])
        features_to_load = features_to_load + self.highlevelfeatures_names

        features_to_load.append(self.mbb_name)
        features_to_load.append(self.label_name)
        features_to_load.append('X_mass')

        print(f"Only loading these features {features_to_load}")

        file = uproot.open(file_name)
        tree = file['Events']
        branches = tree.arrays(features_to_load, entry_start=entry_start, entry_stop=entry_stop)

        self.features = np.array([getattr(branches, feature_name) for feature_name in self.feature_names]).transpose()
        print("Got features, but its a np array")

        default_value = 0.0
        if self.listfeature_names != None: 
            self.listfeatures = np.array([ak.fill_none(ak.pad_none(getattr(branches, feature_name), index+1), default_value)[:,index] for [feature_name,index] in self.listfeature_names]).transpose()
        print("Got the list features")

        #Need to append the value features and the listfeatures together
        if self.listfeature_names != None: 
            print("We have list features!")
            self.features = np.append(self.features, self.listfeatures, axis=1)

        if self.highlevelfeatures_names != None: 
            self.hlv = np.array([getattr(branches, feature_name) for feature_name in self.highlevelfeatures_names]).transpose()
            self.features = np.append(self.features, self.hlv, axis=1)


        labels_branch_values = np.array(getattr(branches, self.label_name))

        self.labels = np.array([self.value_to_label[str(branch_val)] for branch_val in labels_branch_values])
        self.labels_binary = np.array([self.value_to_label_binary[str(branch_val)] for branch_val in labels_branch_values])
        print("Got labels")


        self.mbb, self.mbb_region, self.mbb_region_binary, self.mbb_region_random = self.SetMbbRegion(branches)
        print("Set mbb regions")

        #Add parametric variable
        self.param_values = np.array([[x if (x > 0) else np.random.choice(self.param_list) for x in getattr(branches, 'X_mass') ]]).transpose()
        print("Got the param values")


        self.features_no_param = self.features
        if self.use_parametric: self.features = np.append(self.features, self.param_values, axis=1)


    def ReadWeightFile(self, weight_name, entry_start=None, entry_stop=None):
        print(f"Reading weight file {weight_name}")
        file = uproot.open(weight_name)
        tree = file['weight_tree']
        branches = tree.arrays(entry_start=entry_start, entry_stop=entry_stop)
        self.class_weight = np.array(getattr(branches, 'class_weight'))
        self.adv_weight = np.array(getattr(branches, 'adv_weight'))

    def SetMbbRegion(self, branches):
        print("Inside setting mbb region!")
        # But we want to blind the adversarial part to Signal events, meaning we must filter them out
        mbb = np.array(getattr(branches, self.mbb_name))
        # Start region of CR_low, SR, CR_high
        mbb_region = np.where(
          mbb < 70,
          -1,
          np.where(
            mbb < 150,
            0,
            1
          )
        )

        mbb_region_binary = np.where(
          (mbb_region == 0),
          1,
          0
        )

        mbb_region_random = np.random.choice(2, len(mbb))

        return mbb, mbb_region, mbb_region_binary, mbb_region_random



    def SetClassWeights(self):
        print("Inside the SetCateogryWeights!")
        nTotal = len(self.labels)
        weight_dict = {}

        if self.binary:
            unique_labels = np.unique(self.labels_binary)
            for i, unique_label in enumerate(unique_labels):
                weight_dict[unique_label] = tf.cast(nTotal/np.sum(self.labels_binary == unique_label), tf.float32)

        else:
            unique_labels = np.unique(self.labels)
            weight_dict = {}
            for i, unique_label in enumerate(unique_labels):
                weight_dict[unique_label] = tf.cast(nTotal/np.sum(self.labels == unique_label), tf.float32)

        return weight_dict
    


    def DefineTrainTestSet(self, batch_size, ratio):
        print("Create the self.train_features and self.train_labels lists here")

        nBatches = len(self.labels)/batch_size
        nBatchesTest = int(nBatches*ratio)
        nBatchesTrain = nBatches-nBatchesTest


        trainStart = 0
        trainEnd = int(nBatchesTrain*batch_size)

        testStart = trainEnd
        testEnd = testStart + int(nBatchesTest*batch_size)

        print("Got the start/end, it is")
        print(f"{trainStart} {trainEnd} {testStart} {testEnd}")

        self.train_features = self.features[trainStart:trainEnd]
        self.train_labels = self.labels[trainStart:trainEnd]
        self.train_labels_binary = self.labels_binary[trainStart:trainEnd]
        self.train_mbb = self.mbb_region_binary[trainStart:trainEnd]
        self.train_mbb_random = self.mbb_region_random[trainStart:trainEnd]
        self.train_class_weight = self.class_weight[trainStart:trainEnd]
        self.train_adv_weight = self.adv_weight[trainStart:trainEnd]
        
        self.test_features = self.features[testStart:testEnd]
        self.test_labels = self.labels[testStart:testEnd]
        self.test_labels_binary = self.labels_binary[testStart:testEnd]
        self.test_mbb = self.mbb_region_binary[testStart:testEnd]
        self.test_mbb_random = self.mbb_region_random[testStart:testEnd]
        self.test_class_weight = self.class_weight[testStart:testEnd]
        self.test_adv_weight = self.adv_weight[testStart:testEnd]

    def monitor_param(self, model, param_value):
        self.SetPredictParamValue(param_value)
        arglist = None
        if param_value <= 0: arglist = self.features
        else: arglist = self.features_paramSet

        this_features = arglist
        this_labels = self.labels

        if param_value > 0:
            #For signal, param_values is the true mass -- filter signal to only show confusion of correct mass
            signal_mask = (self.labels != 1) | (self.param_values == param_value).transpose()[0]
            this_features = this_features[signal_mask]
            this_labels = this_labels[signal_mask]

        pred = model.predict(arglist)[0]
        print(f"Prediction is {pred}")

        cm = sklearn.metrics.confusion_matrix(self.labels, np.argmax(pred, axis=1), normalize='true')
        disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.label_names)
        disp.plot(cmap=plt.cm.Blues)
        plt.savefig(os.path.join(self.output_folder, f"conf_matrix_M{param_value}.pdf"))
        plt.close()

        #Lets put the ROC curve for each class on a plot and calculate AUC
        g, c_ax = plt.subplots(1,1, figsize = (8,6))
        for (idx, c_label) in enumerate(self.label_names):
            fpr, tpr, thresholds = sklearn.metrics.roc_curve(tf.keras.utils.to_categorical(self.labels,3)[:,idx].astype(int), pred[:,idx])
            c_ax.plot(fpr, tpr, label = "{label} (AUC: {auc})".format(label = c_label, auc = sklearn.metrics.auc(fpr, tpr)))

        c_ax.plot(fpr, fpr, "b-", label = "Random Guessing")

        c_ax.legend()
        c_ax.grid(True)
        c_ax.set_xlabel("False Positive Rate")
        c_ax.set_ylabel("True Positive Rate")
        g.savefig(os.path.join(self.output_folder, f"roc_curves_M{param_value}.pdf"))
        g.clf()
        plt.close()
        #g.close()

        return pred






    def validate_output(self, sess, model_name, parity_index):
        plotbins = 300
        plotrange = (0.0, 3.0)
        non_quant_binning = np.linspace(0.0, 1.0, 101)

        # Plot of mbb value vs DNN score
        mbb_bins = 20
        mbb_min = 0.0
        mbb_max = 200.0
        mbb_step = mbb_max / mbb_bins

        #After discussion with Konstanin, parametric dnn application should put in mass values for the point we want
        #So we will have a DNN output for masspoint 300, 400, 500, etc
        #For now, lets create a parametric_masspoints list and apply with those masses
        #Theory is that these masses will work very well for signal at that mass, but still remove backgrounds

        save_path = os.path.join(self.output_folder, model_name.split('.')[0])
        os.makedirs(save_path, exist_ok=True)

        for para_masspoint in self.param_list:
            continue
            print(f"Looking at mass {para_masspoint}")
            if para_masspoint > 1000: continue
            if para_masspoint not in [300, 450, 550, 800]: continue



            nBinsForChi2 = 10 # We must set up our quant binning object first so it exists across the loops
            quant_binning = None

            # And lets set up our list of ROOT TH1D for Signal / Bkg1 / Bkg2 / ...
            # This is so we can draw all backgrounds on the same plot at the end using quant binning
            all_SR_plots = []
            all_SR_legend = ROOT.TLegend(0.5, 0.8, 0.9, 0.9)

            for i, label_name in enumerate(self.label_names):
                canvas = ROOT.TCanvas("c1", "c1", 800, 600)
                print(f"On label {label_name}")

                labels = self.labels
                mbb_region = (self.mbb_region).transpose()[0]


                self.SetPredictParamValue(para_masspoint)

                this_sample_mask = labels == i

                # Check that there are any events with this label
                if len(self.features_paramSet[this_sample_mask]) == 0: continue

                features_this_sample = self.features_paramSet[this_sample_mask]
                features_this_sample_no_parametric = self.features_no_param[this_sample_mask]
                mbb_region_this_sample = self.mbb_region[this_sample_mask]
                mbb_this_sample = self.mbb[this_sample_mask]

                features_to_use = features_this_sample if self.use_parametric else features_this_sample_no_parametric

                pred = sess.run(None, {'x': features_to_use})
                pred = pred[0][:,0] #Only get first entry (category prediction), all events, then first category (signal)
                print("pred")
                print(pred)

                mbbSR = mbb_region_this_sample == 0
                mbbCR_low = mbb_region_this_sample == -1
                mbbCR_high = mbb_region_this_sample == 1

                pred_mbb_SR = pred[mbbSR]
                pred_mbb_CR_low = pred[mbbCR_low]
                pred_mbb_CR_high = pred[mbbCR_high]



                if i == 0:
                    #This is signal, lets plot the masses separate
                    for real_mass in self.param_list:
                        if abs(real_mass - para_masspoint) > 0: 
                            continue
                        plotlabel = f'{label_name} M{real_mass}'

                        param_values_this_sample = self.param_values[this_sample_mask]


                        param_values_mbb_SR = param_values_this_sample[mbbSR]
                        this_mass_mask_SR = (param_values_mbb_SR == real_mass).flatten()

                        pred_mbb_SR_thismass = pred_mbb_SR[this_mass_mask_SR]

                        param_values_mbb_CR_low = param_values_this_sample[mbbCR_low]
                        this_mass_mask_CR_low = (param_values_mbb_CR_low == real_mass).flatten()

                        pred_mbb_CR_low_thismass = pred_mbb_CR_low[this_mass_mask_CR_low]

                        param_values_mbb_CR_high = param_values_this_sample[mbbCR_high]
                        this_mass_mask_CR_high = (param_values_mbb_CR_high == real_mass).flatten()

                        pred_mbb_CR_high_thismass = pred_mbb_CR_high[this_mass_mask_CR_high]


                        ks_value_low = scipy.stats.kstest(np.sort(pred_mbb_SR_thismass), np.sort(pred_mbb_CR_low_thismass)).pvalue
                        ks_value_high = scipy.stats.kstest(np.sort(pred_mbb_SR_thismass), np.sort(pred_mbb_CR_high_thismass)).pvalue

                        # Set up signal quant binning in the SignalRegion of m_bb
                        quant_binning = np.zeros(nBinsForChi2+1) # Need +1 because 10 bins actually have 11 edges
                        quant_binning[1:nBinsForChi2] = np.quantile(pred_mbb_SR_thismass, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
                        quant_binning[-1] = 1.0 
                        print("We found quant binning")
                        print(quant_binning)

                        val_mbb_SR, bins_mbb_SR = np.histogram(pred_mbb_SR_thismass, bins=quant_binning, range=(0.0, 1.0))
                        val_mbb_CR_low, bins_mbb_CR_low = np.histogram(pred_mbb_CR_low_thismass, bins=quant_binning, range=(0.0, 1.0))
                        val_mbb_CR_high, bins_mbb_CR_high = np.histogram(pred_mbb_CR_high_thismass, bins=quant_binning, range=(0.0, 1.0))

                        ROOT_hist_mbb_SR = ROOT.TH1D("hist_mbb_SR", "hist_mbb_SR", nBinsForChi2, 0.0, 1.0)
                        ROOT_hist_mbb_CR_low = ROOT.TH1D("hist_mbb_CR_low", "hist_mbb_CR_low", nBinsForChi2, 0.0, 1.0)
                        ROOT_hist_mbb_CR_high = ROOT.TH1D("hist_mbb_CR_high", "hist_mbb_CR_high", nBinsForChi2, 0.0, 1.0)


                        for binnum in range(nBinsForChi2):
                            ROOT_hist_mbb_SR.SetBinContent(binnum+1, val_mbb_SR[binnum])
                            ROOT_hist_mbb_SR.SetBinError(binnum+1, val_mbb_SR[binnum]**(0.5))
                            

                            ROOT_hist_mbb_CR_low.SetBinContent(binnum+1, val_mbb_CR_low[binnum])
                            ROOT_hist_mbb_CR_low.SetBinError(binnum+1, val_mbb_CR_low[binnum]**(0.5))


                            ROOT_hist_mbb_CR_high.SetBinContent(binnum+1, val_mbb_CR_high[binnum])
                            ROOT_hist_mbb_CR_high.SetBinError(binnum+1, val_mbb_CR_high[binnum]**(0.5))




                        ROOT_hist2d = ROOT.TH2D("hist_2d", "hist_2d", mbb_bins, mbb_min, mbb_max, nBinsForChi2, 0.0, 1.0)
                        pred_for_2d = pred[(param_values_this_sample == real_mass).flatten()]
                        mbb_for_2d = mbb_this_sample[(param_values_this_sample == real_mass).flatten()]

                        for dnn_bin in range(nBinsForChi2):
                            dnn_low = quant_binning[dnn_bin]
                            dnn_high = quant_binning[dnn_bin+1]
                            val, bins = np.histogram(mbb_for_2d[(pred_for_2d > dnn_low) & (pred_for_2d < dnn_high)], bins=mbb_bins, range=(mbb_min, mbb_max))
                            for mbb_binnum in range(mbb_bins):
                                val_to_fill = val[mbb_binnum]
                                val_to_fill = val_to_fill / np.sum(val)
                                ROOT_hist2d.SetBinContent(mbb_binnum+1, dnn_bin+1, val_to_fill)
                                ROOT_hist2d.SetBinError(mbb_binnum+1, dnn_bin+1, val_to_fill**(0.5))
                        ROOT_hist2d.SetStats(0)
                        ROOT_hist2d.Draw("colz")
                        ROOT_hist2d.SetTitle(f"2D {label_name} M{para_masspoint}")
                        ROOT_hist2d.GetXaxis().SetTitle("m_bb reco")
                        ROOT_hist2d.GetYaxis().SetTitle("DNN Bin (1 == Signal)")
                        canvas.SaveAs(os.path.join(self.output_folder, model_name, f'2D_parity{parity_index}_{label_name}_M{para_masspoint}.pdf'))


                        chi2_value_low = ROOT_hist_mbb_SR.Chi2Test(ROOT_hist_mbb_CR_low, option='WW')
                        chi2_value_high = ROOT_hist_mbb_SR.Chi2Test(ROOT_hist_mbb_CR_high, option='WW')

                        ROOT_hist_mbb_SR.Scale(1.0/ROOT_hist_mbb_SR.Integral())
                        ROOT_hist_mbb_CR_low.Scale(1.0/ROOT_hist_mbb_CR_low.Integral())
                        ROOT_hist_mbb_CR_high.Scale(1.0/ROOT_hist_mbb_CR_high.Integral())


                        plotlabel = f"{plotlabel} Chi2_low={chi2_value_low} Chi2_high={chi2_value_high}"

                        plt.hist(pred_mbb_SR_thismass+1.0, bins=non_quant_binning+1.0, range=plotrange, density=True, histtype='step', label=plotlabel, alpha=0.5)
                        plt.hist(pred_mbb_CR_low_thismass, bins=non_quant_binning, range=plotrange, density=True, histtype='step', label=plotlabel, alpha=0.5)
                        plt.hist(pred_mbb_CR_high_thismass+2.0, bins=non_quant_binning+2.0, range=plotrange, density=True, histtype='step', label=plotlabel, alpha=0.5)

                        ROOT_hist_mbb_SR.Draw()
                        ROOT_hist_mbb_SR.SetTitle(plotlabel)
                        ROOT_hist_mbb_SR.SetStats(0)
                        min_val = max(0.0001, min(ROOT_hist_mbb_SR.GetMinimum(), ROOT_hist_mbb_CR_low.GetMinimum(), ROOT_hist_mbb_CR_high.GetMinimum()))
                        max_val = max(ROOT_hist_mbb_SR.GetMaximum(), ROOT_hist_mbb_CR_low.GetMaximum(), ROOT_hist_mbb_CR_high.GetMaximum())
                        ROOT_hist_mbb_SR.GetYaxis().SetRangeUser(0.01*min_val, 100*max_val)
                        ROOT_hist_mbb_CR_low.SetLineColor(ROOT.kRed)
                        ROOT_hist_mbb_CR_low.Draw("same")
                        ROOT_hist_mbb_CR_high.SetLineColor(ROOT.kGreen+2)
                        ROOT_hist_mbb_CR_high.Draw("same")

                        legend = ROOT.TLegend(0.5, 0.8, 0.9, 0.9)
                        legend.AddEntry(ROOT_hist_mbb_SR, "m_bb SR")
                        legend.AddEntry(ROOT_hist_mbb_CR_low, "m_bb CR Low")
                        legend.AddEntry(ROOT_hist_mbb_CR_high, "m_bb CR High")
                        legend.Draw()

                        print(f"Setting canvas to log scale with range {min_val}, {max_val}")
                        canvas.SetLogy()
                        canvas.SaveAs(os.path.join(self.output_folder, model_name, f'blindcheck{parity_index}_{label_name}_M{para_masspoint}.pdf'))


                        # Add plot to the all_SR_plots list
                        all_SR_plots.append(ROOT_hist_mbb_SR)

                else:
                    plotlabel = label_name

                    ks_value_low = scipy.stats.kstest(np.sort(pred_mbb_SR), np.sort(pred_mbb_CR_low)).pvalue
                    ks_value_high = scipy.stats.kstest(np.sort(pred_mbb_SR), np.sort(pred_mbb_CR_high)).pvalue

                    nBinsForChi2 = 10
                    val_mbb_SR, bins_mbb_SR = np.histogram(pred_mbb_SR, bins=quant_binning, range=(0.0, 1.0))
                    val_mbb_CR_low, bins_mbb_CR_low = np.histogram(pred_mbb_CR_low, bins=quant_binning, range=(0.0, 1.0))
                    val_mbb_CR_high, bins_mbb_CR_high = np.histogram(pred_mbb_CR_high, bins=quant_binning, range=(0.0, 1.0))

                    ROOT_hist_mbb_SR = ROOT.TH1D("hist_mbb_SR", "hist_mbb_SR", nBinsForChi2, 0.0, 1.0)
                    ROOT_hist_mbb_CR_low = ROOT.TH1D("hist_mbb_CR_low", "hist_mbb_CR_low", nBinsForChi2, 0.0, 1.0)
                    ROOT_hist_mbb_CR_high = ROOT.TH1D("hist_mbb_CR_high", "hist_mbb_CR_high", nBinsForChi2, 0.0, 1.0)


                    for binnum in range(nBinsForChi2):
                        ROOT_hist_mbb_SR.SetBinContent(binnum+1, val_mbb_SR[binnum])
                        ROOT_hist_mbb_SR.SetBinError(binnum+1, val_mbb_SR[binnum]**(0.5))
                        

                        ROOT_hist_mbb_CR_low.SetBinContent(binnum+1, val_mbb_CR_low[binnum])
                        ROOT_hist_mbb_CR_low.SetBinError(binnum+1, val_mbb_CR_low[binnum]**(0.5))


                        ROOT_hist_mbb_CR_high.SetBinContent(binnum+1, val_mbb_CR_high[binnum])
                        ROOT_hist_mbb_CR_high.SetBinError(binnum+1, val_mbb_CR_high[binnum]**(0.5))


                    ROOT_hist2d = ROOT.TH2D("hist_2d", "hist_2d", mbb_bins, mbb_min, mbb_max, nBinsForChi2, 0.0, 1.0)
                    pred_for_2d = pred
                    mbb_for_2d = mbb_this_sample

                    print(f"Inside label {label_name} for 2D hist")
                    for dnn_bin in range(nBinsForChi2):
                        print(f"At dnn bin {dnn_bin}")
                        dnn_low = quant_binning[dnn_bin]
                        dnn_high = quant_binning[dnn_bin+1]
                        val, bins = np.histogram(mbb_for_2d[(pred_for_2d > dnn_low) & (pred_for_2d < dnn_high)], bins=mbb_bins, range=(mbb_min, mbb_max))
                        print(f"Going to fill with vals {val}")
                        for mbb_binnum in range(mbb_bins):
                            val_to_fill = val[mbb_binnum]
                            val_to_fill = val_to_fill / np.sum(val)
                            ROOT_hist2d.SetBinContent(mbb_binnum+1, dnn_bin+1, val_to_fill)
                            ROOT_hist2d.SetBinError(mbb_binnum+1, dnn_bin+1, val_to_fill**(0.5))
                    ROOT_hist2d.SetStats(0)
                    ROOT_hist2d.Draw("colz")
                    ROOT_hist2d.SetTitle(f"2D {label_name} M{para_masspoint}")
                    ROOT_hist2d.GetXaxis().SetTitle("m_bb reco")
                    ROOT_hist2d.GetYaxis().SetTitle("DNN Bin (1 == Signal)")
                    canvas.SaveAs(os.path.join(self.output_folder, model_name, f'2D_parity{parity_index}_{label_name}_M{para_masspoint}.pdf'))



                    chi2_value_low = ROOT_hist_mbb_SR.Chi2Test(ROOT_hist_mbb_CR_low, option='WW')
                    chi2_value_high = ROOT_hist_mbb_SR.Chi2Test(ROOT_hist_mbb_CR_high, option='WW')

           
                    ROOT_hist_mbb_SR.Scale(1.0/ROOT_hist_mbb_SR.Integral())
                    ROOT_hist_mbb_CR_low.Scale(1.0/ROOT_hist_mbb_CR_low.Integral())
                    ROOT_hist_mbb_CR_high.Scale(1.0/ROOT_hist_mbb_CR_high.Integral())

                    plotlabel = f"{plotlabel} Chi2_low={chi2_value_low} Chi2_high={chi2_value_high}"

                    plt.hist(pred_mbb_SR+1.0, bins=non_quant_binning+1.0, range=plotrange, density=True, histtype='step', label=plotlabel, alpha=0.5)
                    plt.hist(pred_mbb_CR_low, bins=non_quant_binning, range=plotrange, density=True, histtype='step', label=plotlabel, alpha=0.5)
                    plt.hist(pred_mbb_CR_high+2.0, bins=non_quant_binning+2.0, range=plotrange, density=True, histtype='step', label=plotlabel, alpha=0.5)

                    # ROOT_hist_mbb_SR.Draw()
                    ROOT_hist_mbb_SR.SetTitle(plotlabel)
                    ROOT_hist_mbb_SR.SetStats(0)
                    min_val = max(0.0001, min(ROOT_hist_mbb_SR.GetMinimum(), ROOT_hist_mbb_CR_low.GetMinimum(), ROOT_hist_mbb_CR_high.GetMinimum()))
                    max_val = max(ROOT_hist_mbb_SR.GetMaximum(), ROOT_hist_mbb_CR_low.GetMaximum(), ROOT_hist_mbb_CR_high.GetMaximum())
                    ROOT_hist_mbb_SR.GetYaxis().SetRangeUser(0.01*min_val, 100*max_val)
                    ROOT_hist_mbb_CR_low.SetLineColor(ROOT.kRed)
                    # ROOT_hist_mbb_CR_low.Draw("same")
                    ROOT_hist_mbb_CR_high.SetLineColor(ROOT.kGreen+2)
                    # ROOT_hist_mbb_CR_high.Draw("same")

                    ratioplot = ROOT.TRatioPlot(ROOT_hist_mbb_CR_high, ROOT_hist_mbb_SR)
                    ratioplot.Draw()

                    legend = ROOT.TLegend(0.5, 0.8, 0.9, 0.9)
                    legend.AddEntry(ROOT_hist_mbb_SR, "m_bb SR")
                    # legend.AddEntry(ROOT_hist_mbb_CR_low, "m_bb CR Low")
                    legend.AddEntry(ROOT_hist_mbb_CR_high, "m_bb CR High")
                    legend.Draw()



                    print(f"Setting canvas to log scale with range {min_val}, {max_val}")
                    canvas.SetLogy()
                    canvas.SaveAs(os.path.join(self.output_folder, model_name, f'blindcheck{parity_index}_{label_name}_M{para_masspoint}.pdf'))




                    # Add plot to the all_SR_plots list
                    all_SR_plots.append(ROOT_hist_mbb_SR)


                all_SR_legend.AddEntry(all_SR_plots[-1], label_name)



            plt.title(f'DNN Output: PredictSignal M{para_masspoint}')
            plt.legend(loc='upper right', fontsize="4")
            plt.yscale('log')
            plt.savefig(os.path.join(self.output_folder, model_name, f'check{parity_index}_dnn_values_M{para_masspoint}.pdf'))
            plt.clf()
            plt.close()


            color_list = [ROOT.kRed, ROOT.kBlue, ROOT.kGreen+2]
            for i, hist in enumerate(all_SR_plots):
                if i == 0:
                    hist.SetTitle("All Samples m_bb SignalRegion")
                    hist.Draw()
                else:
                    hist.Draw("same")

                hist.SetLineColor(color_list[i])

            all_SR_legend.Draw()
            canvas.SaveAs(os.path.join(self.output_folder, model_name, f'all_samples_M{para_masspoint}.pdf'))











        # We want to make some validation of the Adv output
        os.makedirs(save_path, exist_ok=True)

        para_masspoint = 300
        if self.use_parametric: self.SetPredictParamValue(para_masspoint)
        features = self.features_paramSet if self.use_parametric else self.features_no_param

        pred = sess.run(None, {'x': features})
        pred_class = pred[0]
        pred_signal = pred_class[:,0]
        pred_adv = pred[1][:,0]



        adv_weight = self.adv_weight
        class_weight = self.class_weight


        adv_loss_vec = binary_focal_crossentropy(tf.cast(self.mbb_region, dtype=tf.float32), tf.cast(pred[1], dtype=tf.float32), tf.cast(tf.one_hot(self.labels, 2), dtype=tf.float32), tf.cast(pred_class, dtype=tf.float32))[:,0]
        adv_loss = round(np.average(adv_loss_vec, weights=adv_weight),3)
        adv_accuracy_vec = accuracy(tf.cast(self.mbb_region, dtype=tf.float32), tf.cast(pred[1], dtype=tf.float32))[:,0]
        adv_accuracy = round(np.average(adv_accuracy_vec, weights=adv_weight),3)

        class_loss_vec = categorical_crossentropy(tf.cast(tf.one_hot(self.labels, 2), dtype=tf.float32), tf.cast(pred_class, dtype=tf.float32))
        print("Class loss vec")
        print(class_loss_vec)
        class_loss = round(np.average(class_loss_vec, weights=class_weight),3)
        print("Class loss")
        print(class_loss)
        class_accuracy_vec = categorical_accuracy(tf.cast(tf.one_hot(self.labels, 2), dtype=tf.float32), tf.cast(pred_class, dtype=tf.float32))
        class_accuracy = round(np.average(class_accuracy_vec, weights=class_weight),3)

        print("Having a problem with the data types?")
        print(f"Random region {tf.expand_dims(tf.cast(self.mbb_region_random, dtype=tf.float32), axis=-1)}")
        print(f"Prediction {tf.cast(pred[1], dtype=tf.float32)}")
        print(f"True region {tf.cast(self.mbb_region, dtype=tf.float32)}")

        # adv_loss_random_vec = binary_focal_crossentropy(tf.cast(self.mbb_region_random, dtype=tf.float32), tf.cast(pred[1], dtype=tf.float32), tf.cast(tf.one_hot(self.labels, 2), dtype=tf.float32), tf.cast(pred_class, dtype=tf.float32))[:,0]
        adv_loss_random_vec = binary_focal_crossentropy(tf.cast(self.mbb_region, dtype=tf.float32), tf.expand_dims(tf.cast(self.mbb_region_random, dtype=tf.float32), axis=-1), tf.cast(tf.one_hot(self.labels, 2), dtype=tf.float32), tf.cast(pred_class, dtype=tf.float32))[:,0]
        adv_loss_random = round(np.average(adv_loss_random_vec, weights=adv_weight),5)

        mbb_region = self.mbb_region




        # Class Plots
        # Lets build Masks
        Sig_SR_mask = (self.labels == 0) & (self.mbb_region == 0)
        Sig_CR_high_mask = (self.labels == 0) & (self.mbb_region == 1)

        TT_SR_mask = (self.labels == 1) & (self.mbb_region == 0)
        TT_CR_high_mask = (self.labels == 1) & (self.mbb_region == 1)

        DY_SR_mask = (self.labels == 2) & (self.mbb_region == 0)
        DY_CR_high_mask = (self.labels == 2) & (self.mbb_region == 1)

        # Set class quantiles based on signal
        nQuantBins = 10
        quant_binning_class = np.zeros(nQuantBins+1) # Need +1 because 10 bins actually have 11 edges
        quant_binning_class[1:nQuantBins] = np.quantile(pred_signal[Sig_SR_mask], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]) # Change list to something dynamic with nQuantBins
        quant_binning_class[-1] = 1.0 
        print("We found quant binning class")
        print(quant_binning_class)


        quant_binning_adv = np.zeros(nQuantBins+1) # Need +1 because 10 bins actually have 11 edges
        quant_binning_adv[1:nQuantBins] = np.quantile(pred_adv[TT_SR_mask], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]) # Change list to something dynamic with nQuantBins
        quant_binning_adv[-1] = 1.0 
        print("We found quant binning adv")
        print(quant_binning_adv)


        mask_dict = {
          'Signal': {
            'SR': Sig_SR_mask,
            'CR_high': Sig_CR_high_mask,
          },
          'TT': {
            'SR': TT_SR_mask,
            'CR_high': TT_CR_high_mask,
          },
          # 'DY': { # DY weight is turned off
          #   'SR': DY_SR_mask,
          #   'CR_high': DY_CR_high_mask,
          # },
        }

        # Lets look at Sig


        # Lets look at TT

        for process_name in mask_dict.keys():
          SR_mask = mask_dict[process_name]['SR']
          CR_high_mask = mask_dict[process_name]['CR_high']






          class_out_hist_SR, bins = np.histogram(pred_signal[SR_mask], bins=quant_binning_class, range=(0.0, 1.0), weights=class_weight[SR_mask])
          class_out_hist_CR_high, bins = np.histogram(pred_signal[CR_high_mask], bins=quant_binning_class, range=(0.0, 1.0), weights=class_weight[CR_high_mask])

          # Don't use weights for class
          # class_out_hist_SR, bins = np.histogram(pred_signal[SR_mask], bins=quant_binning_class, range=(0.0, 1.0))
          # class_out_hist_CR_high, bins = np.histogram(pred_signal[CR_high_mask], bins=quant_binning_class, range=(0.0, 1.0))

          ROOT_ClassOutput_SR = ROOT.TH1D(f"ClassOutput_{process_name}_SR", f"ClassOutput_{process_name}_SR", nQuantBins, 0.0, 1.0)
          ROOT_ClassOutput_CR_high = ROOT.TH1D(f"ClassOutput_{process_name}_CR_high", f"ClassOutput_{process_name}_CR_high", nQuantBins, 0.0, 1.0)


          for binnum in range(nQuantBins):
              ROOT_ClassOutput_SR.SetBinContent(binnum+1, class_out_hist_SR[binnum])
              ROOT_ClassOutput_SR.SetBinError(binnum+1, class_out_hist_SR[binnum]**(0.5))
              
              ROOT_ClassOutput_CR_high.SetBinContent(binnum+1, class_out_hist_CR_high[binnum])
              ROOT_ClassOutput_CR_high.SetBinError(binnum+1, class_out_hist_CR_high[binnum]**(0.5))
              

          ROOT_ClassOutput_SR.Scale(1.0/ROOT_ClassOutput_SR.Integral())
          ROOT_ClassOutput_CR_high.Scale(1.0/ROOT_ClassOutput_CR_high.Integral())


          canvas = ROOT.TCanvas("c1", "c1", 800, 600)
          p1 = ROOT.TPad("p1", "p1", 0.0, 0.3, 1.0, 0.9, 0, 0, 0)
          p1.SetTopMargin(0)
          p1.Draw()
          
          p2 = ROOT.TPad("p2", "p2", 0.0, 0.1, 1.0, 0.3, 0, 0, 0)
          p2.SetTopMargin(0)
          p2.SetBottomMargin(0)
          p2.Draw()

          p1.cd()

          plotlabel = f"Class Output for {process_name}"
          ROOT_ClassOutput_SR.Draw()
          ROOT_ClassOutput_SR.SetTitle(plotlabel)
          ROOT_ClassOutput_SR.SetStats(0)
          min_val = max(0.0001, min(ROOT_ClassOutput_SR.GetMinimum(), ROOT_ClassOutput_CR_high.GetMinimum()))
          max_val = max(ROOT_ClassOutput_SR.GetMaximum(), ROOT_ClassOutput_CR_high.GetMaximum())
          ROOT_ClassOutput_SR.GetYaxis().SetRangeUser(0.1*min_val, 20) # 1000*max_val)

          ROOT_ClassOutput_CR_high.SetLineColor(ROOT.kRed)
          ROOT_ClassOutput_CR_high.Draw("same")


          legend = ROOT.TLegend(0.5, 0.8, 0.9, 0.9)
          legend.AddEntry(ROOT_ClassOutput_SR, f"{process_name} m_bb SR")
          legend.AddEntry(ROOT_ClassOutput_CR_high, f"{process_name} m_bb CR High")
          legend.Draw()


          chi2_value = ROOT_ClassOutput_SR.Chi2Test(ROOT_ClassOutput_CR_high, option='WW')


          pt = ROOT.TPaveText(0.1,0.8,0.4,0.9, "NDC")
          pt.AddText(f"Loss {class_loss}")
          pt.AddText(f"Accuracy {class_accuracy}")
          pt.AddText(f"Chi2 {chi2_value}")
          pt.Draw()

          print(f"Setting canvas to log scale with range {min_val}, {max_val}")
          p1.SetLogy()
          p1.SetGrid()

          p2.cd()

          ROOT_ClassOutput_Ratio = ROOT_ClassOutput_SR.Clone()
          ROOT_ClassOutput_Ratio.Divide(ROOT_ClassOutput_CR_high)
          ROOT_ClassOutput_Ratio.SetTitle("Ratio (SR/CR)")
          ROOT_ClassOutput_Ratio.GetYaxis().SetRangeUser(0.0, 2.0)
          ROOT_ClassOutput_Ratio.Draw()

          p2.SetGrid()


          canvas.SaveAs(os.path.join(save_path, f'{process_name}_ClassOutput_par{parity_index}_M{para_masspoint}.pdf'))





          class_out_hist_SR, bins = np.histogram(pred_signal[SR_mask], bins=nQuantBins, range=(0.0, 1.0), weights=class_weight[SR_mask])
          class_out_hist_CR_high, bins = np.histogram(pred_signal[CR_high_mask], bins=nQuantBins, range=(0.0, 1.0), weights=class_weight[CR_high_mask])

          # Don't use class weights
          # class_out_hist_SR, bins = np.histogram(pred_signal[SR_mask], bins=nQuantBins, range=(0.0, 1.0))
          # class_out_hist_CR_high, bins = np.histogram(pred_signal[CR_high_mask], bins=nQuantBins, range=(0.0, 1.0))

          ROOT_ClassOutput_SR = ROOT.TH1D(f"ClassOutput_{process_name}_SR", f"ClassOutput_{process_name}_SR", nQuantBins, 0.0, 1.0)
          ROOT_ClassOutput_CR_high = ROOT.TH1D(f"ClassOutput_{process_name}_CR_high", f"ClassOutput_{process_name}_CR_high", nQuantBins, 0.0, 1.0)


          for binnum in range(nQuantBins):
              ROOT_ClassOutput_SR.SetBinContent(binnum+1, class_out_hist_SR[binnum])
              ROOT_ClassOutput_SR.SetBinError(binnum+1, class_out_hist_SR[binnum]**(0.5))
              
              ROOT_ClassOutput_CR_high.SetBinContent(binnum+1, class_out_hist_CR_high[binnum])
              ROOT_ClassOutput_CR_high.SetBinError(binnum+1, class_out_hist_CR_high[binnum]**(0.5))
              

          ROOT_ClassOutput_SR.Scale(1.0/ROOT_ClassOutput_SR.Integral())
          ROOT_ClassOutput_CR_high.Scale(1.0/ROOT_ClassOutput_CR_high.Integral())


          canvas = ROOT.TCanvas("c1", "c1", 800, 600)
          p1 = ROOT.TPad("p1", "p1", 0.0, 0.3, 1.0, 0.9, 0, 0, 0)
          p1.SetTopMargin(0)
          p1.Draw()
          
          p2 = ROOT.TPad("p2", "p2", 0.0, 0.1, 1.0, 0.3, 0, 0, 0)
          p2.SetTopMargin(0)
          p2.SetBottomMargin(0)
          p2.Draw()

          p1.cd()

          plotlabel = f"Class Output for {process_name}"
          ROOT_ClassOutput_SR.Draw()
          ROOT_ClassOutput_SR.SetTitle(plotlabel)
          ROOT_ClassOutput_SR.SetStats(0)
          min_val = max(0.0001, min(ROOT_ClassOutput_SR.GetMinimum(), ROOT_ClassOutput_CR_high.GetMinimum()))
          max_val = max(ROOT_ClassOutput_SR.GetMaximum(), ROOT_ClassOutput_CR_high.GetMaximum())
          ROOT_ClassOutput_SR.GetYaxis().SetRangeUser(0.8*min_val, 20) # 1.5*max_val)

          ROOT_ClassOutput_CR_high.SetLineColor(ROOT.kRed)
          ROOT_ClassOutput_CR_high.Draw("same")


          legend = ROOT.TLegend(0.5, 0.8, 0.9, 0.9)
          legend.AddEntry(ROOT_ClassOutput_SR, f"{process_name} m_bb SR")
          legend.AddEntry(ROOT_ClassOutput_CR_high, f"{process_name} m_bb CR High")
          legend.Draw()

          chi2_value = ROOT_ClassOutput_SR.Chi2Test(ROOT_ClassOutput_CR_high, option='WW')


          pt = ROOT.TPaveText(0.1,0.7,0.4,0.9, "NDC")
          pt.AddText(f"Loss {class_loss}")
          pt.AddText(f"Accuracy {class_accuracy}")
          pt.AddText(f"Chi2 {chi2_value}")
          pt.Draw()

          print(f"Setting canvas to log scale with range {min_val}, {max_val}")
          p1.SetLogy()
          p1.SetGrid()

          p2.cd()

          ROOT_ClassOutput_Ratio = ROOT_ClassOutput_SR.Clone()
          ROOT_ClassOutput_Ratio.Divide(ROOT_ClassOutput_CR_high)
          ROOT_ClassOutput_Ratio.SetTitle("Ratio (SR/CR)")
          ROOT_ClassOutput_Ratio.GetYaxis().SetRangeUser(0.0, 2.0)
          ROOT_ClassOutput_Ratio.Draw()

          p2.SetGrid()


          canvas.SaveAs(os.path.join(save_path, f'{process_name}_ClassOutput_par{parity_index}_M{para_masspoint}_raw.pdf'))



          # Adv Plots


          adv_out_hist_SR, bins = np.histogram(pred_adv[SR_mask], bins=quant_binning_adv, range=(0.0, 1.0), weights=adv_weight[SR_mask])
          adv_out_hist_CR_high, bins = np.histogram(pred_adv[CR_high_mask], bins=quant_binning_adv, range=(0.0, 1.0), weights=adv_weight[CR_high_mask])

          ROOT_AdvOutput_SR = ROOT.TH1D(f"AdvOutput_{process_name}_SR", f"AdvOutput_{process_name}_SR", nQuantBins, 0.0, 1.0)
          ROOT_AdvOutput_CR_high = ROOT.TH1D(f"AdvOutput_{process_name}_CR_high", f"AdvOutput_{process_name}_CR_high", nQuantBins, 0.0, 1.0)


          for binnum in range(nQuantBins):
              ROOT_AdvOutput_SR.SetBinContent(binnum+1, adv_out_hist_SR[binnum])
              ROOT_AdvOutput_SR.SetBinError(binnum+1, adv_out_hist_SR[binnum]**(0.5))
              
              ROOT_AdvOutput_CR_high.SetBinContent(binnum+1, adv_out_hist_CR_high[binnum])
              ROOT_AdvOutput_CR_high.SetBinError(binnum+1, adv_out_hist_CR_high[binnum]**(0.5))
              
          if ROOT_AdvOutput_SR.Integral() == 0:
            print(f"Process {process_name} has no adv entries, maybe the weights are all 0 for adv?")
            continue


          ROOT_AdvOutput_SR.Scale(1.0/ROOT_AdvOutput_SR.Integral())
          ROOT_AdvOutput_CR_high.Scale(1.0/ROOT_AdvOutput_CR_high.Integral())




          canvas = ROOT.TCanvas("c1", "c1", 800, 600)
          p1 = ROOT.TPad("p1", "p1", 0.0, 0.3, 1.0, 0.9, 0, 0, 0)
          p1.SetTopMargin(0)
          p1.Draw()
          
          p2 = ROOT.TPad("p2", "p2", 0.0, 0.1, 1.0, 0.3, 0, 0, 0)
          p2.SetTopMargin(0)
          p2.SetBottomMargin(0)
          p2.Draw()

          p1.cd()

          plotlabel = f"Adv Output for {process_name}"
          ROOT_AdvOutput_SR.Draw()
          ROOT_AdvOutput_SR.SetTitle(plotlabel)
          ROOT_AdvOutput_SR.SetStats(0)
          min_val = max(0.0001, min(ROOT_AdvOutput_SR.GetMinimum(), ROOT_AdvOutput_CR_high.GetMinimum()))
          max_val = max(ROOT_AdvOutput_SR.GetMaximum(), ROOT_AdvOutput_CR_high.GetMaximum())
          ROOT_AdvOutput_SR.GetYaxis().SetRangeUser(0.8*min_val, 20) # 1.5*max_val)

          ROOT_AdvOutput_CR_high.SetLineColor(ROOT.kRed)
          ROOT_AdvOutput_CR_high.Draw("same")


          legend = ROOT.TLegend(0.5, 0.8, 0.9, 0.9)
          legend.AddEntry(ROOT_AdvOutput_SR, f"{process_name} m_bb SR")
          legend.AddEntry(ROOT_AdvOutput_CR_high, f"{process_name} m_bb CR High")
          legend.Draw()

          chi2_value = ROOT_AdvOutput_SR.Chi2Test(ROOT_AdvOutput_CR_high, option='WW')


          pt = ROOT.TPaveText(0.1,0.7,0.4,0.9, "NDC")
          pt.AddText(f"Loss {adv_loss}")
          pt.AddText(f"Accuracy {adv_accuracy}")
          pt.AddText(f"True Random Loss {adv_loss_random}")
          pt.AddText(f"Chi2 {chi2_value}")
          pt.Draw()


          print(f"Setting canvas to log scale with range {min_val}, {max_val}")
          p1.SetLogy()
          p1.SetGrid()

          p2.cd()

          ROOT_AdvOutput_Ratio = ROOT_AdvOutput_SR.Clone()
          ROOT_AdvOutput_Ratio.Divide(ROOT_AdvOutput_CR_high)
          ROOT_AdvOutput_Ratio.SetTitle("Ratio (SR/CR)")
          ROOT_AdvOutput_Ratio.GetYaxis().SetRangeUser(0.0, 2.0)
          ROOT_AdvOutput_Ratio.Draw()

          p2.SetGrid()



          canvas.SaveAs(os.path.join(save_path, f'{process_name}_AdvOutput_par{parity_index}_M{para_masspoint}.pdf'))





          adv_out_hist_SR, bins = np.histogram(pred_adv[SR_mask], bins=nQuantBins, range=(0.0, 1.0), weights=adv_weight[SR_mask])
          adv_out_hist_CR_high, bins = np.histogram(pred_adv[CR_high_mask], bins=nQuantBins, range=(0.0, 1.0), weights=adv_weight[CR_high_mask])

          ROOT_AdvOutput_SR = ROOT.TH1D(f"AdvOutput_{process_name}_SR", f"AdvOutput_{process_name}_SR", nQuantBins, 0.0, 1.0)
          ROOT_AdvOutput_CR_high = ROOT.TH1D(f"AdvOutput_{process_name}_CR_high", f"AdvOutput_{process_name}_CR_high", nQuantBins, 0.0, 1.0)


          for binnum in range(nQuantBins):
              ROOT_AdvOutput_SR.SetBinContent(binnum+1, adv_out_hist_SR[binnum])
              ROOT_AdvOutput_SR.SetBinError(binnum+1, adv_out_hist_SR[binnum]**(0.5))
              
              ROOT_AdvOutput_CR_high.SetBinContent(binnum+1, adv_out_hist_CR_high[binnum])
              ROOT_AdvOutput_CR_high.SetBinError(binnum+1, adv_out_hist_CR_high[binnum]**(0.5))
              

          ROOT_AdvOutput_SR.Scale(1.0/ROOT_AdvOutput_SR.Integral())
          ROOT_AdvOutput_CR_high.Scale(1.0/ROOT_AdvOutput_CR_high.Integral())


          canvas = ROOT.TCanvas("c1", "c1", 800, 600)
          p1 = ROOT.TPad("p1", "p1", 0.0, 0.3, 1.0, 0.9, 0, 0, 0)
          p1.SetTopMargin(0)
          p1.Draw()
          
          p2 = ROOT.TPad("p2", "p2", 0.0, 0.1, 1.0, 0.3, 0, 0, 0)
          p2.SetTopMargin(0)
          p2.SetBottomMargin(0)
          p2.Draw()

          p1.cd()

          plotlabel = f"Adv Output for {process_name}"
          ROOT_AdvOutput_SR.Draw()
          ROOT_AdvOutput_SR.SetTitle(plotlabel)
          ROOT_AdvOutput_SR.SetStats(0)
          min_val = max(0.0001, min(ROOT_AdvOutput_SR.GetMinimum(), ROOT_AdvOutput_CR_high.GetMinimum()))
          max_val = max(ROOT_AdvOutput_SR.GetMaximum(), ROOT_AdvOutput_CR_high.GetMaximum())
          ROOT_AdvOutput_SR.GetYaxis().SetRangeUser(0.8*min_val, 20) # 1.5*max_val)

          ROOT_AdvOutput_CR_high.SetLineColor(ROOT.kRed)
          ROOT_AdvOutput_CR_high.Draw("same")


          legend = ROOT.TLegend(0.5, 0.8, 0.9, 0.9)
          legend.AddEntry(ROOT_AdvOutput_SR, f"{process_name} m_bb SR")
          legend.AddEntry(ROOT_AdvOutput_CR_high, f"{process_name} m_bb CR High")
          legend.Draw()

          chi2_value = ROOT_AdvOutput_SR.Chi2Test(ROOT_AdvOutput_CR_high, option='WW')


          pt = ROOT.TPaveText(0.1,0.7,0.4,0.9, "NDC")
          pt.AddText(f"Loss {adv_loss}")
          pt.AddText(f"Accuracy {adv_accuracy}")
          pt.AddText(f"True Random Loss {adv_loss_random}")
          pt.AddText(f"Chi2 {chi2_value}")
          pt.Draw()


          print(f"Setting canvas to log scale with range {min_val}, {max_val}")
          p1.SetLogy()
          p1.SetGrid()

          p2.cd()

          ROOT_AdvOutput_Ratio = ROOT_AdvOutput_SR.Clone()
          ROOT_AdvOutput_Ratio.Divide(ROOT_AdvOutput_CR_high)
          ROOT_AdvOutput_Ratio.SetTitle("Ratio (SR/CR)")
          ROOT_AdvOutput_Ratio.GetYaxis().SetRangeUser(0.0, 2.0)
          ROOT_AdvOutput_Ratio.Draw()

          p2.SetGrid()

          canvas.SaveAs(os.path.join(save_path, f'{process_name}_AdvOutput_par{parity_index}_M{para_masspoint}_raw.pdf'))






@tf.function
def binary_entropy(target, output):
  epsilon = tf.constant(1e-7, dtype=tf.float32)
  x = tf.clip_by_value(output, epsilon, 1 - epsilon)
  return - target * tf.math.log(x) - (1 - target) * tf.math.log(1 - x)

@tf.function
def binary_focal_crossentropy(target, output, y_class, y_pred_class):
    gamma = 2.0 # Default from keras
    gamma = 0.0

    # Use signal from multiclass for focal check
    if y_class is not None:
      y_class = y_class[:,0]
      y_pred_class = y_pred_class[:,0]

    y_true = tf.expand_dims(target, axis=-1)
    y_pred = output


    bce = binary_entropy(y_true, y_pred)
    return bce

    # bce = tf.keras.ops.binary_crossentropy(
    #     target=y_true,
    #     output=y_pred,
    #     from_logits=False,
    # )

    # Calculate focal factor
    # p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    # focal_factor = tf.keras.ops.power(1.0 - p_t, gamma)

    # focal_bce = focal_factor * bce

    # We want to add more weight to BACKGROUNDS (1 - y_class in binary) when they predict to be SIGNAL
    # class_weight_factor = tf.keras.ops.power((1 - y_class) * y_pred_class, gamma)
    # We want when it is true background (y_class == 1) and when it is accurate (y_pred_class ~ 1) the weight is low

    # Need to normalize the y_pred_class to use whole range [-1, 1]
    # Model might learn to only use a small portion (0.5, 1.0) which can cause problems with gamma factor
    y_pred_class_mean = tf.math.reduce_mean(y_pred_class)
    y_pred_class_std = tf.math.reduce_std(y_pred_class)

    y_pred_class_norm = (y_pred_class - y_pred_class_mean)/(2*y_pred_class_std) + 0.5
    y_pred_class_norm_clipped = tf.clip_by_value(y_pred_class_norm, clip_value_min=0, clip_value_max=1)

    # 1 - y_class gives only background (when it is not signal)
    # y_pred_class_norm_clipped gives when the background is expected to be signal (incorrect)
    class_weight_factor = tf.keras.ops.power((1 - y_class) * (y_pred_class_norm_clipped), gamma)
    class_weight_factor = tf.expand_dims(class_weight_factor, axis=-1)
    # I lost my understanding of why this isn't (1 - y_pred_class)

    focal_bce = class_weight_factor * bce

    norm_factor = tf.math.reduce_sum(1 - y_class) / tf.math.reduce_sum(class_weight_factor)

    focal_bce = focal_bce * norm_factor

    return focal_bce

@tf.function
def accuracy(target, output):
  target = tf.expand_dims(target, axis=-1)
  return tf.cast(tf.equal(target, tf.round(output)), tf.float32)



@tf.function
def categorical_crossentropy(target, output):
    epsilon = tf.constant(1e-7, dtype=tf.float32)
    output = output / tf.reduce_sum(output, axis=-1, keepdims=True)
    output = tf.clip_by_value(output, epsilon, 1.0 - epsilon)
    log_prob = tf.math.log(output)
    return - tf.reduce_sum(target * log_prob, axis=-1)

@tf.function
def categorical_accuracy(target, output):
    y_true = target
    y_pred = output

    y_true = tf.argmax(y_true, axis=-1)

    reshape_matches = False
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.convert_to_tensor(y_true, dtype=y_true.dtype)

    y_true_org_shape = tf.shape(y_true)
    y_pred_rank = len(y_pred.shape)
    y_true_rank = len(y_true.shape)

    # If the shape of y_true is (num_samples, 1), squeeze to (num_samples,)
    if (
        (y_true_rank is not None)
        and (y_pred_rank is not None)
        and (len(y_true.shape) == len(y_pred.shape))
    ):
        y_true = tf.squeeze(y_true, -1)
        reshape_matches = True
    y_pred = tf.argmax(y_pred, axis=-1)

    # If the predicted output and actual output types don't match, force cast
    # them to match.
    if y_pred.dtype != y_true.dtype:
        y_pred = tf.cast(y_pred, dtype=y_true.dtype)
    matches = tf.cast(tf.equal(y_true, y_pred), tf.float32)
    if reshape_matches:
        matches = tf.reshape(matches, y_true_org_shape)
    return matches



class EpochCounterCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        self.model.epoch_counter.assign_add(1)



class AdvOnlyCallback(tf.keras.callbacks.Callback):
  def __init__(self, train_dataset, nSteps=100, TrackerWindowSize=10):
    self.train_dataset = train_dataset.repeat()
    self.trackerWindowSize = TrackerWindowSize
    self.nSteps = nSteps
    self.generator = self.looper()

  def looper(self):
    yield
    n_window = 0
    nStep = 0
    for data in self.train_dataset:
      self.model._step_adv_only(data, True)
      n_window += 1
      if n_window == self.trackerWindowSize:
        print(f'\nSubmodule loss {self.model.adv_loss_tracker_submodule.result()} and accuracy {self.model.adv_accuracy_tracker_submodule.result()} after {nStep+1} nSteps')
        self.model.adv_loss_tracker_submodule.reset_state()
        self.model.adv_accuracy_tracker_submodule.reset_state()
        n_window = 0
      nStep += 1
      if nStep == self.nSteps:
        nStep = 0 # This is only a counter, so its fine to reset
        yield

  def on_batch_end(self, batch, logs=None):
    if self.model.epoch_counter == 0: return
    next(self.generator)
    



class ModelCheckpoint(tf.keras.callbacks.Callback):
  def __init__(self, filepath, monitor="val_loss", verbose=0, mode="min", min_delta=None, min_rel_delta=None,
               save_callback=None, patience=None, predicate=None, input_signature=None):
    super().__init__()
    self.monitor = monitor
    self.verbose = verbose
    self.filepath = filepath
    self.epochs_since_last_save = 0
    self.msg = None
    self.save_callback = save_callback
    self.patience = patience
    self.predicate = predicate
    self.input_signature = input_signature

    if os.path.exists(filepath):
      shutil.rmtree(filepath)

    self.best = None
    self.monitor_op = self._make_monitor_op(mode, min_delta, min_rel_delta)

  def _make_monitor_op(self, mode, min_delta, min_rel_delta):
    if mode == "min":
      if min_delta is None and min_rel_delta is None:
        return lambda current, best: best is None or best - current > 0
      if min_delta is None:
        return lambda current, best: best is None or (best - current) > min_rel_delta * best
      if min_rel_delta is None:
        return lambda current, best: best is None or best - current > min_delta
      return lambda current, best: best is None or (best - current) > min_rel_delta * best or best - current > min_delta
    elif mode == "max":
      if min_delta is None and min_rel_delta is None:
        return lambda current, best: best is None or current - best > 0
      if min_delta is None:
        return lambda current, best: best is None or (current - best) > min_rel_delta * best
      if min_rel_delta is None:
        return lambda current, best: best is None or current - best > min_delta
      return lambda current, best: best is None or (current - best) > min_rel_delta * best or current - best > min_delta
    else:
      raise ValueError(f"Unrecognized mode: {mode}")

  def _print_msg(self):
    if self.msg is not None:
      print(self.msg)
      self.msg = None

  def on_epoch_begin(self, epoch, logs=None):
    self._print_msg()

  def on_train_end(self, logs=None):
    self._print_msg()

  def on_epoch_end(self, epoch, logs=None):
    self.epochs_since_last_save += 1
    current = logs.get(self.monitor)
    # if self.monitor_op(current, self.best) and (self.predicate is None or self.predicate(self.model, logs)):
    if True: # For debugging, lets just save all epochs
      dir_name = f'epoch_{epoch+1}.keras'
      onnx_dir_name = f"epoch_{epoch+1}.onnx"
      os.makedirs(self.filepath, exist_ok = True)
      path = os.path.join(self.filepath, f'{dir_name}')
      if self.save_callback is None:
        self.model.save(path)
        if self.input_signature is not None:
          onnx_model, _ = tf2onnx.convert.from_keras(self.model, self.input_signature, opset=13)
          onnx.save(onnx_model, os.path.join(self.filepath, f"{onnx_dir_name}"))

      else:
        self.save_callback(self.model, path)
      path_best = os.path.join(self.filepath, 'best')
      path_best_keras = os.path.join(self.filepath, 'best.keras')
      if os.path.exists(path_best):
        os.remove(path_best)
        os.remove(path_best_keras)

      os.symlink(onnx_dir_name, path_best)
      os.symlink(dir_name, path_best_keras)

      if self.verbose > 0:
        self.msg = f"\nEpoch {epoch+1}: {self.monitor} "
        if self.best is None:
          self.msg += f"= {current:.5f}."
        else:
          self.msg += f"improved from {self.best:.5f} to {current:.5f} after {self.epochs_since_last_save} epochs."
        self.msg += f" Saving model to {path}\n"
      self.best = current
      self.epochs_since_last_save = 0
    if self.patience is not None and self.epochs_since_last_save >= self.patience:
      self.model.stop_training = True
      if self.verbose > 0:
        if self.msg is None:
          self.msg = '\n'
        self.msg = f"Epoch {epoch+1}: early stopping after {self.epochs_since_last_save} epochs."






class AdversarialModel(tf.keras.Model):
  '''Goal: discriminate class0 vs class1 vs class2 without learning features that can guess class_adv'''

  def __init__(self, setup, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.setup = setup

    self.batch_counter = tf.Variable(0)
    self.epoch_counter = tf.Variable(0)

    # self.adv_optimizer = tf.keras.optimizers.AdamW(
    #     learning_rate=setup['adv_learning_rate'],
    #     weight_decay=setup['adv_weight_decay']
    # )

    self.adv_optimizer = tf.keras.optimizers.Adam(
        learning_rate=setup['adv_learning_rate'],
        # weight_decay=setup['adv_weight_decay']
    )

    # self.adv_optimizer = tf.keras.optimizers.Nadam(
    #     learning_rate=setup['adv_learning_rate'],
    #     # weight_decay=setup['adv_weight_decay']
    # )


    self.apply_common_gradients = setup['apply_common_gradients']

    self.class_grad_factor = setup['class_grad_factor']

    # self.class_loss = tf.keras.losses.CategoricalCrossentropy()
    self.class_loss = categorical_crossentropy
    self.class_accuracy = categorical_accuracy
    # self.class_accuracy = tf.keras.metrics.CategoricalAccuracy()
    # self.class_loss = tf.keras.losses.BinaryCrossentropy()
    # self.class_loss = binary_entropy
    # self.class_accuracy = tf.keras.metrics.BinaryAccuracy()


    self.class_loss_tracker = tf.keras.metrics.Mean(name="class_loss")
    self.class_accuracy_tracker = tf.keras.metrics.Mean(name="class_accuracy")


    self.adv_grad_factor = setup['adv_grad_factor']

    # self.adv_loss = tf.keras.losses.BinaryCrossentropy()
    # self.adv_accuracy = tf.keras.metrics.BinaryAccuracy()

    # self.adv_loss = binary_entropy
    self.adv_loss = binary_focal_crossentropy
    self.adv_accuracy = accuracy

    self.adv_loss_tracker = tf.keras.metrics.Mean(name="adv_loss")
    self.adv_accuracy_tracker = tf.keras.metrics.Mean(name="adv_accuracy")

    self.adv_loss_tracker_submodule = tf.keras.metrics.Mean(name="adv_loss")
    self.adv_accuracy_tracker_submodule = tf.keras.metrics.Mean(name="adv_accuracy")


    self.common_layers = []

    def add_layer(layer_list, n_units, activation, name):
      layer = tf.keras.layers.Dense(n_units, activation=activation, name=name)
      layer_list.append(layer)
      if setup['dropout'] > 0:
        dropout = tf.keras.layers.Dropout(setup['dropout'], name=name + '_dropout')
        layer_list.append(dropout)
      if setup['use_batch_norm']:
        batch_norm = tf.keras.layers.BatchNormalization(name=name + '_batch_norm')
        layer_list.append(batch_norm)

    for n in range(setup['n_common_layers']):
      add_layer(self.common_layers, setup['n_common_units'], setup['activation'], f'common_{n}')

    self.class_layers = []
    self.adv_layers = []
    for n in range(setup['n_adv_layers']):
      add_layer(self.class_layers, setup['n_adv_units'], setup['activation'], f'class_{n}')
      # add_layer(self.adv_layers, setup['n_adv_units'], setup['activation'], f'adv_{n}')
      add_layer(self.adv_layers, setup['n_adv_units'], 'relu', f'adv_{n}')

    add_layer(self.adv_layers, setup['n_adv_units'], 'relu', f'adv_{n+1}')
    add_layer(self.adv_layers, setup['n_adv_units'], 'relu', f'adv_{n+2}')
    add_layer(self.adv_layers, setup['n_adv_units'], 'relu', f'adv_{n+3}')


    self.class_output = tf.keras.layers.Dense(2, activation='softmax', name='class_output')

    self.adv_output = tf.keras.layers.Dense(1, activation='sigmoid', name='adv_output')

    self.output_names = ['class_output', 'adv_output']

  def call(self, x):
    x_common = self.call_common(x)
    class_output = self.call_class(x_common)
    adv_output = self.call_adv(x_common)
    return class_output, adv_output

  def call_common(self, x):
    for layer in self.common_layers:
      x = layer(x)
    return x

  def call_class(self, x_common):
    x = x_common
    for layer in self.class_layers:
      x = layer(x)
    class_output = self.class_output(x)
    return class_output

  def call_adv(self, x_common):
    x = x_common
    for layer in self.adv_layers:
      x = layer(x)
    adv_output = self.adv_output(x)
    return adv_output

  def _step(self, data, training):
    x, y = data

    y_class = tf.cast(y[0], dtype=tf.float32)
    y_adv = tf.cast(y[1], dtype=tf.float32)

    class_weight = tf.cast(y[2], dtype=tf.float32)
    adv_weight = tf.cast(y[3], dtype=tf.float32)
    

    def compute_losses():
      y_pred_class, y_pred_adv = self(x, training=training)

      class_loss_vec = self.class_loss(y_class, y_pred_class)

      class_loss = tf.reduce_mean(class_loss_vec * class_weight)

      adv_loss_vec = self.adv_loss(y_adv, y_pred_adv, y_class, y_pred_class) # Focal loss
      # adv_loss_vec = self.adv_loss(y_adv, y_pred_adv)
      # We want to apply some weights onto the adv loss vector
      # This is to have the SignalRegion and ControlRegion have equal weights

      adv_loss = tf.reduce_mean(adv_loss_vec * adv_weight)
      return y_pred_class, class_loss_vec, class_loss, y_pred_adv, adv_loss_vec, adv_loss

    if training:
      with tf.GradientTape() as class_tape, tf.GradientTape() as adv_tape:
        y_pred_class, class_loss_vec, class_loss, y_pred_adv, adv_loss_vec, adv_loss = compute_losses()
    else:
      y_pred_class, class_loss_vec, class_loss, y_pred_adv, adv_loss_vec, adv_loss = compute_losses()

    class_accuracy_vec = self.class_accuracy(y_class, y_pred_class)

    self.class_loss_tracker.update_state(class_loss_vec, sample_weight=class_weight)
    self.class_accuracy_tracker.update_state(class_accuracy_vec, sample_weight=class_weight)


    adv_accuracy_vec = self.adv_accuracy(y_adv, y_pred_adv)


    self.adv_loss_tracker.update_state(adv_loss_vec, sample_weight=adv_weight)
    self.adv_accuracy_tracker.update_state(adv_accuracy_vec, sample_weight=adv_weight)

    if training:
      common_vars = [ var for var in self.trainable_variables if "/common" in var.path ]
      class_vars = [ var for var in self.trainable_variables if "/class" in var.path ]
      adv_vars = [ var for var in self.trainable_variables if "/adv" in var.path ]
      n_common_vars = len(common_vars)


      grad_class = class_tape.gradient(class_loss, common_vars + class_vars)
      grad_class_excl = grad_class[n_common_vars:]

      grad_adv = adv_tape.gradient(adv_loss, common_vars + adv_vars)
      grad_adv_excl = grad_adv[n_common_vars:]

    #   grad_common = [ self.class_grad_factor * grad_class[i] \
    #                   for i in range(len(common_vars)) ]

      grad_common = [ self.class_grad_factor * grad_class[i] - self.adv_grad_factor * grad_adv[i] \
                      for i in range(len(common_vars)) ]

      grad_common_no_adv = [ grad_class[i] \
                      for i in range(len(common_vars)) ]

      grad_common_only_adv = [ grad_adv[i] \
                      for i in range(len(common_vars)) ]

    #   tf.print("We have to understand why it is not becoming blind")
    #   tf.print(grad_common)
    #   tf.print(grad_common_no_adv)
    #   tf.print(grad_common_only_adv)


      @tf.function
      def cond_true_fn():
        if self.apply_common_gradients:
          if self.epoch_counter == 0:
            self.optimizer.apply_gradients(zip(grad_common_no_adv + grad_class_excl, common_vars + class_vars))
          else:
            self.optimizer.apply_gradients(zip(grad_common + grad_class_excl, common_vars + class_vars))
        # tf.print(f"Batch counter mod 10! {self.batch_counter}")
        return

      @tf.function
      def cond_false_fn():
        # tf.print(f"Batch counter not mod 10! {self.batch_counter}")
        return


      # tf.cond(
      #   (self.batch_counter%10) == (self.epoch_counter%10),
      #   true_fn = cond_true_fn,
      #   false_fn = cond_false_fn
      # )


      cond_true_fn()
      self.adv_optimizer.apply_gradients(zip(grad_adv_excl, adv_vars))


      # if self.apply_common_gradients:
      #   self.optimizer.apply_gradients(zip(grad_common + grad_class_excl, common_vars + class_vars))
      # self.adv_optimizer.apply_gradients(zip(grad_adv_excl, adv_vars))


    return { m.name: m.result() for m in self.metrics }




  def _step_adv_only(self, data, training):
    x, y = data

    y_adv = tf.cast(y[1], dtype=tf.float32)

    adv_weight = tf.cast(y[3], dtype=tf.float32)
    

    def compute_losses(x_common):
      y_pred_adv = self.call_adv(x_common)

      adv_loss_vec = self.adv_loss(y_adv, y_pred_adv, None, None) # Focal loss
      # adv_loss_vec = self.adv_loss(y_adv, y_pred_adv)
      # We want to apply some weights onto the adv loss vector
      # This is to have the SignalRegion and ControlRegion have equal weights

      adv_loss = tf.reduce_mean(adv_loss_vec * adv_weight)
      return y_pred_adv, adv_loss_vec, adv_loss

    if training:
      x_common = self.call_common(x)
      with tf.GradientTape() as adv_tape:
        y_pred_adv, adv_loss_vec, adv_loss = compute_losses(x_common)
    else:
      y_pred_adv, adv_loss_vec, adv_loss = compute_losses()


    adv_accuracy_vec = self.adv_accuracy(y_adv, y_pred_adv)

    self.adv_loss_tracker_submodule.update_state(adv_loss_vec, sample_weight=adv_weight)
    self.adv_accuracy_tracker_submodule.update_state(adv_accuracy_vec, sample_weight=adv_weight)

    if training:
      adv_vars = [ var for var in self.trainable_variables if "/adv" in var.path ]


      grad_adv = adv_tape.gradient(adv_loss, adv_vars)

      self.adv_optimizer.apply_gradients(zip(grad_adv, adv_vars))

    return

 

  def train_step(self, data):
    self.batch_counter = self.batch_counter.assign_add(1)
    return self._step(data, training=True)

  def test_step(self, data):
    return self._step(data, training=False)

  @property
  def metrics(self):
    return [
          self.class_loss_tracker,
          self.class_accuracy_tracker,

          self.adv_loss_tracker,
          self.adv_accuracy_tracker,
    ]









def train_dnn():
    input_folder = "DNN_Datasets/Dataset_2025-03-28-12-49-16"
    yaml_list = [fname for fname in os.listdir(input_folder) if fname.startswith('batch_config_parity')]

    modelname_parity = []

    for i, config_yaml in enumerate(yaml_list):
        config_dict = {}
        with open(os.path.join(input_folder, config_yaml), 'r') as file:
            config_dict = yaml.safe_load(file)  

        val_config_dict = {}
        val_yaml = yaml_list[i+1] if (i+1) != len(yaml_list) else yaml_list[0]
        with open(os.path.join(input_folder, val_yaml), 'r') as file:
            val_config_dict = yaml.safe_load(file)  

        batch_size = config_dict['meta_data']['batch_dict']['batch_size']
        val_batch_size = val_config_dict['meta_data']['batch_dict']['batch_size']

        input_file_name = os.path.join(input_folder, config_dict['meta_data']['input_filename'])
        input_weight_name = os.path.join(input_folder, f"weight{config_dict['meta_data']['input_filename'][5:]}")

        input_file_name_val = os.path.join(input_folder, val_config_dict['meta_data']['input_filename'])
        input_weight_name_val = os.path.join(input_folder, f"weight{val_config_dict['meta_data']['input_filename'][5:]}")


        output_folder = "DNN_Models/v17"
        model_name = config_dict['meta_data']['output_DNNname']
        output_dnn_name = os.path.join(output_folder, model_name)



        dw = DataWrapper()
        dw.AddInputFeatures(['lep1_pt', 'lep1_phi', 'lep1_eta', 'lep1_mass'])
        dw.AddInputFeatures(['lep2_pt', 'lep2_phi', 'lep2_eta', 'lep2_mass'])
        dw.AddInputFeatures(['met_pt', 'met_phi'])
        dw.AddInputFeaturesList(['centralJet_pt', 'centralJet_phi', 'centralJet_eta', 'centralJet_mass'], 0)
        dw.AddInputFeaturesList(['centralJet_pt', 'centralJet_phi', 'centralJet_eta', 'centralJet_mass'], 1)
        dw.AddInputFeaturesList(['centralJet_pt', 'centralJet_phi', 'centralJet_eta', 'centralJet_mass'], 2)
        dw.AddInputFeaturesList(['centralJet_pt', 'centralJet_phi', 'centralJet_eta', 'centralJet_mass'], 3)
        dw.AddHighLevelFeatures([
                                'HT', 'dR_dilep', 'dR_dibjet', 
                                'dR_dilep_dibjet', 'dR_dilep_dijet',
                                'dPhi_lep1_lep2', 'dPhi_jet1_jet2',
                                'dPhi_MET_dilep', 'dPhi_MET_dibjet',
                                'min_dR_lep0_jets', 'min_dR_lep1_jets',
                                'MT', 'MT2_ll', 'MT2_bb', 'MT2_blbl',
                                'll_mass', 'CosTheta_bb'
                                ])


        dw.SetBinary(True)
        # dw.UseParametric(True)
        dw.UseParametric(False)
        dw.SetParamList([ 250, 260, 270, 280, 300, 350, 450, 550, 600, 650, 700, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000, 4000, 5000 ])
        dw.SetOutputFolder(output_folder)

        dw.AddInputLabel('sample_type')

        dw.SetMbbName('bb_mass_PNetRegPtRawCorr_PNetRegPtRawCorrNeutrino')

        # Prep a test dw
        # Must copy before reading file so we can read the test file instead
        dw_val = copy.deepcopy(dw)

        dw.ReadFile(input_file_name)
        dw.ReadWeightFile(input_weight_name)
        print(config_dict)
        dw.DefineTrainTestSet(batch_size, 0.0)


        dw_val.ReadFile(input_file_name_val)
        dw_val.ReadWeightFile(input_weight_name_val)
        dw_val.DefineTrainTestSet(val_batch_size, 0.0)


        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        tf.random.set_seed(42)


        setup = {
            'learning_rate': 0.0001,
            'adv_learning_rate': 0.0001,
            'weight_decay': 0.004,
            'adv_weight_decay': 0.004,
            'adv_grad_factor': 1.0, #0.7
            'class_grad_factor': 0.0001,
            'activation': 'tanh', #'relu' # This one will be deleted
            'common_activation': 'tanh', #'relu'
            'class_activation': 'tanh', #'relu'
            'adv_activation': 'relu', #'relu'
            'use_batch_norm': False,
            'dropout': 0.0,
            'n_common_layers': 10,
            'n_common_units': 256,
            'n_adv_layers': 3,
            'n_adv_units': 128,
            'n_epochs': 20,
            'patience': 100,
            'apply_common_gradients': True,
        }


        model = AdversarialModel(setup)
        model.compile(loss=None,
                    # optimizer=tf.keras.optimizers.AdamW(learning_rate=setup['learning_rate'],
                    #                                     weight_decay=setup['weight_decay']))
                    optimizer=tf.keras.optimizers.Nadam(learning_rate=setup['learning_rate'],
                                                        # weight_decay=setup['weight_decay']))
                    )
        )


        model(dw.train_features)

        model.summary()



        train_tf_dataset = tf.data.Dataset.from_tensor_slices((dw.train_features, (tf.one_hot(dw.train_labels_binary, 2), dw.train_mbb, dw.train_class_weight, dw.train_adv_weight))).batch(batch_size)

        val_tf_dataset = tf.data.Dataset.from_tensor_slices((dw_val.train_features, (tf.one_hot(dw_val.train_labels_binary, 2), dw_val.train_mbb, dw_val.train_class_weight, dw_val.train_adv_weight))).batch(val_batch_size)


        def save_predicate(model, logs):
            return abs(logs['val_adv_accuracy'] - 0.5) < 0.001


        input_shape = [None, dw.train_features.shape[1]]
        input_signature = [tf.TensorSpec(input_shape, tf.double, name='x')]
        callbacks = [
            ModelCheckpoint(output_dnn_name, verbose=1, monitor="val_class_loss", mode='min', min_rel_delta=1e-3,
                            patience=setup['patience'], save_callback=None, predicate=save_predicate, input_signature=input_signature),
            tf.keras.callbacks.CSVLogger(f'{output_dnn_name}_training_log.csv', append=True),
            EpochCounterCallback(),
            AdvOnlyCallback(train_tf_dataset, nSteps=10, TrackerWindowSize=10),
        ]



        print("Save model configuration")
        modelname_parity.append([model_name, config_dict['meta_data']['iterate_cut']])
        features_config = {
            'features': dw.feature_names,
            'listfeatures': dw.listfeature_names,
            'highlevelfeatures': dw.highlevelfeatures_names,
            'use_parametric': dw.use_parametric,
            'modelname_parity': modelname_parity,
            'parametric_list': dw.param_list,
            'model_setup': setup,
        }
        
        with open(os.path.join(dw.output_folder, 'dnn_config.yaml'), 'w') as file:
            yaml.dump(features_config, file)



        # print("Fit model")
        # history = model.fit(
        #     dw.train_features,
        #     # (dw.train_labels, dw.train_mbb),
        #     (tf.one_hot(dw.train_labels_binary, 2), dw.train_mbb, dw.train_class_weight, dw.train_adv_weight),
        #     validation_data=(
        #         dw_val.train_features,
        #         # (dw.test_labels, dw.test_mbb)
        #         # (tf.one_hot(dw.test_labels_binary, 2), dw.test_mbb, dw.test_class_weight, dw.test_adv_weight)
        #         (tf.one_hot(dw_val.train_labels_binary, 2), dw_val.train_mbb, dw_val.train_class_weight, dw_val.train_adv_weight)
        #     ),
        #     verbose=1,
        #     batch_size=batch_size,
        #     epochs=setup['n_epochs'],
        #     shuffle=False,
        #     callbacks=callbacks,
        #     validation_batch_size=val_batch_size,
        # )
        


        print("Fit model")
        history = model.fit(
            train_tf_dataset,
            validation_data=val_tf_dataset,
            verbose=1,
            epochs=setup['n_epochs'],
            shuffle=False,
            callbacks=callbacks,
        )



        model.save(f"{output_dnn_name}.keras")

        input_shape = [None, dw.train_features.shape[1]]
        input_signature = [tf.TensorSpec(input_shape, tf.double, name='x')]
        onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)
        onnx.save(onnx_model, f"{output_dnn_name}.onnx")




        return



def adv_only_training(model_name, model_config, train_file, train_weight, test_file, test_weight, batch_size=1000):

  print("Can I just continue training the same model?")

  dnnConfig = {}
  with open(model_config, 'r') as file:
    dnnConfig = yaml.safe_load(file)  


  #Features to use for DNN application (single vals)
  features = dnnConfig['features']
  #Features to use for DNN application (vectors and index)
  list_features = dnnConfig['listfeatures']
  list_features_dict = {}
  for list_feature in list_features:
    if str(list_feature[1]) not in list_features_dict.keys():
      list_features_dict[str(list_feature[1])] = []
    list_features_dict[str(list_feature[1])].append(list_feature[0])

  #Features to use for DNN application (high level names to create)
  highlevel_features = dnnConfig['highlevelfeatures']

  parametric_list = dnnConfig['parametric_list']

  dw = DataWrapper()
  dw.AddInputFeatures(features)
  for list_feature_key in list_features_dict.keys():
    dw.AddInputFeaturesList(list_features_dict[list_feature_key], int(list_feature_key))
  dw.AddHighLevelFeatures(highlevel_features)

  dw.SetBinary(True)
  dw.UseParametric(False)
  dw.SetParamList(parametric_list)

  dw.AddInputLabel('sample_type')

  dw.SetMbbName('bb_mass_PNetRegPtRawCorr_PNetRegPtRawCorrNeutrino')


  dw_val = copy.deepcopy(dw)

  dw.ReadFile(train_file)
  dw.ReadWeightFile(train_weight)
  dw.DefineTrainTestSet(batch_size, 0.0)

  dw_val.ReadFile(test_file)
  dw_val.ReadWeightFile(test_weight)
  dw_val.DefineTrainTestSet(batch_size, 0.0)


  setup2 = dnnConfig['model_setup']
  setup2['apply_common_gradients'] = False
  setup2['adv_learning_rate'] = 0.01

  model2 = AdversarialModel(setup2)
  model2.compile(loss=None,
              optimizer=tf.keras.optimizers.Nadam(learning_rate=setup2['learning_rate'],
                                                  # weight_decay=setup2['weight_decay']))
              ))

  model2(dw.train_features)

  model2.summary()


  print("Now we will load weights onto model 2")
  # model2.load_weights(os.path.join(output_dnn_name, 'best.keras'))
  model2.load_weights(model_name, skip_mismatch=True)

  # for i, layer in enumerate(model2.layers):
  #   print(f"On layer {i} name {layer.name}")
  #   print(layer.get_config(), layer.get_weights())

  for layer in model2.layers:
      layer_name = layer.name
      if not layer_name.startswith("adv"):
         print(f"Not an adv layer {layer_name}")
         continue

      if not isinstance(layer, tf.keras.layers.Dense):
          raise ValueError(f"Layer '{layer_name}' is not a Dense layer.")
      kernel_weights = layer.kernel_initializer(layer.kernel.shape)
      bias_weights = layer.bias_initializer(layer.bias.shape)
      layer.set_weights([kernel_weights, bias_weights])


  # for i, layer in enumerate(model2.layers):
  #   print(f"On layer {i} name {layer.name}")
  #   print(layer.get_config(), layer.get_weights())



  print("And continue training")
  history2 = model2.fit(
      dw.train_features,
      # (dw.train_labels, dw.train_mbb),
      (tf.one_hot(dw.train_labels_binary, 2), dw.train_mbb, dw.train_class_weight, dw.train_adv_weight),
      validation_data=(
          dw_val.train_features,
          # (dw.test_labels, dw.test_mbb)
          # (tf.one_hot(dw.test_labels_binary, 2), dw.test_mbb, dw.test_class_weight, dw.test_adv_weight)
          (tf.one_hot(dw_val.train_labels_binary, 2), dw_val.train_mbb, dw_val.train_class_weight, dw_val.train_adv_weight)
      ),
      verbose=1,
      batch_size=batch_size,
      epochs=setup2['n_epochs'],
      shuffle=False,
      validation_batch_size=batch_size,
  )


  return


  # PlotMetric(history, model_name, "class_loss", output_folder)
  # PlotMetric(history, model_name, "adv_loss", output_folder)



  # # But model is what was the last epoch, not necessarily the best model (if early stopping triggered with patience)
  # # Need to load the best model file

  # model_load_name = os.path.join(output_dnn_name, 'best')
  # print(f"Model load {model_load_name}")
  # sess = ort.InferenceSession(model_load_name)

  # print(f"Type sess {type(sess)}")
  # print(f"Type model {type(model)}")
  # validate = True
  # if validate:
  #     # Cool! In the debugging stage, now lets also predict the output on batch1 file!
  #     other_parity_files = [os.path.join(input_folder, fname) for fname in os.listdir(input_folder) if fname.startswith('batchfile') and (fname != config_dict['meta_data']['input_filename'])]
  #     for i, other_parity_file in enumerate(other_parity_files):
  #         dw.ReadFile(other_parity_file)

  #         # dw.validate_output(model, model_name, i)
  #         dw.validate_output(sess, model_name, i)
  #         break # For now, don't validate on each nParity 

  # return # And just end training after first model



def validate_model(model_name, model_config, validation_file, validation_weight, nParity):
    # model_load_name = os.path.join(model_name, 'best')
    model_load_name = model_name
    print(f"Model load {model_load_name}")
    sess = ort.InferenceSession(model_load_name)

    print(f"Type sess {type(sess)}")


    dnnConfig = {}
    with open(model_config, 'r') as file:
        dnnConfig = yaml.safe_load(file)  


    #Features to use for DNN application (single vals)
    features = dnnConfig['features']
    #Features to use for DNN application (vectors and index)
    list_features = dnnConfig['listfeatures']
    list_features_dict = {}
    for list_feature in list_features:
      if str(list_feature[1]) not in list_features_dict.keys():
        list_features_dict[str(list_feature[1])] = []
      list_features_dict[str(list_feature[1])].append(list_feature[0])

    #Features to use for DNN application (high level names to create)
    highlevel_features = dnnConfig['highlevelfeatures']

    parametric_list = dnnConfig['parametric_list']

    dw = DataWrapper()
    dw.AddInputFeatures(features)
    for list_feature_key in list_features_dict.keys():
      dw.AddInputFeaturesList(list_features_dict[list_feature_key], int(list_feature_key))
    dw.AddHighLevelFeatures(highlevel_features)

    dw.SetBinary(True)
    dw.UseParametric(False)
    dw.SetParamList(parametric_list)

    dw.AddInputLabel('sample_type')

    dw.SetMbbName('bb_mass_PNetRegPtRawCorr_PNetRegPtRawCorrNeutrino')

    dw.ReadFile(validation_file)
    dw.ReadWeightFile(validation_weight)
    dw.validate_output(sess, model_name, nParity)




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Create TrainTest Files for DNN.')
    #parser.add_argument('--config-folder', required=True, type=str, help="Config Folder from Step1")

    args = parser.parse_args()

    # model = train_dnn()

    model_name = f"DNN_Models/v17/ResHH_Classifier_parity0/epoch_20.keras"
    model_config = "DNN_Models/v17/dnn_config.yaml"
    train_file = f"DNN_Datasets/Dataset_2025-03-28-12-49-16/batchfile0.root"
    train_weight = f"DNN_Datasets/Dataset_2025-03-28-12-49-16/weightfile0.root"
    test_file = f"DNN_Datasets/Dataset_2025-03-28-12-49-16/batchfile1.root"
    test_weight = f"DNN_Datasets/Dataset_2025-03-28-12-49-16/weightfile1.root"
    adv_only_training(model_name, model_config, train_file, train_weight, test_file, test_weight)

    model_name = f"DNN_Models/v17/ResHH_Classifier_parity0/epoch_20.onnx"
    validate_model(model_name, model_config, test_file, test_weight, 1)


    # for i in range(4):
    #   j = i+1 if i != 3 else 0
    #   model_name = f"DNN_Models/v16/ResHH_Classifier_parity{i}/"
    #   model_config = "DNN_Models/v16/dnn_config.yaml"
    #   validation_file = f"DNN_Datasets/Dataset_2025-03-28-12-49-16/batchfile{j}.root"
    #   weight_file = f"DNN_Datasets/Dataset_2025-03-28-12-49-16/weightfile{j}.root"
    #   validate_model(model_name, model_config, validation_file, weight_file, j)
    #   break
