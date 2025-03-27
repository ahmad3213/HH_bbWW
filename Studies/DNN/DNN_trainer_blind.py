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
        self.value_to_label = {'1': 0, '8': 1, '5': 2}
        self.label_names = ["Signal", "TT", "DY"]

        self.value_to_label_binary = {'1': 0, '8': 1, '5': 1} # Binary for now

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
        self.mbb_CR_weight = None
        self.mbb_SR_weight = None

        self.class_weight = None
        self.adv_weight = None

        self.class_weights_lut = None

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
        
    def ReadFile(self, file_name):
        if self.feature_names == None:
            print("Uknown branches to read! DefineInputFeatures first!")
            return
        if self.label_name == None:
            print("No label branch defined!")
            return

        file = uproot.open(file_name)
        tree = file['Events']
        branches = tree.arrays()

        self.features = np.array([getattr(branches, feature_name) for feature_name in self.feature_names]).transpose()
        print("Got features, but its a np array")

        default_value = 0.0
        if self.listfeature_names != None: 
            self.listfeatures = np.array([ak.fill_none(ak.pad_none(getattr(branches, feature_name), index+1), default_value)[:,index] for [feature_name,index] in self.listfeature_names]).transpose()
        print("Got the list features")

        #Need to append the value features and the listfeatures together
        if self.listfeature_names != None: 
            print("We have list features!")
            print(self.features)
            self.features = np.append(self.features, self.listfeatures, axis=1)
            print(self.features)

        if self.highlevelfeatures_names != None: 
            self.hlv = np.array([getattr(branches, feature_name) for feature_name in self.highlevelfeatures_names]).transpose()
            self.features = np.append(self.features, self.hlv, axis=1)


        labels_branch_values = np.array(getattr(branches, self.label_name))

        self.labels = np.array([self.value_to_label[str(branch_val)] for branch_val in labels_branch_values])
        self.labels_binary = np.array([self.value_to_label_binary[str(branch_val)] for branch_val in labels_branch_values])
        print("Got labels")

        # self.mbb_region, self.mbb_CR_weight, self.mbb_SR_weight = self.SetMbbRegion(branches)
        # print(f"Old CR weight {self.mbb_CR_weight} SR weight {self.mbb_SR_weight}")
        self.mbb, self.mbb_region, self.mbb_region_binary, self.mbb_CR_weight, self.mbb_SR_weight = self.SetMbbRegion(branches)
        # print(f"New CR weight {self.mbb_CR_weight} SR weight {self.mbb_SR_weight}")

        self.class_weights_lut = self.SetClassWeights()

        # self.mbb_CR_weight = np.sum(self.mbb_region == 0)/len(self.mbb_region)
        # self.mbb_SR_weight = np.sum(self.mbb_region == 1)/len(self.mbb_region) # Calculate ratio of signal to control
        print(len(self.mbb_region))
        print(f"Calculated the mbb_CR_weight {self.mbb_CR_weight} giving total weight {self.mbb_CR_weight * np.sum((self.mbb_region != 0) & (self.labels != 0))}")
        print(f"Calculated the mbb_SR_weight {self.mbb_SR_weight} giving total weight {self.mbb_SR_weight * np.sum((self.mbb_region == 0) & (self.labels != 0))}")


        #Add parametric variable
        self.param_values = np.array([[x if (x > 0) else np.random.choice(self.param_list) for x in getattr(branches, 'X_mass') ]]).transpose()
        print("Got the param values")


        self.features_no_param = self.features
        if self.use_parametric: self.features = np.append(self.features, self.param_values, axis=1)


    def ReadWeightFile(self, weight_name):
        file = uproot.open(weight_name)
        tree = file['weight_tree']
        branches = tree.arrays()
        self.class_weight = np.array(getattr(branches, 'class_weight'))
        self.adv_weight = np.array(getattr(branches, 'adv_weight'))

    def SetMbbRegion_old(self, branches):
        print("Inside setting mbb region!")
        # But we want to blind the adversarial part to Signal events, meaning we must filter them out
        mbb = np.array(getattr(branches, self.mbb_name))
        mbb_region = np.array(abs(mbb - 125) < 25)
        nTotal = np.sum(self.labels != 0)
        print(f"We have {nTotal} background events")

        mbb_CR_weight = tf.cast(nTotal / (np.sum((mbb_region == 0) & (self.labels != 0)) * 2.0), tf.float32)
        mbb_SR_weight = tf.cast(nTotal / (np.sum((mbb_region == 1) & (self.labels != 0)) * 2.0), tf.float32)

        print(f"Pointing to {np.sum((mbb_region == 0) & (self.labels != 0))} CR events and {np.sum((mbb_region == 1) & (self.labels != 0))} SR events")

        return mbb_region, mbb_region_binary, mbb_CR_weight, mbb_SR_weight

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
        nTotal = np.sum(self.labels != 0)
        print(f"We have {nTotal} background events")

        mbb_CR_weight = tf.cast(nTotal / (np.sum(((mbb_region == -1) | (mbb_region == 1)) & (self.labels != 0)) * 2.0), tf.float32)
        mbb_SR_weight = tf.cast(nTotal / (np.sum((mbb_region == 0) & (self.labels != 0)) * 2.0), tf.float32)

        print(f"Pointing to {np.sum((mbb_region == 0) & (self.labels != 0))} CR events and {np.sum((mbb_region == 1) & (self.labels != 0))} SR events")

        mbb_region_binary = np.where(
          (mbb_region == 0),
          1,
          0
        )

        return mbb, mbb_region, mbb_region_binary, mbb_CR_weight, mbb_SR_weight



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
        self.train_class_weight = self.class_weight[trainStart:trainEnd]
        self.train_adv_weight = self.adv_weight[trainStart:trainEnd]
        
        self.test_features = self.features[testStart:testEnd]
        self.test_labels = self.labels[testStart:testEnd]
        self.test_labels_binary = self.labels_binary[testStart:testEnd]
        self.test_mbb = self.mbb_region_binary[testStart:testEnd]
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

        os.makedirs(os.path.join(self.output_folder, model_name), exist_ok=True)

        for para_masspoint in self.param_list:
            print(f"Looking at mass {para_masspoint}")
            if para_masspoint > 1000: continue
            if para_masspoint not in [300, 450, 550, 800]: continue
            predict_list = []
            weight_list = []


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

                features_this_sample = self.features_paramSet[this_sample_mask]
                features_this_sample_no_parametric = self.features_no_param[this_sample_mask]
                mbb_region_this_sample = self.mbb_region[this_sample_mask]
                mbb_this_sample = self.mbb[this_sample_mask]

                features_to_use = features_this_sample if self.use_parametric else features_this_sample_no_parametric

                pred = sess.run(None, {'x': features_to_use})
                pred = pred[0][:,0] #Only get first entry (category prediction), all events, then first category (signal)


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
                    ratioplot.SetStats(0)
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
    y_class = y_class[:,0]
    y_pred_class = y_pred_class[:,0]

    y_true = tf.expand_dims(target, axis=-1)
    y_pred = output

    bce = binary_entropy(y_true, y_pred)

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
    if self.monitor_op(current, self.best) and (self.predicate is None or self.predicate(self.model, logs)):
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
      if os.path.exists(path_best):
        os.remove(path_best)

      os.symlink(onnx_dir_name, path_best)

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

    self.adv_optimizer = tf.keras.optimizers.AdamW(
        learning_rate=setup['adv_learning_rate'],
        weight_decay=setup['adv_weight_decay']
    )

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
      add_layer(self.adv_layers, setup['n_adv_units'], setup['activation'], f'adv_{n}')

    self.class_output = tf.keras.layers.Dense(2, activation='softmax', name='class_output')
    # self.adv_output = tf.keras.layers.Dense(2, activation='softmax', name='adv_output')

    # self.class_output = tf.keras.layers.Dense(1, activation='sigmoid', name='class_output')
    self.adv_output = tf.keras.layers.Dense(1, activation='sigmoid', name='adv_output')

    self.output_names = ['class_output', 'adv_output']

    self.class_weights_lut = setup['class_weights_lut']

  def call(self, x):
    for layer in self.common_layers:
      x = layer(x)
    x_common = x
    for layer in self.class_layers:
      x = layer(x)
    class_output = self.class_output(x)
    x = x_common
    for layer in self.adv_layers:
      x = layer(x)
    adv_output = self.adv_output(x)
    return class_output, adv_output

  def _step(self, data, training):
    x, y = data

    y_class = tf.cast(y[0], dtype=tf.float32)
    y_adv = tf.cast(y[1], dtype=tf.float32)

    class_weight = tf.cast(y[2], dtype=tf.float32)
    adv_weight = tf.cast(y[3], dtype=tf.float32)
    

    # class_weights = tf.where(
    #   y_class[:,0] == 1,
    #   self.class_weights_lut[0],
    #   self.class_weights_lut[1]
    # )


    # adv_weights = tf.where(
    #   y_adv == 0,
    #   self.setup['mbb_CR_weight'],
    #   self.setup['mbb_SR_weight']
    # )
    # Only use adv part when true category is signal
    # true_signal_mask = tf.expand_dims(y_class, axis=-1) == 0

    # true_signal_mask = (y_class[:,0] == 1)

    # adv_weights = tf.where(
    #   true_signal_mask,
    #   0.0,
    #   adv_weights
    # )

    # adv_weights = tf.expand_dims(adv_weights, axis=-1)


    # And maybe we want to only apply adv part on some loose DNN cut
    # This can't be done here since we don't have y_pred_class available yet
    # low_signal_prediction_mask = tf.expand_dims(y_pred_class[:,0], axis=-1) <= 0.3
    # adv_weights = tf.where(
    #   low_signal_prediction_mask,
    #   0.0,
    #   adv_weights
    # )

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

      grad_common_no_adv = [ self.class_grad_factor * grad_class[i] \
                      for i in range(len(common_vars)) ]

      grad_common_only_adv = [ self.adv_grad_factor * grad_adv[i] \
                      for i in range(len(common_vars)) ]

    #   tf.print("We have to understand why it is not becoming blind")
    #   tf.print(grad_common)
    #   tf.print(grad_common_no_adv)
    #   tf.print(grad_common_only_adv)


      self.optimizer.apply_gradients(zip(grad_common + grad_class_excl, common_vars + class_vars))
      self.adv_optimizer.apply_gradients(zip(grad_adv_excl, adv_vars))


    return { m.name: m.result() for m in self.metrics }

  def train_step(self, data):
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
    input_folder = "DNN_dataset_2025-03-19-13-41-00"
    yaml_list = [fname for fname in os.listdir(input_folder) if fname.startswith('batch_config_parity')]

    modelname_parity = []

    for config_yaml in yaml_list:
        config_dict = {}
        with open(os.path.join(input_folder, config_yaml), 'r') as file:
            config_dict = yaml.safe_load(file)  

        batch_size = config_dict['meta_data']['batch_dict']['batch_size']

        input_file_name = os.path.join(input_folder, config_dict['meta_data']['input_filename'])
        input_weight_name = os.path.join(input_folder, f"weight{config_dict['meta_data']['input_filename'][5:]}")

        output_folder = "DNN_Blind_v11"
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

        dw.ReadFile(input_file_name)
        dw.ReadWeightFile(input_weight_name)
        print(config_dict)
        dw.DefineTrainTestSet(batch_size, 0.2)


        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        tf.random.set_seed(42)


        setup = {
            'learning_rate': 0.0001,
            'adv_learning_rate': 0.0001,
            'weight_decay': 0.004,
            'adv_weight_decay': 0.004,
            'adv_grad_factor': 1.0, #0.7
            'class_grad_factor': 0.1,
            'activation': 'tanh', #'relu'
            'use_batch_norm': False,
            'dropout': 0.0,
            'n_common_layers': 10,
            'n_common_units': 256,
            'n_adv_layers': 3,
            'n_adv_units': 128,
            'n_epochs': 10,
            'patience': 10,
            'mbb_CR_weight': dw.mbb_CR_weight,
            'mbb_SR_weight': dw.mbb_SR_weight,
            'class_weights_lut': dw.class_weights_lut,
        }


        model = AdversarialModel(setup)
        model.compile(loss=None,
                    optimizer=tf.keras.optimizers.AdamW(learning_rate=setup['learning_rate'],
                                                        weight_decay=setup['weight_decay']))

        model(dw.train_features)

        model.summary()



        def save_predicate(model, logs):
            return abs(logs['val_adv_accuracy'] - 0.5) < 0.01


        input_shape = [None, dw.train_features.shape[1]]
        input_signature = [tf.TensorSpec(input_shape, tf.double, name='x')]
        callbacks = [
            ModelCheckpoint(output_dnn_name, verbose=1, monitor="val_class_loss", mode='min', min_rel_delta=1e-3,
                            patience=setup['patience'], save_callback=None, predicate=save_predicate, input_signature=input_signature),
            tf.keras.callbacks.CSVLogger(f'{output_dnn_name}_training_log.csv', append=True),
        ]

        history = model.fit(
            dw.train_features,
            # (dw.train_labels, dw.train_mbb),
            (tf.one_hot(dw.train_labels_binary, 2), dw.train_mbb, dw.train_class_weight, dw.train_adv_weight),
            validation_data=(
                dw.test_features,
                # (dw.test_labels, dw.test_mbb)
                (tf.one_hot(dw.test_labels_binary, 2), dw.test_mbb, dw.test_class_weight, dw.test_adv_weight)
            ),
            verbose=1,
            batch_size=batch_size,
            epochs=setup['n_epochs'],
            shuffle=False,
            callbacks=callbacks,
        )
        

        model.save(f"{output_dnn_name}.keras")

        input_shape = [None, dw.train_features.shape[1]]
        input_signature = [tf.TensorSpec(input_shape, tf.double, name='x')]
        onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)
        onnx.save(onnx_model, f"{output_dnn_name}.onnx")






        print("Saved model")
        modelname_parity.append([model_name, config_dict['meta_data']['iterate_cut']])
        features_config = {
            'features': dw.feature_names,
            'listfeatures': dw.listfeature_names,
            'highlevelfeatures': dw.highlevelfeatures_names,
            'use_parametric': dw.use_parametric,
            'modelname_parity': modelname_parity
        }
        
        with open(os.path.join(dw.output_folder, 'dnn_config.yaml'), 'w') as file:
            yaml.dump(features_config, file)



        PlotMetric(history, model_name, "class_loss", output_folder)
        PlotMetric(history, model_name, "adv_loss", output_folder)



        # But model is what was the last epoch, not necessarily the best model (if early stopping triggered with patience)
        # Need to load the best model file


        model_load_name = os.path.join(output_dnn_name, 'best')
        print(f"Model load {model_load_name}")
        sess = ort.InferenceSession(model_load_name)

        print(f"Type sess {type(sess)}")
        print(f"Type model {type(model)}")
        validate = True
        if validate:
            # Cool! In the debugging stage, now lets also predict the output on batch1 file!
            other_parity_files = [os.path.join(input_folder, fname) for fname in os.listdir(input_folder) if fname.startswith('batchfile') and (fname != config_dict['meta_data']['input_filename'])]
            for i, other_parity_file in enumerate(other_parity_files):
                dw.ReadFile(other_parity_file)

                # dw.validate_output(model, model_name, i)
                dw.validate_output(sess, model_name, i)
                break # For now, don't validate on each nParity 

        return # And just end training after first model


        # An example of how to load the onnx version
        # load_model = False
        # if load_model:
        #     print("ORT result")
        #     sess = ort.InferenceSession(os.path.join(dw.output_folder, f"{model_name}.onnx"))
        #     res = sess.run(None, {'x': dw.features})
        #     print(res)
        







if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Create TrainTest Files for DNN.')
    #parser.add_argument('--config-folder', required=True, type=str, help="Config Folder from Step1")

    args = parser.parse_args()

    model = train_dnn()
