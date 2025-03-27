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



def Scheduler(epoch, lr):
    if epoch < 20:
        return lr
    else:
        if epoch % 2 == 0:
            return 0.9*lr
        return lr





#Artem's custom loss, will need to create a loss to uncorrelate bkg estimation vars (H_bb for Muhammad and DY?)
def CustomLoss(y_true, y_pred):
    # base_loss = tf.keras.losses.log_cosh(y_true, y_pred)
    base_loss = tf.keras.losses.mse(y_true, y_pred)

    hbb_p3 = y_pred[:, 0:3]
    hbb_E = y_pred[:, 3]
    mh_bb_sqr = hbb_E*hbb_E - tf.reduce_sum(hbb_p3**2, axis=1, keepdims=True)

    hww_p3 = y_pred[:, 4:7]
    hww_E = y_pred[:, 7]
    mh_ww_sqr = hww_E*hww_E - tf.reduce_sum(hww_p3**2, axis=1, keepdims=True)

    mh_bb = tf.sign(mh_bb_sqr)*tf.sqrt(tf.abs(mh_bb_sqr))
    mh_ww = tf.sign(mh_ww_sqr)*tf.sqrt(tf.abs(mh_ww_sqr))

    # custom = 0.5*tf.keras.losses.log_cosh(mh_bb, mh) + 0.5*tf.keras.losses.log_cosh(mh_ww, mh)
    custom_loss = tf.reduce_mean(0.5*(mh_bb - mh)**2 + 0.5*(mh_ww - mh)**2)
    return base_loss + 0.1*custom_loss
    # return custom_loss


def PlotMetric(history, model, metric):
    plt.plot(history.history[metric], label=f'train_{metric}')
    plt.plot(history.history[f'val_{metric}'], label=f'val_{metric}')
    plt.title(f'{model} {metric}')
    plt.ylabel(metric)
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig(f"{metric}_{model}.pdf", bbox_inches='tight')
    plt.clf()


#Need to get train_features and train_labels
class DataWrapper():
    def __init__(self):
        print("Init data wrapper")
        self.value_to_label = {'1': 0, '8': 1, '5': 2}
        self.label_names = ["Signal", "TT", "DY"]

        self.feature_names = None
        self.listfeature_names = None
        self.highlevelfeatures_names = None
        self.label_name = None

        self.features_no_param = None
        self.features = None
        self.listfeatures = None
        self.hlv = None
        self.param_values = None
        self.labels = None

        self.train_features = None
        self.train_labels = None

        self.test_features = None
        self.test_labels = None

        self.param_list = [250, 260, 270, 280, 300, 350, 450, 550, 600, 650, 700, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000, 4000, 5000 ]
        self.use_parametric = False

        self.features_paramSet = None

        self.output_folder = ""

    def SetOutputFolder(self, foldername):
        self.output_folder = foldername
        os.makedirs(foldername, exist_ok = True)

    def UseParametric(self, use_parametric):
        self.use_parametric = use_parametric
        print(f"Parametric feature set to {use_parametric}")

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

    def AddHighLevelFeature(self, features):
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
        print("Got labels")

        #Add parametric variable
        self.param_values = np.array([[x if (x > 0) else np.random.choice(self.param_list) for x in getattr(branches, 'X_mass') ]]).transpose()
        print("Got the param values")


        self.features_no_param = self.features
        if self.use_parametric: self.features = np.append(self.features, self.param_values, axis=1)



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

        self.test_features = self.features[testStart:testEnd]
        self.test_labels = self.labels[testStart:testEnd]


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

        pred = model.predict(arglist)
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






    def validate_output(self, model):
        plotbins = 100
        plotrange = (0.0, 1.0)
        #After discussion with Konstanin, parametric dnn application should put in mass values for the point we want
        #So we will have a DNN output for masspoint 300, 400, 500, etc
        #For now, lets create a parametric_masspoints list and apply with those masses
        #Theory is that these masses will work very well for signal at that mass, but still remove backgrounds


        #X points for the acceptance scan and significance scan
        x = np.linspace(0.0, 1.0, 101)
        all_significances = []
        all_acceptances = []

        for para_masspoint in self.param_list:
            print(f"Looking at mass {para_masspoint}")
            if para_masspoint > 1000: continue
            if para_masspoint not in [300, 450, 550, 800]: continue
            predict_list = []
            weight_list = []

            for i, label_name in enumerate(self.label_names):

                features_thislabel = self.features[self.labels == i]

                pred_thislabel = model.predict(features_thislabel)

                if i == 0:
                    #This is signal, lets plot the masses separate
                    for real_mass in self.param_list:
                        if abs(real_mass - para_masspoint) > 200: 
                            continue
                        plotlabel = f'{label_name} M{real_mass}'
                        this_mass_mask = self.param_values[self.labels == i] == real_mass
                        this_mass_mask = this_mass_mask.flatten()
                        print(this_mass_mask)
                        pred_thismass = pred_thislabel[this_mass_mask]
                        plt.hist(pred_thismass[:,0], bins=plotbins, range=plotrange, density=True, histtype='step', label=plotlabel, alpha=0.5)


                else:
                    plotlabel = label_name
                    plt.hist(pred_thislabel[:,0], bins=plotbins, range=plotrange, density=True, histtype='step', label=plotlabel, alpha=0.5)



            plt.title(f'DNN Output: PredictSignal M{para_masspoint}')
            plt.legend(loc='upper right', fontsize="4")
            plt.yscale('log')
            plt.savefig(os.path.join(self.output_folder, f'dnn_values_M{para_masspoint}.pdf'))
            plt.clf()
            plt.close()



def train_dnn():
    input_file_name = "DNN_dataset_2025-02-22-15-17-42/batchfile0.root"


    dw = DataWrapper()
    dw.AddInputFeatures(['lep1_pt', 'lep1_phi', 'lep1_eta', 'lep1_mass'])
    dw.AddInputFeatures(['lep2_pt', 'lep2_phi', 'lep2_eta', 'lep2_mass'])
    dw.AddInputFeatures(['met_pt', 'met_phi'])
    dw.AddInputFeaturesList(['centralJet_pt', 'centralJet_phi', 'centralJet_eta', 'centralJet_mass'], 0)
    dw.AddInputFeaturesList(['centralJet_pt', 'centralJet_phi', 'centralJet_eta', 'centralJet_mass'], 1)
    dw.AddHighLevelFeature([
                            'HT', 'dR_dilep', 'dR_dibjet', 
                            'dR_dilep_dijet', #'dR_dilep_dibjet',
                            'dPhi_MET_dilep', 'dPhi_MET_dibjet',
                            'min_dR_lep0_jets', 'min_dR_lep1_jets',
                            'MT', 'MT2_bbWW'
                            #'mt2_ll_lester', 'mt2_bb_lester', 'mt2_blbl_lester'
                            ])



    dw.UseParametric(True)
    dw.SetOutputFolder("hlv_from_analysis_MT2Fixed")

    dw.AddInputLabel('sample_type')
    dw.ReadFile(input_file_name)
    dw.DefineTrainTestSet(900, 0.1)


    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.random.set_seed(42)

    model_name = "ResHH_Classifier"
    cpkt_path = f"{model_name}_cpkt.keras"

    schedule_callback = tf.keras.callbacks.LearningRateScheduler(Scheduler)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(dw.output_folder, cpkt_path), 
        monitor='val_loss', 
        verbose=0, 
        save_best_only=False, 
        mode='min'
    )


    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(256, activation='relu', kernel_initializer='random_normal', bias_initializer='random_normal'))
    # model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(256, activation='relu', kernel_initializer='random_normal', bias_initializer='random_normal'))
    # model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(256, activation='relu', kernel_initializer='random_normal', bias_initializer='random_normal'))
    # model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(256, activation='relu', kernel_initializer='random_normal', bias_initializer='random_normal'))
    # model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(256, activation='relu', kernel_initializer='random_normal', bias_initializer='random_normal'))
    # model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(256, activation='relu', kernel_initializer='random_normal', bias_initializer='random_normal'))
    # model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(256, activation='relu', kernel_initializer='random_normal', bias_initializer='random_normal'))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(256, activation='relu', kernel_initializer='random_normal', bias_initializer='random_normal'))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(256, activation='relu', kernel_initializer='random_normal', bias_initializer='random_normal'))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(256, activation='relu', kernel_initializer='random_normal', bias_initializer='random_normal'))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(256, activation='relu', kernel_initializer='random_normal', bias_initializer='random_normal'))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
        optimizer=tf.keras.optimizers.Adam(3e-4),
        metrics=[
            tf.keras.metrics.SparseCategoricalCrossentropy(),
            tf.keras.metrics.SparseCategoricalAccuracy(),
        ]                  
    )
    # model.compile(loss=CustomLoss, 
    #               optimizer=tf.keras.optimizers.Adam(3e-4))

    model.build(dw.train_features.shape)

    print("Going to fit the model! We have features")
    print(dw.train_features)
    print(dw.train_features.shape)

    history = None
    history = model.fit(
        dw.train_features,
        dw.train_labels,
        validation_data=(
            dw.test_features,
            dw.test_labels
        ),
        verbose=1,
        batch_size=900,
        epochs=10,
        callbacks=[schedule_callback, checkpoint_callback]
    )

    model.save(os.path.join(dw.output_folder, f"{model_name}.keras"))

    print("Saved model, now save features")
    features_config = {
        'features': dw.feature_names,
        'listfeatures': dw.listfeature_names,
        'highlevelfeatures': dw.highlevelfeatures_names,
        'use_parametric': dw.use_parametric
    }
     
    with open(os.path.join(dw.output_folder, 'dnn_config.yaml'), 'w', ) as file:
        yaml.dump(features_config, file)


    PlotMetric(history, model_name, "loss")




    # Cool! In the debugging stage, now lets also predict the output on batch1 file!
    input_file_name = "DNN_dataset_2025-02-22-15-17-42/batchfile1.root"
    dw.ReadFile(input_file_name)

    dw.monitor_param(model, 300)
    dw.monitor_param(model, 700)

    dw.validate_output(model)






if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Create TrainTest Files for DNN.')
    #parser.add_argument('--config-folder', required=True, type=str, help="Config Folder from Step1")

    args = parser.parse_args()

    train_dnn()
