import os, sys, gc
import psutil
from datetime import datetime
import yaml
from tqdm import tqdm
import ROOT
import FLAF.Common.Utilities as Utilities
import Analysis.hh_bbww as analysis

ROOT.gROOT.SetBatch(True)
ROOT.EnableThreadSafety()

sys.path.append(os.environ['ANALYSIS_PATH'])
ana_path = os.environ['ANALYSIS_PATH']

header_path_AnalysisTools = "FLAF/include/AnalysisTools.h"
ROOT.gInterpreter.Declare(f'#include "{os.path.join(ana_path,header_path_AnalysisTools)}"')
header_path_AnalysisMath = "FLAF/include/AnalysisMath.h"
ROOT.gInterpreter.Declare(f'#include "{os.path.join(ana_path,header_path_AnalysisMath)}"')
header_path_MT2 = "FLAF/include/MT2.h"
ROOT.gInterpreter.Declare(f'#include "{os.path.join(ana_path,header_path_MT2)}"')
header_path_Lester_mt2_bisect = "FLAF/include/Lester_mt2_bisect.cpp"
ROOT.gInterpreter.Declare(f'#include "{os.path.join(ana_path,header_path_Lester_mt2_bisect)}"')
lep1_p4 = "ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>(lep1_pt,lep1_eta,lep1_phi,lep1_mass)"
lep2_p4 = "ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>(lep2_pt,lep2_eta,lep2_phi,lep2_mass)"
b1_p4 = "ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>(centralJet_pt[0],centralJet_eta[0],centralJet_phi[0],centralJet_mass[0])"
b2_p4 = "ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>(centralJet_pt[1],centralJet_eta[1],centralJet_phi[1],centralJet_mass[1])"
MET_p4 = "ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>(met_pt, 0., met_phi, 0.)"
initialized = True



def create_file(config_dict, output_folder, out_filename):
    print(f"Starting create file. Memory usage in MB is {psutil.Process(os.getpid()).memory_info()[0] / float(2 ** 20)}")
    nBatches = None
    print(config_dict.keys())
    for process in config_dict['processes']:
        if (nBatches is None) or ((process['nBatches'] <= nBatches) and (process['nBatches'] != 0)):
            nBatches = process['nBatches']

    print(f"Going to make {nBatches} batches")
    batch_size = config_dict['meta_data']['batch_dict']['batch_size']

    step_idx = 0

    #Get the name/type (And order!) of signal columns
    master_column_names = []
    master_column_types = []
    master_column_names_vec = ROOT.std.vector("string")()
    #Assume master(signal) is saved first and use idx==0 entry to fill

    for process in config_dict['processes']:
        process_filelist = [ f"{x}/*.root" for x in process['datasets'] ]

        tmp_filename = os.path.join(output_folder, f'tmp{step_idx}.root')
        tmpnext_filename = os.path.join(output_folder, f'tmp{step_idx+1}.root')

        print(process_filelist)
        df_in = ROOT.RDataFrame('Events', process_filelist)

        #Filter for nLeps and Parity (iterate cut in config)
        df_in = df_in.Filter(config_dict['meta_data']['iterate_cut'])

        nEntriesPerBatch = process['batch_size']
        nBatchStart = process['batch_start']
        nBatchEnd = nBatchStart+nEntriesPerBatch

        #Load df_out, if first iter then load an empty, otherwise load the past file
        if step_idx == 0:
            df_out = ROOT.RDataFrame(nBatches*batch_size)
            df_out = df_out.Define("is_valid", 'false')
            #Fill master column nametype
            master_column_names = df_in.GetColumnNames()
            master_column_types = [str(df_in.GetColumnType(str(c))) for c in master_column_names]
            for name in master_column_names:
                master_column_names_vec.push_back(name)
        else:
            df_out = ROOT.RDataFrame('Events', tmp_filename)


        local_column_names = df_in.GetColumnNames()
        local_column_types = [str(df_in.GetColumnType(str(c))) for c in local_column_names]
        local_column_names_vec = ROOT.std.vector("string")()
        for name in local_column_names:
            local_column_names_vec.push_back(name)


        #Need a local_to_master_map so that local columns keep the same index as the master columns
        local_to_master_map = [list(master_column_names).index(local_name) for local_name in local_column_names]
        master_size = len(master_column_names)

        queue_size = 10
        max_entries = nEntriesPerBatch*nBatches

        tuple_maker = ROOT.analysis.TupleMaker(*local_column_types)(queue_size, max_entries)

        df_out = tuple_maker.FillDF(ROOT.RDF.AsRNode(df_out), ROOT.RDF.AsRNode(df_in), local_to_master_map, master_size, local_column_names_vec, nBatchStart, nBatchEnd, batch_size)

        for column_idx, column_name in enumerate(master_column_names):
            column_type = master_column_types[column_idx]

            if step_idx == 0:
                df_out = df_out.Define(str(column_name), f'_entry ? _entry->GetValue<{column_type}>({column_idx}) : {column_type}() ')
            else:
                if column_name not in local_column_names: continue
                df_out = df_out.Redefine(str(column_name), f'_entry ? _entry->GetValue<{column_type}>({column_idx}) : {column_name} ')


        df_out = df_out.Redefine('is_valid', '(is_valid) || (_entry)')


        snapshotOptions = ROOT.RDF.RSnapshotOptions()
        #snapshotOptions.fOverwriteIfExists=False
        #snapshotOptions.fLazy=True
        snapshotOptions.fMode="RECREATE"
        snapshotOptions.fCompressionAlgorithm = getattr(ROOT.ROOT, 'k' + 'ZLIB')
        snapshotOptions.fCompressionLevel = 4
        ROOT.RDF.Experimental.AddProgressBar(df_out)
        print("Going to snapshot")
        save_column_names = ROOT.std.vector("string")(master_column_names)
        save_column_names.push_back('is_valid')
        df_out.Snapshot('Events', tmpnext_filename, save_column_names, snapshotOptions)

        tuple_maker.join()

        step_idx += 1



    print("Finished create file loop, now we must add the DNN variables")
    # Increment the name indexes before I embarass myself again
    tmp_filename = os.path.join(output_folder, f'tmp{step_idx}.root')
    tmpnext_filename = os.path.join(output_folder, f'tmp{step_idx+1}.root')

    df_out = ROOT.RDataFrame('Events', tmp_filename)


    snapshotOptions = ROOT.RDF.RSnapshotOptions()
    #snapshotOptions.fOverwriteIfExists=False
    #snapshotOptions.fLazy=True
    snapshotOptions.fMode="RECREATE"
    snapshotOptions.fCompressionAlgorithm = getattr(ROOT.ROOT, 'k' + 'ZLIB')
    snapshotOptions.fCompressionLevel = 4
    ROOT.RDF.Experimental.AddProgressBar(df_out)
    print("Going to snapshot")
    # Only need to save the prexisting columns plus the new DNN variables
    save_column_names = ROOT.std.vector("string")(df_out.GetColumnNames())
    df_out = analysis.AddDNNVariables(df_out)
    highlevel_names = [
        'HT', 'dR_dilep', 'dR_dibjet', 
        'dR_dilep_dijet', 'dR_dilep_dibjet',
        'dPhi_lep1_lep2', 'dPhi_jet1_jet2',
        'dPhi_MET_dilep', 'dPhi_MET_dibjet',
        'min_dR_lep0_jets', 'min_dR_lep1_jets',
        'MT', 'MT2', 
        'MT2_ll', 'MT2_bb', 'MT2_blbl',
        'CosTheta_bb',
        'll_mass',
        'bb_mass', 'bb_mass_PNetRegPtRawCorr', 'bb_mass_PNetRegPtRawCorr_PNetRegPtRawCorrNeutrino'
    ]
    for highlevel_name in highlevel_names:
        save_column_names.push_back(highlevel_name)
    df_out.Snapshot('Events', tmpnext_filename, save_column_names, snapshotOptions)


    print(f"Finished create file, will copy tmp file to final output {out_filename}")

    os.system(f"cp {tmpnext_filename} {out_filename}")




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Create TrainTest Files for DNN.')
    parser.add_argument('--config-folder', required=True, type=str, help="Config Folder from Step1")

    args = parser.parse_args()


    headers_dir = os.path.dirname(os.path.abspath(__file__))
    #headers = [ 'AnalysisTools.h', 'TupleMaker.h' ] #Order here matters since TupleMaker requires AnalysisTools
    headers = [ 'TupleMaker.h' ] #Order here matters since TupleMaker requires AnalysisTools
    for header in headers:
        header_path = os.path.join(headers_dir, header)
        if not ROOT.gInterpreter.Declare(f'#include "{header_path}"'):
            raise RuntimeError(f'Failed to load {header_path}')


    output_folder = args.config_folder
    print(f"Starting the create file loop. Memory usage in MB is {psutil.Process(os.getpid()).memory_info()[0] / float(2 ** 20)}")
    yaml_list = [fname for fname in os.listdir(output_folder) if ((".yaml" in fname) and ("batch_config_parity" in fname))]
    yaml_list.sort()
    for i, yamlname in enumerate(yaml_list):
        print(f"Starting batch {i} with yaml {yamlname}")
        config_dict = {}
        with open(os.path.join(output_folder, yamlname), 'r') as file:
            config_dict = yaml.safe_load(file)            
        create_file(config_dict, output_folder, os.path.join(output_folder, config_dict['meta_data']['input_filename']))
