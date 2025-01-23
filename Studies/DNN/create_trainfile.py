import os, sys, gc
import psutil
from datetime import datetime
import yaml
from tqdm import tqdm
import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.EnableThreadSafety()

def create_file(config_yaml, out_filename):
    print(f"Starting create file. Memory usage in MB is {psutil.Process(os.getpid()).memory_info()[0] / float(2 ** 20)}")
    config_dict = {}
    with open(config_yaml, 'r') as file:
        config_dict = yaml.safe_load(file)
    nBatches = None
    print(config_dict.keys())
    for process in config_dict['processes']:
        if (nBatches is None) or ((process['nBatches'] <= nBatches) and (process['nBatches'] != 0)):
            nBatches = process['nBatches']

    #nBatches = 1
    print(f"Going to make {nBatches} batches")
    batch_size = config_dict['meta_data']['batch_dict']['batch_size']
    #batch_size = 100 #Super small batch just to test

    step_idx = 0

    #Get the name/type (And order!) of signal columns
    master_column_names = []
    master_column_types = []
    master_column_names_vec = ROOT.std.vector("string")()
    #Assume master(signal) is saved first and use idx==0 entry to fill

    for process in config_dict['processes']:
        process_filelist = [ f"{x}/*.root" for x in process['datasets'] ]

        tmp_filename = f'tmp{step_idx}.root'
        tmpnext_filename = f'tmp{step_idx+1}.root'

        print(process_filelist)
        #process_filelist = ["/eos/user/d/daebi/ANA_FOLDER_DEV/anaTuples/Run3_2022EE_11Dec24/Run3_2022EE/TTto2L2Nu/nano_0.root"]
        #process_filelist = ["/afs/cern.ch/work/d/daebi/diHiggs/HH_bbWW/Studies/DNN/DNN_dataset_2025-01-07-16-54-05/GluGlutoRadiontoHHto2B2Vto2B2L2Nu_Combined/GluGlutoRadiontoHHto2B2Vto2B2L2Nu_Combined.root"]
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

        tuple_maker = ROOT.analysis.TupleMaker(*local_column_types)(queue_size, max_entries) #For debug, lets fill whole thing with ttbar

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

    print("Finished create file, will copy tmp file to final output")

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
    for i, yamlname in enumerate([fname for fname in os.listdir(output_folder) if ((".yaml" in fname) and ("batch_config_parity" in fname))]):
        print(f"Starting batch {i}")
        create_file(os.path.join(output_folder, yamlname), os.path.join(output_folder, f"batchfile{i}.root"))
