import os
import uproot
import numpy as np
import psutil
from datetime import datetime
import yaml
import awkward as ak
from tqdm import tqdm

def variable_dict(batch_size, process_subdict, process_iter, output_array, mass_values):
    vardict = {}
    write_array = np.array(output_array)
    nBatchesThisChunk = len(write_array['lep1_pt'])//batch_size
    if len(write_array['lep1_pt'])%batch_size != 0: print("UH OH")
    if nBatchesThisChunk == 0: return vardict

    class_value = process_subdict['class_value']
    mass_value = process_subdict['mass_value']

    for nBatch in range(nBatchesThisChunk):
        process_array = next(process_iter)
        index_start = process_subdict['batch_start'] + (batch_size*nBatch)
        index_end = index_start+process_subdict['batch_size']

        write_array['class_value'][index_start:index_end] = np.full(index_end-index_start, class_value)
        if mass_value <= 0:
            write_array['mass_value'][index_start:index_end] = np.random.choice(mass_values, size=index_end-index_start).astype(float)
        else:
            write_array['mass_value'][index_start:index_end] = np.full(index_end-index_start, mass_value)


        write_array['lep1_pt'][index_start:index_end] =     np.asarray(process_array['lep1_pt'])
        write_array['lep1_eta'][index_start:index_end] =    np.asarray(process_array['lep1_eta'])
        write_array['lep1_phi'][index_start:index_end] =    np.asarray(process_array['lep1_phi'])
        write_array['lep1_mass'][index_start:index_end] =   np.asarray(process_array['lep1_mass'])

        write_array['lep2_pt'][index_start:index_end] =     np.asarray(process_array['lep2_pt'])
        write_array['lep2_eta'][index_start:index_end] =    np.asarray(process_array['lep2_eta'])
        write_array['lep2_phi'][index_start:index_end] =    np.asarray(process_array['lep2_phi'])
        write_array['lep2_mass'][index_start:index_end] =   np.asarray(process_array['lep2_mass'])

    vardict = {
        'class_value':  write_array['class_value'],
        'mass_value':   write_array['mass_value'],

        'lep1_pt':      write_array['lep1_pt'],
        'lep1_eta':     write_array['lep1_eta'],
        'lep1_phi':     write_array['lep1_phi'],
        'lep1_mass':    write_array['lep1_mass'],

        'lep2_pt':      write_array['lep2_pt'],
        'lep2_eta':     write_array['lep2_eta'],
        'lep2_phi':     write_array['lep2_phi'],
        'lep2_mass':    write_array['lep2_mass'],
    }
    return vardict


def init_empty_vardict(batch_size):
    empty_dict = {
        'class_value':  np.zeros(batch_size),
        'mass_value':   np.zeros(batch_size),

        'lep1_pt':      np.zeros(batch_size),
        'lep1_eta':     np.zeros(batch_size),
        'lep1_phi':     np.zeros(batch_size),
        'lep1_mass':    np.zeros(batch_size),

        'lep2_pt':      np.zeros(batch_size),
        'lep2_eta':     np.zeros(batch_size),
        'lep2_phi':     np.zeros(batch_size),
        'lep2_mass':    np.zeros(batch_size),
    }
    return empty_dict




def iterate_uproot(fnames, batch_size, nBatchesPerChunk, selection_branches, selection_cut):
    for iter in uproot.iterate(fnames, step_size=nBatchesPerChunk*batch_size):
        p = 0
        while p < nBatchesPerChunk*batch_size:
            if p+batch_size > len(iter):
                print("Bad new bears, end of file")
                print(fnames)
                print(p+batch_size)
                print(len(iter))
                break
            else:
                tree = iter.arrays(selection_branches)
                filtered = eval(selection_cut)
                print("here dummy")
                print(filtered)
                out_array = iter[p:p+batch_size]
                p+= batch_size
                yield out_array


def create_signal_files(config_dict, output_folder):
    storage_folder = config_dict['storage_folder']

    for signal_name in config_dict['signal']:
        signal_dict = config_dict['signal'][signal_name]
        mass_points = signal_dict['mass_points']
        dataset_name_format = signal_dict['dataset_name_format']
        use_combined = signal_dict['use_combined']

        if use_combined:
            combined_name = signal_dict['combined_name']

            out_file = uproot.recreate(f"{os.path.join(output_folder, combined_name)}.root")

            new_array = {}
            nEvents = 0

            print(f"Starting to merge signal {signal_name} files")

            for mass_point in tqdm(mass_points):
                dataset_name = dataset_name_format.format(mass_point)
                extension_list = [ fn for fn in os.listdir(storage_folder) if fn.startswith(f"{dataset_name}_ext") ]

                for ext_name in ([dataset_name] + extension_list):
                    process_dir = os.path.join(storage_folder, ext_name)
                    for nano_file in os.listdir(process_dir):
                        with uproot.open(f"{os.path.join(process_dir, nano_file)}:Events") as h:
                            tree = h.arrays()
                            nEvents += h.num_entries

                            keys = tree.fields
                            for key in keys:
                                if key not in new_array.keys():
                                    new_array[key] = tree[key]
                                else:
                                    new_array[key] = ak.concatenate([new_array[key], tree[key]])


            #Shuffle the signal data
            index = np.arange(nEvents)
            np.random.shuffle(index)
            for key in new_array.keys():
                new_array[key] = new_array[key][index]

            out_file['Events'] = new_array
            out_file.close()



def create_dict(config_dict, output_folder):
    batch_dict = config_dict['batch_dict']
    storage_folder = config_dict['storage_folder']
    selection_branches = config_dict['selection_branches']
    selection_cut = config_dict['selection_cut']

    if 'batch_size' not in batch_dict.keys():
        batch_dict['batch_size'] = 0
    process_dict = {}
    for key in batch_dict.keys():
        if key == 'batch_size': continue
        batch_dict['batch_size'] += batch_dict[key]
        process_dict[key] = {}

    for nParity in range(config_dict['nParity']):
        nParity_Cut = config_dict['parity_func'].format(nParity = config_dict['nParity'], parity_scan = nParity)
        total_cut = f"{selection_cut} & {nParity_Cut}"

        out_yaml = f"batch_config_parity{nParity}.yaml"

        print("Looping over signals in config")
        for signal_name in tqdm(config_dict['signal']):
            signal_dict = config_dict['signal'][signal_name]
            class_value = signal_dict['class_value']
            mass_points = signal_dict['mass_points']
            dataset_name_format = signal_dict['dataset_name_format']
            use_combined = signal_dict['use_combined']


            #If a combined file exists, lets use that
            #if f"{signal_dict['combined_name']}.root" in os.listdir(output_folder):
            if use_combined:
                dataset_name = signal_dict['combined_name']

                process_dict[signal_name][dataset_name] = {
                    'total': 0,
                    'total_cut': 0,
                    'weight_cut': 0,
                    'nBatches': 0,
                    'batch_size': 0,
                    'batch_start': 0,
                    'class_value': class_value,
                    'spin': 0,
                    'mass': -1,
                    'all_extensions': [],
                    'storage_folder': os.path.join(os.getcwd(), output_folder)
                }
                    
                process_dict[signal_name][dataset_name]['all_extensions'] = [dataset_name]

                with uproot.open(f"{os.path.join(output_folder, dataset_name)}.root:Events") as h:
                    tree = h.arrays(selection_branches)
                    process_dict[signal_name][dataset_name]['total'] += int(h.num_entries)
                    process_dict[signal_name][dataset_name]['total_cut'] += int(np.sum(eval(total_cut)))
                    eval_string = f"float(np.sum(tree[{total_cut}].weight_MC_Lumi_pu))"
                    process_dict[signal_name][dataset_name]['weight_cut'] += eval(eval_string)



            for mass_point in mass_points:
                dataset_name = dataset_name_format.format(mass_point)

                process_dict[signal_name][dataset_name] = {
                    'total': 0,
                    'total_cut': 0,
                    'weight_cut': 0,
                    'nBatches': 0,
                    'batch_size': 0,
                    'batch_start': 0,
                    'class_value': class_value,
                    'spin': 0,
                    'mass': mass_point,
                    'all_extensions': [],
                    'storage_folder': storage_folder,
                }

                extension_list = [ fn for fn in os.listdir(storage_folder) if fn.startswith(f"{dataset_name}_ext") ]
                process_dict[signal_name][dataset_name]['all_extensions'] = [dataset_name] + extension_list

                for ext_name in process_dict[signal_name][dataset_name]['all_extensions']:
                    process_dir = os.path.join(storage_folder, ext_name)
                    for nano_file in os.listdir(process_dir):
                        with uproot.open(f"{os.path.join(process_dir, nano_file)}:Events") as h:
                            tree = h.arrays(selection_branches)
                            process_dict[signal_name][dataset_name]['total'] += int(h.num_entries)
                            process_dict[signal_name][dataset_name]['total_cut'] += int(np.sum(eval(total_cut)))
                            eval_string = f"float(np.sum(tree[{total_cut}].weight_MC_Lumi_pu))"
                            process_dict[signal_name][dataset_name]['weight_cut'] += eval(eval_string)



        print("Looping over backgrounds in config")
        for background_name in tqdm(config_dict['background']):
            background_dict = config_dict['background'][background_name]
            class_value = background_dict['class_value']
            dataset_names = background_dict['background_datasets']

            for dataset_name in dataset_names:
                process_dict[background_name][dataset_name] = {
                    'total': 0,
                    'total_cut': 0,
                    'weight_cut': 0,
                    'nBatches': 0,
                    'batch_size': 0,
                    'batch_start': 0,
                    'class_value': class_value,
                    'all_extensions': [],
                    'storage_folder': storage_folder 
                }

                extension_list = [ fn for fn in os.listdir(storage_folder) if fn.startswith(f"{dataset_name}_ext") ]
                process_dict[background_name][dataset_name]['all_extensions'] = [dataset_name] + extension_list

                for ext_name in process_dict[background_name][dataset_name]['all_extensions']:
                    process_dir = os.path.join(storage_folder, ext_name)
                    for nano_file in os.listdir(process_dir):
                        with uproot.open(f"{os.path.join(process_dir, nano_file)}:Events") as h:
                            tree = h.arrays(selection_branches)
                            process_dict[background_name][dataset_name]['total'] += int(h.num_entries)
                            process_dict[background_name][dataset_name]['total_cut'] += int(np.sum(eval(total_cut)))
                            eval_string = f"float(np.sum(tree[{total_cut}].weight_MC_Lumi_pu))"
                            process_dict[background_name][dataset_name]['weight_cut'] += eval(eval_string)







        #Add totals to start the spin/mass dist and remove the individual signal files
        signal_names = config_dict['signal'].keys()
        for process in process_dict:
            process_dict[process]['total'] = 0
            process_dict[process]['weight'] = 0
            use_combined = False
            if process in signal_names:
                use_combined = config_dict['signal'][process]['use_combined']
            for subprocess in process_dict[process].keys():
                if subprocess.startswith('total') or subprocess.startswith('weight'): continue
                if use_combined:
                    if subprocess == config_dict['signal'][process]['combined_name']: continue
                process_dict[process]['total'] += process_dict[process][subprocess]['total_cut']
                process_dict[process]['weight'] += process_dict[process][subprocess]['weight_cut']


        #Calculate the random spin/mass distribution for backgrounds to be assigned during parametric DNN
        spin_mass_dist = {}

        total_signal = 0
        for signal_name in config_dict['signal']:
            total_signal += process_dict[signal_name]['total']
        for signal_name in config_dict['signal']:
            keys_to_remove = [] #Keys we want to remove if the combined option is being used
            use_combined = config_dict['signal'][signal_name]['use_combined']
            for subprocess in process_dict[signal_name]:
                if subprocess.startswith('total') or subprocess.startswith('weight'): continue
                if use_combined:
                    if subprocess == config_dict['signal'][signal_name]['combined_name']: continue
                subprocess_dict = process_dict[signal_name][subprocess]
                if f"{subprocess_dict['spin']}" not in spin_mass_dist.keys():
                    spin_mass_dist[f"{subprocess_dict['spin']}"] = {}
                spin_mass_dist[f"{subprocess_dict['spin']}"][f"{subprocess_dict['mass']}"] = subprocess_dict['total_cut']/total_signal
                if use_combined:
                    keys_to_remove.append(subprocess)
            #Remove unneeded keys since we will use combined anyway
            for key in keys_to_remove:
                del process_dict[signal_name][key]




        for process in process_dict:
            batch_size_sum = 0
            for subprocess in process_dict[process]:
                if subprocess.startswith('total') or subprocess.startswith('weight'): continue
                process_dict[process][subprocess]['batch_size'] = int(batch_dict[process] * process_dict[process][subprocess]['weight_cut'] / process_dict[process]['weight'])
                nBatches = 0
                if process_dict[process][subprocess]['batch_size'] != 0:
                    nBatches = int(process_dict[process][subprocess]['total_cut']/process_dict[process][subprocess]['batch_size'])
                process_dict[process][subprocess]['nBatches'] = nBatches
                batch_size_sum += process_dict[process][subprocess]['batch_size']

            print(f"Process {process} has batch size sum {batch_size_sum}")
            while batch_size_sum != batch_dict[process]:
                print(f"Warning this is bad batch size, size={batch_size_sum} where goal is {batch_dict[process]}")
                max_batches_subprocess = ""
                max_batches_val = 0
                for subprocess in process_dict[process].keys():
                    if subprocess.startswith('total') or subprocess.startswith('weight'): continue
                    if process_dict[process][subprocess]['nBatches'] > max_batches_val:
                        max_batches_val = process_dict[process][subprocess]['nBatches']
                        max_batches_subprocess = subprocess

                print(f"Trying to fix, incrementing {max_batches_subprocess} batch size {process_dict[process][max_batches_subprocess]['batch_size']} by 1")
                process_dict[process][max_batches_subprocess]['batch_size'] += 1
                print(f"nBatches went from {process_dict[process][max_batches_subprocess]['nBatches']}")
                process_dict[process][max_batches_subprocess]['nBatches'] = int(process_dict[process][max_batches_subprocess]['total']/process_dict[process][max_batches_subprocess]['batch_size'])
                print(f"To {process_dict[process][max_batches_subprocess]['nBatches']}")
                batch_size_sum += 1

        current_index = 0
        for process in process_dict.keys():
            for subprocess in process_dict[process].keys():
                if subprocess.startswith('total') or subprocess.startswith('weight'): continue
                process_dict[process][subprocess]['batch_start'] = current_index
                current_index += process_dict[process][subprocess]['batch_size']


        nBatches = 1e100
        for process in process_dict.keys():
            for subprocess in process_dict[process].keys():
                if subprocess.startswith('total') or subprocess.startswith('weight'): continue
                if process_dict[process][subprocess]['nBatches'] < nBatches and (process_dict[process][subprocess]['nBatches'] != 0):
                    nBatches = process_dict[process][subprocess]['nBatches']




        print(f"Creating {nBatches} batches, according to distribution. ")
        print(process_dict)
        print(f"And total batch size is {batch_dict['batch_size']}")


        machine_yaml = {
            'meta_data': {},
            'processes': [],
        }

        machine_yaml['meta_data']['storage_folder'] = storage_folder
        machine_yaml['meta_data']['batch_dict'] = batch_dict
        machine_yaml['meta_data']['selection_branches'] = selection_branches
        machine_yaml['meta_data']['selection_cut'] = total_cut

        machine_yaml['meta_data']['spin_mass_dist'] = spin_mass_dist #Dict of spin/mass distribution values for random choice parametric


        for process in process_dict:
            for subprocess in process_dict[process]:
                if subprocess.startswith('total') or subprocess.startswith('weight'): continue                
                print("Using subprocess ", subprocess)
                subprocess_dict = process_dict[process][subprocess]
                datasets_full_pathway = [ os.path.join(subprocess_dict['storage_folder'], fn) for fn in subprocess_dict['all_extensions'] ]
                tmp_process_dict = {
                    'datasets': datasets_full_pathway,
                    'class_value': subprocess_dict['class_value'],
                    'batch_start': subprocess_dict['batch_start'],
                    'batch_size': subprocess_dict['batch_size'],
                    'nBatches': subprocess_dict['nBatches'],
                }
                machine_yaml['processes'].append(tmp_process_dict)


        with open(os.path.join(output_folder, out_yaml), 'w') as outfile:
            yaml.dump(machine_yaml, outfile)




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Create TrainTest Files for DNN.')
    parser.add_argument('--config', required=True, type=str, help="Config YAML")

    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config_dict = yaml.safe_load(file)

    output_folder = f"DNN_dataset_{datetime.today().strftime('%Y-%m-%d-%H-%M-%S')}"
    os.makedirs(output_folder, exist_ok=True)
    os.system(f"cp {args.config} {output_folder}/.")

    create_signal_files(config_dict, output_folder)
    create_dict(config_dict, output_folder)
    #create_file(out_file)