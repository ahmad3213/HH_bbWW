import law
import os
import yaml
import contextlib
import luigi
import threading


from FLAF.RunKit.run_tools import ps_call
from FLAF.RunKit.crabLaw import cond as kInit_cond, update_kinit_thread
from FLAF.run_tools.law_customizations import Task, HTCondorWorkflow, copy_param,get_param_value
from FLAF.AnaProd.tasks import AnaTupleTask, DataMergeTask, AnaCacheTupleTask, DataCacheMergeTask, AnaCacheTask

unc_cfg_dict = None
def load_unc_config(unc_cfg):
    global unc_cfg_dict
    with open(unc_cfg, 'r') as f:
        unc_cfg_dict = yaml.safe_load(f)
    return unc_cfg_dict


def getYear(period):
    year_dict = {
        'Run2_2016_HIPM':'2016_HIPM',
        'Run2_2016':'2016',
        'Run2_2017':'2017',
        'Run2_2018':'2018',
        'Run3_2022':'2022',
        'Run3_2022EE':'2022EE',
        'Run3_2023':'2023',
        'Run3_2023BPix':'2023BPix',
    }
    return year_dict[period]

def parseVarEntry(var_entry):
    if type(var_entry) == str:
        var_name = var_entry
        need_cache = False
    else:
        var_name = var_entry['name']
        need_cache = var_entry.get('need_cache', False)
    return var_name, need_cache

def GetSamples(samples, backgrounds, signals=['GluGluToRadion','GluGluToBulkGraviton']):
    global samples_to_consider
    samples_to_consider = ['data']

    for sample_name in samples.keys():
        sample_type = samples[sample_name]['sampleType']
        if sample_type in signals or sample_name in backgrounds:
            samples_to_consider.append(sample_name)
    return samples_to_consider

def getCustomisationSplit(customisations):
    customisation_dict = {}
    if customisations is None or len(customisations) == 0: return {}
    if type(customisations) == str:
        customisations = customisations.split(';')
    if type(customisations) != list:
        raise RuntimeError(f'Invalid type of customisations: {type(customisations)}')
    for customisation in customisations:
        substrings = customisation.split('=')
        if len(substrings) != 2 :
            raise RuntimeError("len of substring is not 2!")
        customisation_dict[substrings[0]] = substrings[1]
    return customisation_dict




class AnalysisCacheTask(Task, HTCondorWorkflow, law.LocalWorkflow):
    max_runtime = copy_param(HTCondorWorkflow.max_runtime, 30.0)
    n_cpus = copy_param(HTCondorWorkflow.n_cpus, 1)

    def workflow_requires(self):
        workflow_dict = {}
        workflow_dict["anaTuple"] = {
            br_idx: AnaTupleTask.req(self, branch=br_idx)
            for br_idx, _ in self.branch_map.items()
        }
        return workflow_dict

    def requires(self):
        return [ AnaTupleTask.req(self, max_runtime=AnaTupleTask.max_runtime._default) ]

    def create_branch_map(self):
        branches = {}
        anaProd_branch_map = AnaTupleTask.req(self, branch=-1, branches=()).branch_map
        for br_idx, (sample_id, sample_name, sample_type, input_file) in anaProd_branch_map.items():
            branches[br_idx] = (sample_name, sample_type)
        return branches

    def output(self):
        sample_name, sample_type = self.branch_data
        outFileName = os.path.basename(self.input()[0].path)
        outDir = os.path.join('anaCacheTuples', self.period, sample_name, self.version)
        finalFile = os.path.join(outDir, outFileName)
        return self.remote_target(finalFile, fs=self.fs_anaCacheTuple)

    def run(self):
        #For now, this is only for bbWW, for the bbtautau we still use the AnaCahceTupleTask found in AanProd folder
        sample_name, sample_type = self.branch_data
        unc_config = os.path.join(self.ana_path(), 'FLAF', 'config',self.period, f'weights.yaml')
        producer_anacachetuples = os.path.join(self.ana_path(), 'Analysis', 'DNN_Application.py')

        global_config = os.path.join(self.ana_path(), self.global_params['analysis_config_area'], f'global.yaml')
        thread = threading.Thread(target=update_kinit_thread)
        thread.start()
        try:
            job_home, remove_job_home = self.law_job_home()
            input_file = self.input()[0]
            print(f"considering sample {sample_name}, {sample_type} and file {input_file.path}")
            customisation_dict = getCustomisationSplit(self.customisations)
            deepTauVersion = customisation_dict['deepTauVersion'] if 'deepTauVersion' in customisation_dict.keys() else ""
            with input_file.localize("r") as local_input, self.output().localize("w") as outFile:
                anaCacheTupleProducer_cmd = ['python3', producer_anacachetuples,'--inFileName', local_input.path, '--outFileName', outFile.path,  '--uncConfig', unc_config, '--globalConfig', global_config]
                if self.global_params['store_noncentral'] and sample_type != 'data':
                    anaCacheTupleProducer_cmd.extend(['--compute_unc_variations', 'True'])
                if deepTauVersion!="":
                    anaCacheTupleProducer_cmd.extend([ '--deepTauVersion', deepTauVersion])
                useDNNModel = "HH_bbWW" in self.global_params['analysis_config_area'] #Now bbtautau won't use this DNN model arg (even though this task is only for bbWW right now)
                useDNNModel = 'bbww' == self.global_params['anlaysis_name']
                if useDNNModel:
                    dnnFolder = os.path.join(self.ana_path(), self.global_params['analysis_config_area'], 'DNN', 'v24') #'ResHH_Classifier.keras')
                    anaCacheTupleProducer_cmd.extend([ '--dnnFolder', dnnFolder])
                ps_call(anaCacheTupleProducer_cmd, verbose=1)
            print(f"finished to produce anacachetuple")

        finally:
            kInit_cond.acquire()
            kInit_cond.notify_all()
            kInit_cond.release()
            thread.join()



class AnalysisCacheMergeTask(Task, HTCondorWorkflow, law.LocalWorkflow):
    max_runtime = copy_param(HTCondorWorkflow.max_runtime, 5.0)

    def workflow_requires(self):
        workflow_dep = {}
        for idx, prod_branches in self.branch_map.items():
            workflow_dep[idx] = AnalysisCacheTask.req(self, branches=prod_branches)
        return workflow_dep

    def requires(self):
        prod_branches = self.branch_data
        deps = [ AnalysisCacheTask.req(self, max_runtime=AnaCacheTask.max_runtime._default, branch=prod_br) for prod_br in prod_branches ]
        return deps

    def create_branch_map(self):
        anaProd_branch_map = AnalysisCacheTask.req(self, branch=-1, branches=()).branch_map
        prod_branches = []
        for prod_br, (sample_name, sample_type) in anaProd_branch_map.items():
            if sample_type == "data":
                prod_branches.append(prod_br)
        return { 0: prod_branches }

    def output(self, force_pre_output=False):
        outFileName = 'nanoHTT_0.root'
        output_path = os.path.join('anaCacheTuple', self.period, 'data',self.version, outFileName)
        return self.remote_target(output_path, fs=self.fs_anaCacheTuple)

    def run(self):
        producer_dataMerge = os.path.join(self.ana_path(), 'FLAF', 'AnaProd', 'MergeNtuples.py')
        with contextlib.ExitStack() as stack:
            local_inputs = [stack.enter_context(inp.localize('r')).path for inp in self.input()]
            with self.output().localize("w") as tmp_local_file:
                tmpFile = tmp_local_file.path
                dataMerge_cmd = [ 'python3', producer_dataMerge, '--outFile', tmpFile]
                dataMerge_cmd.extend(local_inputs)
                #print(dataMerge_cmd)
                ps_call(dataMerge_cmd,verbose=1)


