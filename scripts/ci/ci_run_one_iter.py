import os
import sys
import random
from multiprocessing import Pool, Queue, Manager
import subprocess as sp
import time
import yaml
import multiprocessing
import signal


#set some params
max_parall = 8
device_type = sys.argv[1]
github_job = sys.argv[2]
gpu_requests = sys.argv[3]
slurm_par_arg = sys.argv[4:]
slurm_par = ' '.join(slurm_par_arg)
print("github_job:{},slurm_par:{},gpu_requests:{}".format(github_job, slurm_par, gpu_requests))
error_flag = multiprocessing.Value('i',0) #if encount error

if device_type == 'cuda':
    random_model_num = 100
    print("we use cuda!")
else:
    random_model_num = 100
    print("we use camb")

print("now pid!!!!:",os.getpid(),os.getppid())


print("python path: {}".format(os.environ.get('PYTHONPATH', None)), flush = True)

os.environ['DIPU_DUMP_OP_ARGS'] = "0"
os.environ['DIPU_DEBUG_ALLOCATOR'] = "3"

# os.environ['ONE_ITER_TOOL_IOSAVE_RATIO'] = "1.0"  #we set 0.2 by default


def run_cmd(cmd):
    cp = sp.run(cmd, shell = True, encoding = "utf-8")
    if cp.returncode != 0:
        error = "Some thing wrong has happened when running command [{}]:{}".format(cmd, cp.stderr)
        raise Exception(error)


def process_one_iter(model_info):

    begin_time = time.time()

    model_info_list = model_info['model_cfg'].split()
    if(len(model_info_list) < 3 or len(model_info_list) > 4):
        print("Wrong model info in  {}".format(model_info), flush = True)
        exit(1)
    p1 = model_info_list[0]
    p2 = model_info_list[1]
    p3 = model_info_list[2]
    p4 = model_info_list[3] if len(model_info_list) == 4 else ""

    if("mm" in p1):
        train_path = p1 + "/tools/train.py"
        config_path = p1 + "/configs/" + p2
        work_dir = "--work-dir=./one_iter_data/" + p3
        opt_arg = p4
        package_name = "mmlab"
    elif("DI" in p1):
        train_path = p1/p2
        config_path = ""
        work_dir = ""
        opt_arg = ""
        package_name = "diengine"
    else:
        print("Wrong model info in  {}".format(model_info), flush = True)
        exit(1)

    os.environ['ONE_ITER_TOOL_STORAGE_PATH'] = os.getcwd()+"/one_iter_data/" + p3

    storage_path = os.environ['ONE_ITER_TOOL_STORAGE_PATH']
    
    if 'fallback_op_list' in model_info:
        os.environ['DIPU_FORCE_FALLBACK_OPS_LIST'] = model_info['fallback_op_list']
    else:
        os.environ['DIPU_FORCE_FALLBACK_OPS_LIST'] = ""


    print("train_path = {}, config_path = {}, work_dir = {}, opt_arg = {}".format(train_path, config_path, work_dir, opt_arg), flush = True)

    if not os.path.exists(storage_path):
        os.makedirs(storage_path)

    if device_type == 'camb':
        base_data_src = '/mnt/lustre/share/parrotsci/github/model_baseline_data'
    elif device_type == 'cuda':
        base_data_src = '/mnt/cache/share/parrotsci/github/model_baseline_data'
    src = f'{base_data_src}/{p3}/baseline'
    if not os.path.exists(src):
        os.makedirs(src)
    dst = f'{storage_path}/baseline'
    if not os.path.exists(dst):
        os.symlink(src, dst)

    print("model:{}".format(p2), flush = True)

    # github_job_name = github_job+"_"+p2
    github_job_name = github_job #为了方便统一scancel，因此使用同样的jobname

    if device_type == 'cuda':
        cmd_run_one_iter = "srun --job-name={} --partition={}  --gres={} --cpus-per-task=5 --mem=16G --time=40 sh SMART/tools/one_iter_tool/run_one_iter.sh {} {} {} {}".format(github_job_name, slurm_par, gpu_requests, train_path, config_path, work_dir, opt_arg)
        cmd_cp_one_iter = "srun --job-name={} --partition={}  --gres={} --cpus-per-task=5 --mem=16G --time=30 sh SMART/tools/one_iter_tool/compare_one_iter.sh {}".format(github_job_name, slurm_par, gpu_requests, package_name)
    else:
        cmd_run_one_iter = "srun --job-name={} --partition={}  --gres={} --time=40 sh SMART/tools/one_iter_tool/run_one_iter.sh {} {} {} {}".format(github_job_name, slurm_par, gpu_requests, train_path, config_path, work_dir, opt_arg)
        cmd_cp_one_iter = "srun --job-name={} --partition={}  --gres={} --time=30 sh SMART/tools/one_iter_tool/compare_one_iter.sh {}".format(github_job_name, slurm_par, gpu_requests, package_name)

    run_cmd(cmd_run_one_iter)
    run_cmd(cmd_cp_one_iter)

    end_time = time.time()
    run_time = round(end_time - begin_time)
    hour = run_time // 3600
    minute = (run_time - 3600 * hour) // 60
    second = run_time - 3600 * hour - 60 * minute
    print ("The running time of {} :{} hours {} mins {} secs".format(p2, hour, minute, second), flush = True)



def handle_error(error):
    print("Error: {}".format(error), flush = True)
    if p is not None:
        print("Kill all!", flush = True)
        p.terminate()
    error_flag.value = 1


if __name__=='__main__':
    curPath = os.path.dirname(os.path.realpath(__file__))
    yamlPath = os.path.join(curPath, "test_one_iter_model_list.yaml")
    original_list_f = open(yamlPath, 'r', encoding = 'utf-8')
    original_list_cfg = original_list_f.read()
    original_list_d = yaml.safe_load(original_list_cfg)

    try:
        original_list = original_list_d[device_type]
    except:
        print("The device is not supported!", flush = True)
        exit(1)

    length = len(original_list)

    if(random_model_num > length):
        random_model_num = length

    print("model num:{}, chosen model num:{}".format(length, random_model_num), flush = True)

    #random choose model
    selected_list = random.sample(original_list, random_model_num)

    os.environ['ONE_ITER_TOOL_DEVICE'] = "dipu"
    os.environ['ONE_ITER_TOOL_DEVICE_COMPARE'] = "cpu"


    os.mkdir("one_iter_data")


    p = Pool(max_parall)
    try:
        # if device_type == 'cuda':
        #     run_cmd("salloc -p {} -N4 -n4 --gres=gpu:8 --cpus-per-task 40".format(slurm_par))
        for i in range(random_model_num):
            p.apply_async(process_one_iter, args = (selected_list[i],), error_callback = handle_error)
        print('Waiting for all subprocesses done...', flush = True)
        p.close()
        p.join()
        if(error_flag.value != 0):
            exit(1)
        print('All subprocesses done.', flush = True)
    except:
        exit(1)

