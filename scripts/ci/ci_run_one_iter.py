import os
import sys
import random
from multiprocessing import Pool
import subprocess as sp
import time
import yaml
import multiprocessing
import logging
log_format = '%(asctime)s - %(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format, datefmt='%Y-%m-%d %H:%M:%S')


def run_cmd(cmd: str) -> None:
    cp = sp.run(cmd, shell=True, encoding="utf-8")
    if cp.returncode != 0:
        error = f"Some thing wrong has happened when running command [{cmd}]:{cp.stderr}"
        raise Exception(error)


def process_one_iter(model_info: dict) -> None:
    begin_time = time.time()

    model_info_list = model_info['model_cfg'].split()
    if (len(model_info_list) < 3 or len(model_info_list) > 4):
        logging.error(f"Wrong model info in {model_info}")
        exit(1)
    p1 = model_info_list[0]
    p2 = model_info_list[1]
    p3 = model_info_list[2]
    p4 = model_info_list[3] if len(model_info_list) == 4 else ""

    if ("mm" in p1):
        train_path = p1 + "/tools/train.py"
        config_path = p1 + "/configs/" + p2
        work_dir = "--work-dir=./one_iter_data/" + p3
        opt_arg = p4
        package_name = "mmlab"
    elif ("DI" in p1):
        train_path = p1 + "/" + p2
        config_path = ""
        work_dir = ""
        opt_arg = ""
        package_name = "diengine"
    elif ("trans" in p1):
        train_path = p1 + "/" + p2
        config_path = ""
        work_dir = ""
        opt_arg = ""
        package_name = "transformer"
    else:
        logging.error(f"Wrong model info in {model_info}")
        exit(1)

    os.environ['ONE_ITER_TOOL_STORAGE_PATH'] = os.getcwd() + "/one_iter_data/" + p3

    storage_path = os.environ['ONE_ITER_TOOL_STORAGE_PATH']

    if model_info.get('fallback_op_list', None):
        os.environ['DIPU_FORCE_FALLBACK_OPS_LIST'] = model_info['fallback_op_list']
    else:
        os.environ['DIPU_FORCE_FALLBACK_OPS_LIST'] = ""

    logging.info(f"train_path = {train_path}, config_path = {config_path}, work_dir = {work_dir}, opt_arg = {opt_arg}")

    if not os.path.exists(storage_path):
        os.makedirs(storage_path)

    if device == 'camb':
        base_data_src = '/mnt/lustre/share/parrotsci/github/model_baseline_data'
    elif device == 'cuda':
        base_data_src = '/mnt/cache/share/parrotsci/github/model_baseline_data'
    src = f'{base_data_src}/{p3}/baseline'
    if not os.path.exists(src):
        os.makedirs(src)
    dst = f'{storage_path}/baseline'
    if not os.path.exists(dst):
        os.symlink(src, dst)

    logging.info(f"model:{p2}")

    precision = model_info.get('precision', {})
    atol = precision.get('atol', 1e-2)
    rtol = precision.get('rtol', 1e-4)
    metric = precision.get('metric', 1e-2)
    logging.info(f'Using pricision: atol-{atol}, rtol-{rtol}, metric-{metric}')

    if device == 'cuda':
        if (p2 == "stable_diffusion/stable-diffusion_ddim_denoisingunet_infer.py"):
            cmd_run_one_iter = f"srun --job-name={job_name} --partition={partition}  --gres={gpu_requests} --cpus-per-task=5 --mem=16G --time=40 sh mmagic/configs/stable_diffusion/stable-diffusion_ddim_denoisingunet_one_iter.sh"
            cmd_cp_one_iter = ""
        else:
            cmd_run_one_iter = f"srun --job-name={job_name} --partition={partition}  --gres={gpu_requests} --cpus-per-task=5 --mem=16G --time=40 sh SMART/tools/one_iter_tool/run_one_iter.sh {train_path} {config_path} {work_dir} {opt_arg}"
            cmd_cp_one_iter = f"srun --job-name={job_name} --partition={partition}  --gres={gpu_requests} --cpus-per-task=5 --mem=16G --time=30 sh SMART/tools/one_iter_tool/compare_one_iter.sh {package_name}"
    else:
        cmd_run_one_iter = f"srun --job-name={job_name} --partition={partition}  --gres={gpu_requests} --time=40 sh SMART/tools/one_iter_tool/run_one_iter.sh {train_path} {config_path} {work_dir} {opt_arg}"
        cmd_cp_one_iter = f"srun --job-name={job_name} --partition={partition}  --gres={gpu_requests} --time=30 sh SMART/tools/one_iter_tool/compare_one_iter.sh {package_name} {atol} {rtol} {metric}"

    run_cmd(cmd_run_one_iter)
    run_cmd(cmd_cp_one_iter)

    end_time = time.time()
    run_time = round(end_time - begin_time)
    hour = run_time // 3600
    minute = (run_time - 3600 * hour) // 60
    second = run_time - 3600 * hour - 60 * minute
    logging.info(f"The running time of {p2} :{hour} hours {minute} mins {second} secs")


def handle_error(error: str) -> None:
    logging.error(f"Error: {error}")
    if p is not None:
        logging.error("Kill all!")
        p.terminate()
    error_flag.value = 1


if __name__ == '__main__':
    # set some params
    max_parall = 8
    device = sys.argv[1]
    job_name = sys.argv[2]
    gpu_requests = sys.argv[3]
    partition_arg = sys.argv[4:]
    partition = ' '.join(partition_arg)
    logging.info(f"job_name: {job_name}, partition: {partition}, gpu_requests:{gpu_requests}")
    error_flag = multiprocessing.Value('i', 0)  # if encount error

    if device == 'cuda':
        model_num = 100
        logging.info("we use cuda!")
    else:
        model_num = 100
        logging.info("we use camb")

    logging.info(f"now pid!!!!: {os.getpid()} {os.getppid()}")


    logging.info(f"python path: {os.environ.get('PYTHONPATH', None)}")

    os.environ['DIPU_DUMP_OP_ARGS'] = "0"
    os.environ['DIPU_DEBUG_ALLOCATOR'] = "0"
    os.environ['ONE_ITER_TOOL_DEVICE'] = "dipu"
    os.environ['ONE_ITER_TOOL_DEVICE_COMPARE'] = "cpu"
    # os.environ['ONE_ITER_TOOL_IOSAVE_RATIO'] = "1.0"  # 0.2 by default
    curPath = os.path.dirname(os.path.realpath(__file__))
    yamlPath = os.path.join(curPath, "test_one_iter_model_list.yaml")
    with open(yamlPath, 'r', encoding='utf-8') as f:
        original_list = yaml.safe_load(f.read()).get(device, None)
        if not original_list:
            logging.error(f"Device type: {device} is not supported!")
            exit(1)

        model_num = min(len(original_list), model_num)
        logging.info(f"model nums: {len(original_list)}, chosen model num: {model_num}")

        # random choose model
        selected_list = random.sample(original_list, model_num)

        os.mkdir("one_iter_data")

        p = Pool(max_parall)
        try:
            for i in range(model_num):
                p.apply_async(process_one_iter, args=(selected_list[i],), error_callback=handle_error)
            logging.info('Waiting for all subprocesses done...')
            p.close()
            p.join()
            if (error_flag.value != 0):
                exit(1)
            logging.info('All subprocesses done.')
        except Exception as e:
            logging.error(e)
            exit(1)
