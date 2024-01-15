import os
import sys
import random
from multiprocessing import Pool
import subprocess as sp
import time
import yaml
import multiprocessing
import argparse
import logging
log_format = '%(asctime)s - %(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format, datefmt='%Y-%m-%d %H:%M:%S')

def run_cmd(cmd: str) -> None:
    cp = sp.run(cmd, shell=True, encoding="utf-8")
    if cp.returncode != 0:
        error = f"Some thing wrong has happened when running command [{cmd}]:{cp.stderr}"
        raise Exception(error)

def process_one_iter(log_file, clear_log, model_info: dict) -> None:
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
    elif ("light" in p1):
        train_path = p1 + "/" + p2
        config_path = ""
        work_dir = ""
        opt_arg = ""
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
        # For the inference of large language models, simply compare the inference results on the current device directly with the results generated on the GPU
        elif ('infer' in p2 and 'infer' in p3):
            cmd_run_one_iter = f"srun --job-name={job_name} --partition={partition}  --gres={gpu_requests} --time=40 python {train_path}"
            cmd_cp_one_iter = ""
        else:
            cmd_run_one_iter = f"srun --job-name={job_name} --partition={partition}  --gres={gpu_requests} --cpus-per-task=5 --mem=16G --time=40 sh SMART/tools/one_iter_tool/run_one_iter.sh {train_path} {config_path} {work_dir} {opt_arg}"
            cmd_cp_one_iter = f"srun --job-name={job_name} --partition={partition}  --gres={gpu_requests} --cpus-per-task=5 --mem=16G --time=30 sh SMART/tools/one_iter_tool/compare_one_iter.sh {package_name}"
    elif device == 'sco':
        current_path = os.getcwd()
        parent_directory = os.path.dirname(current_path)
        if (p2 == "stable_diffusion/stable-diffusion_ddim_denoisingunet_infer.py"):
            cmd_run_one_iter = f"""srun --job-name={job_name} bash -c "cd {parent_directory} && source /mnt/cache/share/deeplinkci/github/dipu_env && export ONE_ITER_TOOL_STORAGE_PATH={storage_path} && bash {current_path}/mmagic/configs/stable_diffusion/stable-diffusion_ddim_denoisingunet_one_iter.sh" """
            cmd_cp_one_iter = ""
        elif ('infer' in p2 and 'infer' in p3):
            cmd_run_one_iter = f"""srun --job-name={job_name} bash -c "cd {parent_directory} && source /mnt/cache/share/deeplinkci/github/dipu_env && export ONE_ITER_TOOL_STORAGE_PATH={storage_path} && python {current_path}/{train_path}" """
            cmd_cp_one_iter = ""
        else:
            cmd_run_one_iter = f"""srun --job-name={job_name} bash -c "cd {parent_directory}  && source /mnt/cache/share/deeplinkci/github/dipu_env && export ONE_ITER_TOOL_STORAGE_PATH={storage_path} && bash {current_path}/SMART/tools/one_iter_tool/run_one_iter.sh {train_path} {config_path} {work_dir} {opt_arg}" """
            cmd_cp_one_iter = f"""srun --job-name={job_name} bash -c "cd {parent_directory}  && source /mnt/cache/share/deeplinkci/github/dipu_env && export ONE_ITER_TOOL_STORAGE_PATH={storage_path} && bash {current_path}/SMART/tools/one_iter_tool/compare_one_iter.sh {package_name}" """
    elif device == "camb" :
        # For the inference of large language models, simply compare the inference results on the current device directly with the results generated on the GPU
        if ('infer' in p2 and 'infer' in p3):
            cmd_run_one_iter = f"srun --job-name={job_name} --partition={partition}  --gres={gpu_requests} --time=40 python {train_path}"
            cmd_cp_one_iter = ""
        else:
            cmd_run_one_iter = f"srun --job-name={job_name} --partition={partition}  --gres={gpu_requests} --time=40 sh SMART/tools/one_iter_tool/run_one_iter.sh {train_path} {config_path} {work_dir} {opt_arg}"
            cmd_cp_one_iter = f"srun --job-name={job_name} --partition={partition}  --gres={gpu_requests} --time=30 sh SMART/tools/one_iter_tool/compare_one_iter.sh {package_name} {atol} {rtol} {metric}"
    elif device == "ascend":
        if ('infer' in p2 and 'infer' in p3):
            cmd_run_one_iter = f"python {train_path}"
            cmd_cp_one_iter = ""
        else:
            cmd_run_one_iter = f"bash SMART/tools/one_iter_tool/run_one_iter.sh {train_path} {config_path} {work_dir} {opt_arg}"
            cmd_cp_one_iter = f"bash SMART/tools/one_iter_tool/compare_one_iter.sh {package_name} {atol} {rtol} {metric}"
    if clear_log:
        run_cmd(cmd_run_one_iter + f" 2>&1 > {log_file}")
    else:
        run_cmd(cmd_run_one_iter + f" 2>&1 >> {log_file}")
    run_cmd(cmd_cp_one_iter + f" 2>&1 >> {log_file}")

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

def print_file(file_name):
    with open(file_name) as f:
        lines = f.read()
        logging.info(lines)


if __name__ == '__main__':
    # set some params
    max_parall = 8
    parser = argparse.ArgumentParser(description='set some params.')
    parser.add_argument('device', type=str, help='the device to use')
    parser.add_argument('job_name', type=str, help='the name of the job')
    parser.add_argument('gpu_requests', type=str, help='the number of GPUs to request')
    parser.add_argument('partition_arg', type=str, help='the arg of partition')
    parser.add_argument('selection_of_model_list', type=str, nargs='?', default="traditional", choices=["traditional", "llm"], help='the selection of model list')
    args = parser.parse_args()

    device = args.device
    if device == 'sco':
        max_parall = 5
    job_name = args.job_name
    gpu_requests = args.gpu_requests
    partition = args.partition_arg
    # traditional and llm options are represented as one iter test for traditional models and large language models, respectively
    if args.selection_of_model_list == "traditional":
        selected_model_list = "test_one_iter_traditional_model_list.yaml"
    elif args.selection_of_model_list == "llm":
        selected_model_list = "test_one_iter_large_language_model_list.yaml"      
    logging.info(f"device: {device}, job_name: {job_name}, partition: {partition}, gpu_requests: {gpu_requests}, selected_model_list: {selected_model_list}")
    error_flag = multiprocessing.Value('i', 0)  # if encount error
    max_model_num = 100
    if device == 'cuda':
        logging.info("we use cuda!")
    elif device == "camb":
        logging.info("we use camb!")
    elif device == "ascend":
        logging.info("we use ascend!")

    logging.info(f"main process id (ppid): {os.getpid()} {os.getppid()}")

    logging.info(f"python path: {os.environ.get('PYTHONPATH', None)}")

    os.environ['DIPU_DUMP_OP_ARGS'] = "0"
    os.environ['DIPU_DEBUG_ALLOCATOR'] = "0"
    os.environ['ONE_ITER_TOOL_DEVICE'] = "dipu"
    # For traditional models, the baseline data is generated on the CPU. However, for large language models, the baseline data needs to be generated 
    # on the GPU due to the limitation of the fp16 dtype.
    if 'traditional' in selected_model_list:
        os.environ['ONE_ITER_TOOL_DEVICE_COMPARE'] = "cpu"
    else:
        os.environ['ONE_ITER_TOOL_DEVICE_COMPARE'] = "gpu"
    # os.environ['ONE_ITER_TOOL_IOSAVE_RATIO'] = "1.0"  # 0.2 by default
    curPath = os.path.dirname(os.path.realpath(__file__))
    yamlPath = os.path.join(curPath, selected_model_list)
    file_path = os.path.join(curPath, "environment_exported")
    env_variables = os.environ
    keywords_to_filter = ['DIPU', 'ONE_ITER']
    if os.path.exists(file_path):
        os.remove(file_path)
    with open("environment_exported", "w") as file:
        file.write("pwd\n")
        for key, value in env_variables.items():
            if any(keyword in key for keyword in keywords_to_filter):
                file.write(f'export {key}="{value}"\n')
    with open(yamlPath, 'r', encoding='utf-8') as f:
        if device == 'sco':
            original_list = yaml.safe_load(f.read()).get("cuda", None)
        else:
            original_list = yaml.safe_load(f.read()).get(device, None)

        if not original_list:
            logging.warning(f"Device type: {device} is not supported!")
            exit(0)

        if len(original_list) > max_model_num:
            # random choose model
            selected_list = random.sample(original_list, max_model_num)
        else:
            selected_list = original_list

        selected_model_num = len(selected_list)
        logging.info(f"model nums: {len(original_list)}, chosen model num: {selected_model_num}")

        os.mkdir("one_iter_data")

        p = Pool(max_parall)
        log_files=[]
        try:
            for i in range(selected_model_num):
                log_file = f"child_{i%max_parall}_log.txt"
                log_files.append(log_file)
                p.apply_async(process_one_iter, args=(log_file, i<max_parall, selected_list[i],), error_callback=handle_error)
            logging.info('Waiting for all subprocesses done...')
            p.close()
            p.join()
            for log_file in log_files:
                print_file(log_file)
            if (error_flag.value != 0):
                exit(1)
            logging.info('All subprocesses done.')
        except Exception as e:
            logging.error(e)
            exit(1)
