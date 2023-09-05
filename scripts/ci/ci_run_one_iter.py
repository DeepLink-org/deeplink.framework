import os
import sys
import random
from multiprocessing import Pool
import subprocess as sp
import time
import yaml
import multiprocessing
import logging


class Logger(object):
    def __init__(self, handler, name, level=logging.INFO):
        self.logger = logging.getLogger(name)
        formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(level)

    def get_logger(self):
        return self.logger

main_logger = Logger(logging.StreamHandler(), "parrent log").get_logger()


def run_cmd(cmd: str) -> None:
    cp = sp.run(cmd, shell=True, encoding="utf-8")
    if cp.returncode != 0:
        error = f"Some thing wrong has happened when running command [{cmd}]:{cp.stderr}"
        raise Exception(error)


def log_pipe_conn(func):
    def wrapper(pipe_conn, *args, **kwargs):
        func(*args, **kwargs)
        pipe_conn.close()
    return wrapper


@log_pipe_conn
def process_one_iter(model_info: dict, logger) -> None:

    begin_time = time.time()
    model_info_list = model_info['model_cfg'].split()
    if (len(model_info_list) < 3 or len(model_info_list) > 4):
        logger.error(f"Wrong model info in {model_info}")
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
    else:
        logger.error(f"Wrong model info in {model_info}")
        exit(1)

    os.environ['ONE_ITER_TOOL_STORAGE_PATH'] = os.getcwd() + "/one_iter_data/" + p3

    storage_path = os.environ['ONE_ITER_TOOL_STORAGE_PATH']

    if model_info.get('fallback_op_list', None):
        os.environ['DIPU_FORCE_FALLBACK_OPS_LIST'] = model_info['fallback_op_list']
    else:
        os.environ['DIPU_FORCE_FALLBACK_OPS_LIST'] = ""

    logger.info(f"train_path = {train_path}, config_path = {config_path}, work_dir = {work_dir}, opt_arg = {opt_arg}")

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

    logger.info(f"model:{p2}")

    precision = model_info.get('precision', {})
    atol = precision.get('atol', 1e-2)
    rtol = precision.get('rtol', 1e-4)
    metric = precision.get('metric', 1e-2)
    logger.info(f'Using pricision: atol-{atol}, rtol-{rtol}, metric-{metric}')

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
    logger.info(f"The running time of {p2} :{hour} hours {minute} mins {second} secs")


def handle_error(error: str) -> None:
    main_logger.error(f"Error: {error}")
    if p is not None:
        main_logger.error("Kill all!")
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
    max_model_num = 100
    if device == 'cuda':
        logging.info("we use cuda!")
    else:
        logging.info("we use camb")

    logging.info(f"main process id (ppid): {os.getpid()} {os.getppid()}")


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

        if len(original_list) > max_model_num:
            selected_list = random.sample(original_list, max_model_num)
        else:
            selected_list = original_list

        selected_model_num = len(selected_list)
        logging.info(f"model nums: {len(original_list)}, chosen model num: {selected_model_num}")

        # random choose model


        os.mkdir("one_iter_data")

        p = Pool(max_parall)
        parent_conns = []
        child_conns = []
        try:
            for i in range(selected_model_num):
                parent_conn, child_conn = multiprocessing.Pipe()
                parent_conns.append(parent_conn)

                class PipeHandler(logging.Handler):
                    def emit(self, record):
                        msg = self.format(record)
                        child_conn.send(msg)

                logger = Logger(PipeHandler(), "child{i} log").get_logger()
                p.apply_async(process_one_iter, args=(child_conn, selected_list[i], logger), error_callback=handle_error)
            logging.info('Waiting for all subprocesses done...')
            p.close()
            p.join()

            for i in range(selected_model_num):
                message = parent_conns[i].recv()
                print(message)

            if (error_flag.value != 0):
                exit(1)
            logging.info('All subprocesses done.')
        except Exception as e:
            logging.error(e)
            exit(1)
