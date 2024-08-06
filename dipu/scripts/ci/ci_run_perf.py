import os
import sys
from multiprocessing import Pool
import subprocess as sp
import time
import yaml
import multiprocessing
import argparse
import logging
import json

log_format = "%(asctime)s - %(levelname)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format, datefmt="%Y-%m-%d %H:%M:%S")


def run_cmd(cmd: str) -> None:
    cp = sp.run(cmd, shell=True, encoding="utf-8")
    if cp.returncode != 0:
        error = (
            f"Some thing wrong has happened when running command [{cmd}]:{cp.stderr}"
        )
        raise Exception(error)


def parse_device_task(device):
    # get json file path
    device_config = dict()
    current_path = os.path.dirname(os.path.realpath(__file__))
    config_path = current_path + "/test_perf_config.json"
    with open(config_path) as json_config:
        json_content = json.loads(json_config.read())
        if device in json_content:
            device_config = json_content[device]
    return device_config


def process_test_perf(log_file, clear_log, task: dict) -> None:
    # READ CONFIG

    task_name = task["name"]
    storage_path = os.getcwd() + "/perf_data/" + task_name
    partition = task["partition"]
    job_name = "trial"
    gpu_requests = task["gpu_requests"]
    relative_workdir = task["relative_workdir"]
    task_script = task["script"]
    filter_pattern = task["filter"]
    op_args = task["op_args"]

    os.environ["ONE_ITER_TOOL_STORAGE_PATH"] = storage_path
    os.environ["DIPU_FORCE_FALLBACK_OPS_LIST"] = (
        task["fallback_op_list"] if "fallback_op_list" in task else ""
    )

    logging.info(f"task_name = {task_name}")

    if not os.path.exists(storage_path):
        os.makedirs(storage_path)

    # GENERATE RUN COMMAND
    cmd_run_test_perf = f"srun --job-name={job_name} --partition={partition}  --gres={gpu_requests} python {task_script} {op_args}"
    if device == "sco":
        current_path = os.getcwd()
        parent_directory = os.path.dirname(current_path)
        cmd_run_test_perf = f"""srun --job-name={job_name} bash -c "cd {parent_directory} && source scripts/ci/ci_one_iter.sh export_pythonpath_cuda {current_path} && source /mnt/cache/share/deeplinkci/github/dipu_env && cd mmlab_pack && source environment_exported && export ONE_ITER_TOOL_STORAGE_PATH={storage_path} && python {current_path}/{task_script}" """

    print(cmd_run_test_perf)

    current_path = os.getcwd()
    os.chdir(relative_workdir)
    if clear_log:
        run_cmd(cmd_run_test_perf + f" 2>&1 > {current_path}/{log_file}")
    else:
        run_cmd(cmd_run_test_perf + f" 2>&1 >> {current_path}/{log_file}")
    os.chdir(current_path)

    print("MATCH_PATTERN:", filter_pattern)
    import re

    log_content = open(f"{current_path}/{log_file}").read()
    pattern = re.compile(filter_pattern)
    match_result = pattern.search(log_content)
    run_perf = 0.0

    if match_result:
        match_result = match_result.group(0)
        float_pattern = re.compile("\d+(\.\d+)?")
        run_perf = float(float_pattern.search(match_result).group(0))
    print("RUNNING PERF:{}".format(run_perf))


def run_perf_task(device_config):
    error_flag = multiprocessing.Value("i", 0)  # if encount error

    device = device_config["name"]

    logging.info("we use {}!".format(device))
    logging.info(f"main process id (ppid): {os.getpid()} {os.getppid()}")
    logging.info(f"python path: {os.environ.get('PYTHONPATH', None)}")

    os.environ["DIPU_DUMP_OP_ARGS"] = "0"
    os.environ["DIPU_DEBUG_ALLOCATOR"] = "0"
    os.environ["ONE_ITER_TOOL_DEVICE"] = "dipu"

    current_path = os.path.dirname(os.path.realpath(__file__))
    env_file_path = os.path.join(current_path, "environment_exported")
    env_variables = os.environ
    keywords_to_filter = ["DIPU", "ONE_ITER"]
    if os.path.exists(env_file_path):
        os.remove(env_file_path)

    with open("environment_exported", "w") as file:
        file.write("pwd\n")
        for key, value in env_variables.items():
            if any(keyword in key for keyword in keywords_to_filter):
                file.write(f'export {key}="{value}"\n')

    tasks = device_config["tasks"]
    logging.info(f"tasks nums: {len(tasks)}")

    if not os.path.exists("perf_data"):
        os.mkdir("perf_data")

    pool = Pool(max_parall)
    log_files = []
    try:
        for i in range(len(tasks)):
            task = tasks[i]
            log_file = f"child_{i % max_parall}_log.txt"
            log_files.append(log_file)
            pool.apply_async(
                process_test_perf,
                args=(
                    log_file,
                    True,
                    task,
                ),
                error_callback=handle_error,
            )
        logging.info("Waiting for all subprocesses done...")
        pool.close()
        pool.join()
        for log_file in log_files:
            print_file(log_file)
        if error_flag.value != 0:
            exit(1)
        logging.info("All subprocesses done.")
    except Exception as e:
        logging.error(e)
        exit(1)


def handle_error(error: str) -> None:
    logging.error(f"Error: {error}")
    if pool is not None:
        logging.error("Kill all!")
        pool.terminate()
    error_flag.value = 1


def print_file(file_name):
    with open(file_name) as f:
        lines = f.read()
        logging.info(lines)


if __name__ == "__main__":
    # set some params
    max_parall = 8
    parser = argparse.ArgumentParser(description="set some params.")
    parser.add_argument("device", type=str, help="the device to use")
    parser.add_argument("job_name", type=str, help="the name of the job")
    args = parser.parse_args()

    device = args.device
    job_name = args.job_name

    device_config = parse_device_task(device)
    print(device_config)

    logging.info(f"device: {device}, job_name: {job_name}")
    run_perf_task(device_config)
