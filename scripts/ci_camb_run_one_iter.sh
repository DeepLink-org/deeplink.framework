#!/bin/bash

#创建一个二维的列表，分别为train文件位置，配置文件位置，workdir位置和可选参数
original_list=(
    "mmpretrain resnet/resnet50_8xb32_in1k.py workdirs_resnet50_8xb32_in1k --no-pin-memory"  
)

length=${#original_list[@]}
max_parall=1
random_model_num=100 #如果超过，会自动设置为模型总数

if [ $random_model_num -gt $length ]; then
    random_model_num=$length
fi

echo $length
selected_list=()

# 随机选取模型
for ((i=0; i<random_model_num; i++)); do
    random_index=$((RANDOM % length))
    random_element=${original_list[random_index]}
    selected_list+=("$random_element")
    original_list=("${original_list[@]:0:random_index}" "${original_list[@]:random_index+1}")
    length=${#original_list[@]}
done


mkfifo ./fifo.$$ && exec 796<> ./fifo.$$ && rm -f ./fifo.$$
for ((i=0; i<$max_parall; i++)); do
    echo  "init add placed row $i" >&796
done 

pids=()

export ONE_ITER_TOOL_DEVICE=dipu
export ONE_ITER_TOOL_DEVICE_COMPARE=origin


mkdir one_iter_data

for ((i=0; i<$random_model_num; i++)); do
{
    set -e
    
    # 记录开始时间（以纳秒为单位）
    startTime=$(date +%s%N)

    pid=$BASHPID  # 存储子进程的PID号
    read -u 796
    read -r p1 p2 p3 p4 <<< ${selected_list[i]}
    train_path="${p1}/tools/train.py"
    config_path="${p1}/configs/${p2}"
    work_dir="--work-dir=./one_iter_data/${p3}"
    opt_arg="${p4}"
    export ONE_ITER_TOOL_STORAGE_PATH=$(pwd)/one_iter_data/${p3}
    echo "${train_path} ${config_path} ${work_dir} ${opt_arg}"
    if [ -d "$ONE_ITER_TOOL_STORAGE_PATH" ]; then
        echo "File already exists $ONE_ITER_TOOL_STORAGE_PATH"
    else
        # 创建当前文件夹路径
        mkdir -p "$ONE_ITER_TOOL_STORAGE_PATH"
        echo "make dir"
    fi
    sh SMART/tools/one_iter_tool/run_one_iter.sh ${train_path} ${config_path} ${work_dir} ${opt_arg}
    sh SMART/tools/one_iter_tool/compare_one_iter.sh
    echo  "after add place row $i"  1>&796
    touch "$pid.done" 

    # 记录结束时间（以纳秒为单位）
    endTime=$(date +%s%N)

    # 计算时间差（以纳秒为单位）
    timeDiff=$((endTime - startTime))

    # 将时间差转换为小时、分钟和秒
    seconds=$((timeDiff / 1000000000))
    minutes=$((seconds / 60))
    seconds=$((seconds % 60))
    hours=$((minutes / 60))
    minutes=$((minutes % 60))

    # 显示结果
    echo "The running time of ${p2} ：${hours} H ${minutes} min ${seconds} min"

}&
pid=$!  # 存储子进程的PID号
pids+=("$pid")
read -r p1 p2 p3 p4 <<< ${selected_list[i]}
echo "PID: $pid ,name:$p2"  # 输出子进程的PID号
done

while true; do
    all_finished=true
    for index in "${!pids[@]}"; do
        pid="${pids[index]}"
        if ! kill -0 "$pid" 2>/dev/null; then
            # 如果存在 "$pid.done"，那直接删
            if [ -f "$pid.done" ]; then
                echo "Child process with PID $pid exited successfully."
                rm -rf "$pid.done"
                unset 'pids[index]'  # 删除相应的数组元素
                continue
            fi
            echo "Child process with PID $pid encountered an error. Exiting all child processes."
            # 结束所有子进程
            for pid_to_kill in "${pids[@]}"; do
                kill "$pid_to_kill" 2>/dev/null
            done
            exit 1
        fi
        all_finished=false
    done

    if $all_finished; then
        break
    fi

    sleep 2  # 适当调整轮询的间隔时间
done


echo Done
