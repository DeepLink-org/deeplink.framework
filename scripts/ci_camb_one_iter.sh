#创建一个二维的列表，分别为train文件位置，配置文件位置，workdir位置和可选参数
test_model_list=(
    "mmpretrain resnet/resnet50_8xb32_in1k.py workdirs_resnet50_8xb32_in1k --no-pin-memory"
    "mmpretrain mobilenet_v2/mobilenet-v2_8xb32_in1k.py workdirs_mobilenet-v2_8xb32_in1k --no-pin-memory"
    "mmpretrain swin_transformer/swin-large_16xb64_in1k.py workdirs_swin-large_16xb64_in1k --no-pin-memory"
    "mmpretrain vision_transformer/vit-base-p16_64xb64_in1k-384px.py workdirs_vit-base-p16_64xb64_in1k-384px --no-pin-memory"
)

length=${#test_model_list[@]}
max_parall=3

mkfifo ./fifo.$$ && exec 796<> ./fifo.$$ && rm -f ./fifo.$$
for ((i=0; i<$max_parall; i++)); do
    echo  "init add placed row $i" >&796
done 

export ONE_ITER_TOOL_DEVICE=dipu
export ONE_ITER_TOOL_DEVICE_COMPARE=cpu

for ((i=0; i<$length; i++)); do
{
    read -u 796
    read -r p1 p2 p3 p4 <<< ${test_model_list[i]}
    train_path="${p1}/tools/train.py"
    config_path="${p1}/configs/${p2}"
    work_dir="--work-dir=./${p3}"
    opt_arg="${p4}"
    export ONE_ITER_TOOL_STORAGE_PATH=$(pwd)/${p3}/one_iter_data
    echo "${train_path} ${config_path} ${work_dir} ${opt_arg}"
    sh SMART/tools/one_iter_tool/run_one_iter_test.sh ${train_path} ${config_path} ${work_dir} ${opt_arg}
    sh SMART/tools/one_iter_tool/compare_one_iter_test.sh
    echo  "after add place row $i"  1>&796
}&
done

wait

echo Done
