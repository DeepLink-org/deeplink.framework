#!/bin/bash

#创建一个二维的列表，分别为train文件位置，配置文件位置，workdir位置和可选参数
original_list=(
    "mmpretrain resnet/resnet50_8xb32_in1k.py workdirs_resnet50_8xb32_in1k --no-pin-memory"
    "mmpretrain mobilenet_v2/mobilenet-v2_8xb32_in1k.py workdirs_mobilenet-v2_8xb32_in1k --no-pin-memory"
    "mmpretrain swin_transformer/swin-large_16xb64_in1k.py workdirs_swin-large_16xb64_in1k --no-pin-memory"
    "mmpretrain vision_transformer/vit-base-p16_64xb64_in1k-384px.py workdirs_vit-base-p16_64xb64_in1k-384px --no-pin-memory"
    # "mmdetection detr/detr_r50_8xb2-150e_coco.py workdirs_detr_r50_8xb2-150e_coco"
    # "mmdetection faster_rcnn/faster-rcnn_r101_fpn_1x_coco.py workdirs_faster-rcnn_r101_fpn_1x_coco"
    # "mmdetection yolo/yolov3_d53_8xb8-320-273e_coco.py workdirs_yolov3_d53_8xb8-320-273e_coco"
    # "mmsegmentation deeplabv3/deeplabv3_r50-d8_4xb2-40k_cityscapes-512x1024.py workdirs_r50-d8_4xb2-40k_cityscapes-512x1024"
    # "mmsegmentation unet/unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024.py workdirs_unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024"
    # "mmpose body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_udp-8xb64-210e_coco-256x192.py workdirs_td-hm_hrnet-w32_udp-8xb64-210e_coco-256x192"
    # "mmdetection3d pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py workdirs_pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class"
    # "mmdetection ssd/ssd300_coco.py workdirs_ssd300_coco"
    # "mmaction2 recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py workdirs_ssd300_coco"
    # "mmdetection dyhead/atss_swin-l-p4-w12_fpn_dyhead_ms-2x_coco.py workdirs_atss_swin-l-p4-w12_fpn_dyhead_ms-2x_coco"
    # "mmpretrain efficientnet/efficientnet-b2_8xb32_in1k.py workdirs_efficientnet-b2_8xb32_in1k"
    # "mmdetection fcos/fcos_r50-dcn-caffe_fpn_gn-head-center-normbbox-centeronreg-giou_1x_coco.py workdirs_fcos_r50-dcn-caffe_fpn_gn-head-center-normbbox-centeronreg-giou_1x_coco"
    # "mmdetection mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py workdirs_mask-rcnn_r50_fpn_1x_coco"
    # "mmpretrain mobilenet_v3/mobilenet-v3-large_8xb128_in1k.py workdirs_mobilenet-v3-large_8xb128_in1k"
    # "mmdetection retinanet/retinanet_r50_fpn_1x_coco.py workdirs_retinanet_r50_fpn_1x_coco"
    # "mmpretrain convnext/convnext-small_32xb128_in1k.py workdirs_convnext-small_32xb128_in1k"
    # "mmsegmentation deeplabv3plus/deeplabv3plus_r50-d8_4xb2-40k_cityscapes-512x1024.py workdirs_deeplabv3plus_r50-d8_4xb2-40k_cityscapes-512x1024"
    # "mmsegmentation pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py workdirs_pspnet_r50-d8_4xb2-40k_cityscapes-512x1024"
    # "mmocr textdet/dbnet/dbnet_resnet50-dcnv2_fpnc_1200e_icdar2015.py workdirs_dbnet_resnet50-dcnv2_fpnc_1200e_icdar2015"
    # "mmocr textrecog/crnn/crnn_mini-vgg_5e_mj.py workdirs_crnn_mini-vgg_5e_mj"
)

length=${#original_list[@]}
max_parall=4
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
export ONE_ITER_TOOL_DEVICE_COMPARE=cpu

# 没有的一些包，进行安装
pip install terminaltables
pip install pycocotools
pip install shapely

for ((i=0; i<$random_model_num; i++)); do
{
    set -e
    pid=$BASHPID  # 存储子进程的PID号
    read -u 796
    read -r p1 p2 p3 p4 <<< ${selected_list[i]}
    train_path="${p1}/tools/train.py"
    config_path="${p1}/configs/${p2}"
    work_dir="--work-dir=./${p3}"
    opt_arg="${p4}"
    export ONE_ITER_TOOL_STORAGE_PATH=$(pwd)/${p3}/one_iter_data
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
}&
pid=$!  # 存储子进程的PID号
pids+=("$pid")
echo "PID: $pid"  # 输出子进程的PID号
done

while true; do
    all_finished=true
    for pid in "${pids[@]}"; do
        if [ -f "$pid.done" ]; then
                echo "Child process with PID $pid exited successfully."
                unset -v "pids[$pid]"  # 从数组中删除相应的元素
                continue
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            # 如果存在 "$pid.done"，那直接删
            if [ -f "$pid.done" ]; then
                echo "Child process with PID $pid exited successfully."
                unset -v "pids[$pid]"  # 从数组中删除相应的元素
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
