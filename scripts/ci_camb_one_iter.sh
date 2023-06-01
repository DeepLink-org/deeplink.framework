#创建一个二维的列表，分别为train文件位置，配置文件位置，workdir位置和可选参数
test_model_list=(
    "mmpretrain resnet/resnet50_8xb32_in1k.py workdirs_resnet50_8xb32_in1k --no-pin-memory"
    "mmpretrain mobilenet_v2/mobilenet-v2_8xb32_in1k.py workdirs_mobilenet-v2_8xb32_in1k --no-pin-memory"
    "mmpretrain swin_transformer/swin-large_16xb64_in1k.py workdirs_swin-large_16xb64_in1k --no-pin-memory"
    "mmpretrain vision_transformer/vit-base-p16_64xb64_in1k-384px.py workdirs_vit-base-p16_64xb64_in1k-384px --no-pin-memory"
    "mmdetection detr/detr_r50_8xb2-150e_coco.py workdirs_detr_r50_8xb2-150e_coco"
    "mmdetection faster_rcnn/faster-rcnn_r101_fpn_1x_coco.py workdirs_faster-rcnn_r101_fpn_1x_coco"
    "mmdetection yolo/yolov3_d53_8xb8-320-273e_coco.py workdirs_yolov3_d53_8xb8-320-273e_coco"
    "mmsegmentation deeplabv3/deeplabv3_r50-d8_4xb2-40k_cityscapes-512x1024.py workdirs_r50-d8_4xb2-40k_cityscapes-512x1024"
    "mmsegmentation unet/unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024.py workdirs_unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024"
    "mmpose body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_udp-8xb64-210e_coco-256x192.py workdirs_td-hm_hrnet-w32_udp-8xb64-210e_coco-256x192"
    "mmdetection3d pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py workdirs_pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class"
    "mmdetection ssd/ssd300_coco.py workdirs_ssd300_coco"
    "mmaction2 recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py workdirs_ssd300_coco"
)

length=${#test_model_list[@]}
max_parall=4

mkfifo ./fifo.$$ && exec 796<> ./fifo.$$ && rm -f ./fifo.$$
for ((i=0; i<$max_parall; i++)); do
    echo  "init add placed row $i" >&796
done 

export ONE_ITER_TOOL_DEVICE=dipu
export ONE_ITER_TOOL_DEVICE_COMPARE=cpu

#建立软链接，方便找到数据集
mkdir data
ln -s /mnt/lustre/share_data/PAT/datasets/Imagenet data/imagenet
ln -s /mnt/lustre/share_data/PAT/datasets/mscoco2017  data/coco
ln -s /mnt/lustre/share_data/PAT/datasets/mmseg/cityscapes data/cityscapes
ln -s /mnt/lustre/share_data/slc/mmdet3d/mmdet3d data/kitti
ln -s /mnt/lustre/share_data/PAT/datasets/mmaction/Kinetics400 data/kinetics400

# 没有的一些包，进行安装
pip install terminaltables
pip install pycocotools

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
    if [ -d "$ONE_ITER_TOOL_STORAGE_PATH" ]; then
        echo "File already exists $ONE_ITER_TOOL_STORAGE_PATH"
    else
        # 创建当前文件夹路径
        mkdir -p "$ONE_ITER_TOOL_STORAGE_PATH"
        echo "make dir"
    fi
    sh SMART/tools/one_iter_tool/run_one_iter_test.sh ${train_path} ${config_path} ${work_dir} ${opt_arg}
    sh SMART/tools/one_iter_tool/compare_one_iter_test.sh
    echo  "after add place row $i"  1>&796
}&
done

wait

echo Done
