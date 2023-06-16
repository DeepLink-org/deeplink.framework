import os
import sys
import random
from multiprocessing import Pool, Queue
import pynvml

sys.stdout.flush = True

#set some params
max_parall = 4
random_model_num = 4

print("python path: {}".format(os.environ.get('PYTHONPATH',None)))

os.environ['DIPU_DUMP_OP_ARGS'] = "0"


def get_gpu_info():
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    gpu_info = []
    
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        gpu_name = pynvml.nvmlDeviceGetName(handle).decode()
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_memory = memory_info.total / 1024**3  # 转换为以MB为单位
        free_memory = memory_info.free / 1024**3  # 转换为以MB为单位
        
        gpu_info.append({"gpu_id": i, "gpu_name": gpu_name, "total_memory": total_memory, "free_memory": free_memory})
    
    pynvml.nvmlShutdown()
    
    return device_count, gpu_info

def find_available_card(mem_threshold):
    device_count,gpu_info = get_gpu_info()
    while True:
        for i in range(device_count):
            if(gpu_info[i]["free_memory"]>=mem_threshold)

def process_one_iter(q,model_info):




if __name__=='__main__':
    original_list=[
        "mmpretrain resnet/resnet50_8xb32_in1k.py workdirs_resnet50_8xb32_in1k --no-pin-memory"   ,
        "mmpretrain swin_transformer/swin-large_16xb64_in1k.py workdirs_swin-large_16xb64_in1k --no-pin-memory"   ,
        "mmpretrain vision_transformer/vit-base-p16_64xb64_in1k-384px.py workdirs_vit-base-p16_64xb64_in1k-384px --no-pin-memory"  ,
        "mmdetection detr/detr_r50_8xb2-150e_coco.py workdirs_detr_r50_8xb2-150e_coco"  ,
        "mmdetection yolo/yolov3_d53_8xb8-320-273e_coco.py workdirs_yolov3_d53_8xb8-320-273e_coco" ,
        "mmpose body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_udp-8xb64-210e_coco-256x192.py workdirs_td-hm_hrnet-w32_udp-8xb64-210e_coco-256x192", 
        "mmdetection ssd/ssd300_coco.py workdirs_ssd300_coco" ,
        "mmaction2 recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py workdirs_tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb"   ,
        "mmpretrain efficientnet/efficientnet-b2_8xb32_in1k.py workdirs_efficientnet-b2_8xb32_in1k --no-pin-memory"  ,
        "mmpretrain mobilenet_v3/mobilenet-v3-large_8xb128_in1k.py workdirs_mobilenet-v3-large_8xb128_in1k --no-pin-memory"   ,
        "mmocr textrecog/crnn/crnn_mini-vgg_5e_mj.py workdirs_crnn_mini-vgg_5e_mj",
        "mmdetection fcos/fcos_r50-dcn-caffe_fpn_gn-head-center-normbbox-centeronreg-giou_1x_coco.py workdirs_fcos_r50-dcn-caffe_fpn_gn-head-center-normbbox-centeronreg-giou_1x_coco", 
        "mmdetection retinanet/retinanet_r50_fpn_1x_coco.py workdirs_retinanet_r50_fpn_1x_coco"  ,
        "mmsegmentation deeplabv3/deeplabv3_r50-d8_4xb2-40k_cityscapes-512x1024.py workdirs_r50-d8_4xb2-40k_cityscapes-512x1024" ,
        "mmsegmentation deeplabv3plus/deeplabv3plus_r50-d8_4xb2-40k_cityscapes-512x1024.py workdirs_deeplabv3plus_r50-d8_4xb2-40k_cityscapes-512x1024",
    ]

    length = len(original_list)

    if(random_model_num>length):
        random_model_num = length  

    print("model num:{}, chosen model num:{}".format(length,random_model_num))

    #random choose model
    selected_list = random.sample(original_list, random_model_num)

    os.environ['ONE_ITER_TOOL_DEVICE'] = "dipu"
    os.environ['ONE_ITER_TOOL_DEVICE_COMPARE'] = "cpu"

    os.mkdir("one_iter_data")

    q = Queue()
    # p = Pool(max_parall)
    # for i in range(random_model_num):
    #     p.apply_async(process_one_iter, args=(q,selected_list[i]))
    # print('Waiting for all subprocesses done...')
    # p.close()
    # p.join()
    # print('All subprocesses done.')
    try:
        with Pool(max_parall) as p:
            for i in range(random_model_num):
                p.apply_async(process_one_iter, args=(q,selected_list[i]))
                print('Waiting for all subprocesses done...')
                p.close()
                p.join()
                print('All subprocesses done.')
    except Exception as e:
        # 捕获子进程的异常
        print("Error:", e)
        # 终止所有子进程和父进程
        p.terminate()