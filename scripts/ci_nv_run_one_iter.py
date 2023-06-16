import os
import sys
import random
from multiprocessing import Pool, Queue
import subprocess as sp
import pynvml
import time


#set some params
max_parall = 4
random_model_num = 4

print("python path: {}".format(os.environ.get('PYTHONPATH',None)))

os.environ['DIPU_DUMP_OP_ARGS'] = "0"


def run_cmd(cmd):
    cp = sp.run(cmd,shell=True,encoding="utf-8")
    if cp.returncode != 0:
        error = "Some thing wrong has happened when running command [{cmd}]:{cp.stderr}"
        raise Exception(error)

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

def find_available_card(mem_threshold,used_card):
    device_count,gpu_info = get_gpu_info()
    while True:
        for i in range(device_count):
            if(gpu_info[i]["free_memory"]>=mem_threshold and (not i in used_card) ):
                return i,gpu_info[i]["free_memory"]
            
        time.sleep(5)

def process_one_iter(q,model_info):
    used_card = q.get(True)
    available_card , cur_gpu_free = find_available_card(30,used_card)
    used_card.append(available_card)
    q.put(used_card)

    begin_time = time.time()

    model_info_list = model_info.split()
    if(len(model_info_list)<3 or len(model_info_list)>4):
        print("wrong model info in  {}".format(model_info))
    p1 = model_info_list[0]
    p2 = model_info_list[1]
    p3 = model_info_list[2]
    p4 = model_info_list[3] if len(model_info_list)==4 else ""

    train_path = p1+"/tools/train.py"
    config_path = p1 + "/configs/" + p2
    work_dir="--work-dir=./one_iter_data/" + p3
    opt_arg = p4
    os.environ['ONE_ITER_TOOL_STORAGE_PATH'] = os.getcwd()+"/one_iter_data/" + p3

    print(train_path,config_path,work_dir,opt_arg)

    if not os.path.exists(os.environ['ONE_ITER_TOOL_STORAGE_PATH']):            
        os.makedirs(os.environ['ONE_ITER_TOOL_STORAGE_PATH']) 

    print("cardnum:{},model:{},cur_card_free:{}".format(available_card,p2,cur_gpu_free))

    if(p2=="configs/stable_diffusion/stable-diffusion_ddim_denoisingunet_infer.py"):
        cmd = "CUDA_VISIBLE_DEVICES={} python mmagic/configs/stable_diffusion/stable-diffusion_ddim_denoisingunet_infer.py".format(available_card)
        run_cmd(cmd)
    else:
        cmd1 = "CUDA_VISIBLE_DEVICES={} sh SMART/tools/one_iter_tool/run_one_iter.sh {} {} {} {}".format(available_card,train_path,config_path,work_dir,opt_arg)
        cmd2 = "CUDA_VISIBLE_DEVICES={} sh SMART/tools/one_iter_tool/compare_one_iter.sh".format(available_card)
        run_cmd(cmd1)
        run_cmd(cmd2)

    end_time = time.time()
    run_time = round(end_time-begin_time)
    hour = run_time//3600
    minute = (run_time-3600*hour)//60
    second = run_time-3600*hour-60*minute
    print ("The running time of {} :{h} hours {m} mins {} secs".format(p2,hour,minute,second))

    used_card = q.get(True)
    used_card.remove(available_card)
    q.put(used_card)



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
    used_card = []
    q.put(used_card)
    p = None
    try:
        with Pool(max_parall) as p:
            for i in range(random_model_num):
                p.apply_async(process_one_iter, args=(q,selected_list[i]))
                print('Waiting for all subprocesses done...')
                p.close()
                p.join()
                print('All subprocesses done.')
    except Exception as e:
        print("Error:", e)
        if p is not None:
            p.terminate()
        exit(1)