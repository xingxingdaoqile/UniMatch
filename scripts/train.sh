#!/bin/bash
#now=$(date +"%Y%m%d_%H%M%S")#这行代码使用 date 命令获取当前日期和时间，并将其格式化为 YYYYMMDD_HHMMSS 格式。该变量 now 之后会被用来命名日志文件，确保每次运行时生成的日志文件不会覆盖掉之前的日志。
now=$(date +"%Y%m%d_%H%M%S")

# modify these augments if you want to try other datasets, splits or methods
# dataset: ['pascal', 'cityscapes', 'coco']
# method: ['unimatch', 'fixmatch', 'supervised']
# exp: just for specifying the 'save_path'
# split: ['92', '1_16', 'u2pl_1_16', ...]. Please check directory './splits/$dataset' for concrete splits
dataset='cityscapes'
method='unimatch'
exp='r101'
split='1_2'

#config=configs/${dataset}.yaml
config=/kaggle/working/UniMatch/configs/${dataset}.yaml
#labeled_id_path=splits/$dataset/$split/labeled.txt
labeled_id_path=/kaggle/working/UniMatch/splits/$dataset/$split/labeled.txt
#unlabeled_id_path=splits/$dataset/$split/unlabeled.txt
unlabeled_id_path=/kaggle/working/UniMatch/splits/$dataset/$split/unlabeled.txt
#save_path=exp/$dataset/$method/$exp/$split
save_path=/kaggle/working/UniMatch/exp/$dataset/$method/$exp/$split

mkdir -p $save_path

'''python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    $method.py \
    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
    --save-path $save_path --port $2 2>&1 | tee $save_path/$now.log'''
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    /kaggle/working/UniMatch/$method.py \
    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
    --save-path $save_path --port $2 2>&1 | tee $save_path/$now.log

 #--nproc_per_node=$1：指定每个节点的进程数，由脚本的第一个参数 $1 决定（例如 GPU 数量）。
 #--master_addr=localhost：设置主节点的地址为 localhost（即本地机器）。
 #--master_port=$2：设置主节点通信的端口，由脚本的第二个参数 $2 决定。

 #2>&1：将标准错误输出重定向到标准输出，使得所有输出（包括错误信息）都可以记录。
 #tee $save_path/$now.log：使用 tee 命令将输出同时保存到终端和日志文件 $save_path/$now.log 中。