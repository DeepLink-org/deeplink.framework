#!/bin/bash

# 用于检测可用卡，找到第一张剩余显存大于输入显存的卡，并且返回卡号和剩余显存大小

#获取指定显卡的显存信息
get_memory_info() {
  # 通过nvidia-smi命令获取显存信息，并提取剩余显存部分
  nvidia-smi --id=$1 --query-gpu=memory.free --format=csv,noheader,nounits | awk '{print $1}'
}

# 函数用于获取指定显卡的总显存大小
get_total_memory() {
  # 通过nvidia-smi命令获取显存信息，并提取总显存部分
  nvidia-smi --id=$1 --query-gpu=memory.total --format=csv,noheader,nounits | awk '{print $1}'
}

# 函数用于获取显卡数量
get_num_cards() {
  # 通过nvidia-smi命令获取显卡数量，并提取数字部分
  nvidia-smi --list-gpus | wc -l
}


# 设置显卡数量
num_cards=$(get_num_cards)

memory_threshold=$1

read -r -a used_card_list <<< "$USED_CARD"


# 检测显存剩余大小
for ((id=0; id<num_cards; id++)); do
  flag=0
  for j in ${used_card_list[*]}; do
    if [[ $id == $j ]]; then
        flag=-1
        break
    fi
  done
  if [[ $flag == -1 ]]; then
        continue
  fi

  # 获取当前显卡的剩余显存大小
  free_memory=$(get_memory_info $id)
  total_memory=$(get_total_memory $id)

  # 计算已使用显存大小
  used_memory=$((total_memory - free_memory))

  remained_per=$(echo "scale=2; $free_memory / $total_memory" | bc)
  remained_G=$(echo "scale=2; $free_memory / $((1024))" | bc)
  remained_int=$((free_memory/1024))
  # echo "$id  $remained_per  $remained_G"

  if [[ $remained_int -gt $memory_threshold ]]; then
      echo $id
      echo $remained_G
      exit 0
  fi
done

echo -1
echo 0



