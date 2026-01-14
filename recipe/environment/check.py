import json
import re

# 加载 JSON 文件
file_path = '/inspire/hdd/global_user/xucaijun-253108120121/verl/recipe/environment/save_json_list_20251117_062834.json'

with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 初始化统计变量
count_distance_reward_zeros = 0
count_content_with_mod = 0
count_negative_one = 0
total_items = len(data)

# 定义正则表达式，匹配 "mod", "modulo" 且避免匹配到 "modern" 等
mod_pattern = re.compile(r'\b(mod|modulo)\b')

# 遍历数据并进行统计
for item in data:
    # 判断 distance = 0 和 reward = 0.0
    if item.get('distance') == 0 and item.get('reward') == 0.0:
        count_distance_reward_zeros += 1
        
        # 判断 content 中是否包含 "mod" 或 "modulo"
        for prompt in item.get('prompt', []):
            if 'content' in prompt:
                content = prompt['content']
                if mod_pattern.search(content):  # 使用正则表达式匹配关键词
                    count_content_with_mod += 1
                    
        if item.get("response") =="\\boxed{-1}":
            count_negative_one +=1
        print(f"mod :{count_content_with_mod},negative : {count_negative_one}")
# 计算占比
percentage_with_mod = (count_content_with_mod / count_distance_reward_zeros) * 100 if count_distance_reward_zeros > 0 else 0
percentage_negative_one = (count_negative_one / count_distance_reward_zeros) * 100 if count_distance_reward_zeros > 0 else 0
# 输出结果
print(f"distance = 0 且 reward = 0.0 的个数: {count_distance_reward_zeros}")
print(f"content 中包含 'mod' 或 'modulo' 的个数: {count_content_with_mod}")
print(f"'mod' 或 'modulo' 占所有问题的比例: {percentage_with_mod:.2f}%")

print(f"ground truth -1 : {count_negative_one}")
print(f"ground truth -1 占所有问题的比例: {percentage_negative_one:.2f}%")
