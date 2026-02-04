import os
import re
import json
from openai import OpenAI

# Configuration
API_KEY = os.getenv("DASHSCOPE_API_KEY")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen3-max"

# Definition of Categories (from monitor.py)
CLASSES = [
    '酚酞试剂', '石蕊试剂', '药剂颗粒瓶', '试管', '烧杯', '玻璃棒', '量筒', '胶头滴管', 
    '广口玻璃瓶', '塑料滴管', '点滴板', '废液缸', '药匙', '手', '蒸馏水水杯', '试管架'
]

# Action Mapping to Constants
ACTION_MAP = {
    "拿取": "ACT_FETCH",
    "加料": "ACT_ADD",
    "倾倒": "ACT_POUR",
    "搅拌": "ACT_STIR",
    "滴加": "ACT_DROP",
    "放置": "ACT_PUT",
    "刷洗": "ACT_WASH"
}

# Raw Steps Example (Default Input)
DEFAULT_RAW_STEPS = """
(1)配置硫酸铜溶液：
    拿取硫酸铜颗粒药剂瓶；
    使用药匙取适量硫酸铜颗粒加入烧杯中；
    放回药剂瓶；
    向烧杯中加入适量蒸馏水；
    使用玻璃棒将烧杯中溶液搅拌均匀；
    将烧杯中溶液倒入广口瓶中；
    向烧杯中加入适量蒸馏水；
    刷洗烧杯和玻璃棒；
    将烧杯中的废液倒入废液缸中；
(2)配置氢氧化钠溶液：
    拿取氢氧化钠颗粒药剂瓶；
    使用药匙取适量氢氧化钠颗粒加入烧杯中；
    放回药剂瓶；
    向烧杯中加入适量蒸馏水；
    使用玻璃棒将烧杯中溶液搅拌均匀；
    将烧杯中溶液倒入广口瓶中；
    向烧杯中加入适量蒸馏水；
    刷洗烧杯和玻璃棒；
    将烧杯中的废液倒入废液缸中；
(3)配置柠檬酸溶液：
    拿取柠檬酸颗粒药剂瓶；
    使用药匙取适量柠檬酸颗粒加入烧杯中；
    放回药剂瓶；
    向烧杯中加入适量蒸馏水；
    使用玻璃棒将烧杯中溶液搅拌均匀；
    将烧杯中溶液倒入广口瓶中；
    向烧杯中加入适量蒸馏水；
    刷洗烧杯和玻璃棒；
    将烧杯中的废液倒入废液缸中；
(4)在点滴板上进行实验：
    拿取点滴板；
    使用氢氧化钠溶液中的胶头滴管在点滴板上滴加3-5滴溶液；
    使用柠檬酸溶液中的胶头滴管在点滴板上滴加3-5滴溶液；
    使用石蕊溶液中的胶头滴管在点滴板上滴加2-3滴溶液；
    使用酚酞溶液中的胶头滴管在点滴板上滴加2-3滴溶液；
    将点滴板上的溶液倒入废液缸；
    放置点滴板；
(5)在试管中进行实验：
    拿取试管；
    使用硫酸铜溶液中的塑料滴管向试管中滴加9-10滴溶液；
    使用氢氧化钠溶液中的胶头滴管向试管中滴加3-5滴溶液；
    将试管放置回试管架；
"""

def get_standardized_steps(raw_steps):
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    system_prompt = f"""
你是一个化学实验步骤标准化助手。请将用户输入的原始实验步骤转换为标准的结构化格式。
严格遵守以下要求：
1. 输出格式必须为每行一个步骤，格式如下：
   - Step ID: {{Action: 动作, Subject: 主体, Object: 客体, Constraint: 约束条件, Label: 标签}}
2. 对于每个大流程（例如"（1）. 配置硫酸铜溶液"），请在第一个步骤的Label字段标记为"start"，最后一个步骤的Label字段标记为"end"，中间步骤标记为"none"。
3. 在每个大流程开始前，必须输出流程名称，格式为：**流程名称** (例如 **(1)配置硫酸铜溶液**)
4. Action只能从以下列表中选择：{', '.join(ACTION_MAP.keys())}
5. Subject and Object must be selected ONLY from the following list: {', '.join(CLASSES)}
6. 如果Subject或Object不在列表中，请选择最接近的类别或留空。
7. 保持步骤顺序与原始输入一致。
8. 输出必须包含 Step ID (例如 - Step 1: ...).
"""

    user_prompt = f"原始实验步骤：\n{raw_steps}"

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            extra_body={"enable_thinking": False},
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return None

def parse_steps(llm_output):
    flat_steps = []
    processes = []
    
    # Regex to match: - Step 1: {Action: ..., Subject: ..., Object: ..., Constraint: ..., Label: ...}
    pattern = r"- Step (\d+):\s*\{Action:\s*([^,]+),\s*Subject:\s*([^,]+),\s*Object:\s*([^,]+),\s*Constraint:\s*(.*?),\s*Label:\s*(.*?)\}"
    
    lines = llm_output.split('\n')
    current_phase_name = ""
    current_process = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check for phase headers (e.g., **1. 配置硫酸铜溶液**)
        if line.startswith('**'):
            # If we have an existing process, save it
            if current_process:
                processes.append(current_process)
            
            # Start new process
            current_phase_name = line.replace('*', '').strip()
            current_process = {
                "process_name": current_phase_name,
                "state": "not executed",
                "steps": [],
                "pass_check": False,
                "missing_incorrect_steps": "",
                "qwen_analysis": "",
                "video_link": "",
                "sop_text": "", # Placeholder, to be filled if we had raw text mapping
                "sop_video_link": ""
            }
            continue
            
        match = re.search(pattern, line)
        if match:
            step_id = match.group(1)
            action = match.group(2).strip()
            subject = match.group(3).strip()
            obj = match.group(4).strip()
            constraint = match.group(5).strip()
            label = match.group(6).strip()
            
            # Flat step for FSM
            flat_steps.append({
                'id': step_id,
                'phase': current_phase_name,
                'action': action,
                'subject': subject,
                'object': obj,
                'constraint': constraint,
                'label': label
            })
            
            # Step for Process Report
            if current_process:
                step_desc = f"{action}{obj}"
                if constraint and constraint != "none":
                    step_desc += f" ({constraint})"
                
                current_process["steps"].append({
                    "step_id": int(step_id),
                    "description": step_desc,
                    "passed": "no" # Default
                })
                # Append to SOP text (reconstructed)
                current_process["sop_text"] += f"{step_desc}; "

    # Append the last process
    if current_process:
        processes.append(current_process)
        
    return flat_steps, processes


def main():
    print("Generating LabFSM from raw steps...")
    
    # 1. Get Standardized Steps
    print("Calling Qwen3-max API...")
    # Use DEFAULT_RAW_STEPS or read from file
    raw_steps = DEFAULT_RAW_STEPS
    
    llm_output = get_standardized_steps(raw_steps)
    if not llm_output:
        print("Failed to get response from LLM.")
        return
        
    print("Received standardized steps:")
    print(llm_output)
    print("-" * 40)
    
    # 2. Parse Steps
    flat_steps, processes = parse_steps(llm_output)
    print(f"Parsed {len(flat_steps)} steps and {len(processes)} processes.")
    
    # 3. Save to JSON
    # Save flat steps for FSM
    json_output_file = "steps.json"
    with open(json_output_file, "w", encoding="utf-8") as f:
        json.dump(flat_steps, f, ensure_ascii=False, indent=4)
    print(f"Flat steps saved to {json_output_file}")
    
    # Save hierarchical processes for Report
    process_output_file = "process_structure.json"
    with open(process_output_file, "w", encoding="utf-8") as f:
        json.dump(processes, f, ensure_ascii=False, indent=4)
    print(f"Process structure saved to {process_output_file}")
    
if __name__ == "__main__":
    main()
