import os
import json
import torch
import re
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ================= 配置区域 =================

# 1. 输入数据路径 (FACED 数据集文件夹)
# 🔧 请修改为你的帧图像目录
INPUT_ROOT = r"./FACED_frames"

# 2. 输出保存路径 (建议修改一下输出文件夹名，以区分之前的带有情感先验的数据)
OUTPUT_ROOT = r"./features/image_describe_Qwen_to_text_objective"

# 3. 模型权重保存路径
CACHE_DIR = r"./huggingface_cache"

# 4. 模型 ID
MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"

# 5. 🌟 核心修改：强调客观、去主观化的 System Prompt
SYSTEM_PROMPT = """You are an expert Visual Content Analyst. 
Task: Provide a highly accurate, objective description of the physical and visual elements in the provided movie frame. 
Write a SINGLE, fluent natural language paragraph (< 50 words).

CRITICAL RULES:
1. NATURAL LANGUAGE ONLY: Do NOT use plus signs (+), brackets ([]), parentheses (), or symbols. Write in continuous, natural English prose.
2. BE COMPLETELY OBJECTIVE: Describe exactly what is physically visible: the characters (appearance, posture, clear actions), the environment, lighting, and colors. Do NOT infer hidden emotions, thoughts, or plot context.
3. NO HALLUCINATION: Only describe what is physically visible. If the scene is dark, blurry, or lacks people, describe it as such. Do not invent objects or feelings.
"""


def smart_truncate_and_clean(text, max_words=55):
    """
    清洗生成的文本，剔除特殊符号，确保为纯自然语言。
    """
    text = text.replace("\n", " ").strip()

    # 如果模型偶尔还是加了 "Image shows:" 之类的前缀，去掉它
    if ":" in text[:25]:
        text = text.split(":", 1)[1].strip()

    # 强制剔除所有干扰符号
    text = text.replace('+', ',').replace('[', '').replace(']', '').replace('(', '').replace(')', '').replace('*', '')
    text = re.sub(r'\s+', ' ', text)
    text = text.replace(' ,', ',').replace(' .', '.')

    # 截断控制
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "."


# ================= 主函数 =================

def main():
    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)

    # 1. 加载全血版 Qwen2-VL (bfloat16)
    print(f"🚀 正在加载全血版 Qwen2-VL (bfloat16)...")
    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="auto",
            cache_dir=CACHE_DIR
        )
        processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            min_pixels=256 * 28 * 28,
            max_pixels=1024 * 28 * 28,  # 限制像素防止 OOM
            cache_dir=CACHE_DIR
        )
        print("✅ 模型加载成功！")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 2. 遍历视频文件夹
    video_folders = [f for f in os.listdir(INPUT_ROOT) if os.path.isdir(os.path.join(INPUT_ROOT, f))]
    video_folders.sort()

    for video_name in tqdm(video_folders, desc="Processing FACED Videos (Objective)"):
        video_path = os.path.join(INPUT_ROOT, video_name)
        output_json_path = os.path.join(OUTPUT_ROOT, f"{video_name}.json")

        if os.path.exists(output_json_path):
            continue

        results = []
        frame_files = [f for f in os.listdir(video_path) if "middle.jpg" in f]
        frame_files.sort()

        for frame_file in frame_files:
            image_abs_path = os.path.join(video_path, frame_file)

            # 🌟 核心修改：去除了情感暗示，直接要求客观描述
            user_prompt_text = (
                "Describe the specific visual elements in this frame objectively. "
                "Focus on the characters' physical state, the environment, and the lighting."
            )

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image", "image": image_abs_path},
                    {"type": "text", "text": user_prompt_text},
                ]}
            ]

            # 推理流程
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
            ).to("cuda")

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=96,
                    repetition_penalty=1.1,
                    do_sample=True,
                    temperature=0.7
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            description = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            # 最终清洗
            final_desc = smart_truncate_and_clean(description)

            results.append({
                "file_name": frame_file,
                "sec_id": frame_file.split("_")[0],
                "description": final_desc
            })

        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"✅ 客观图像描述提取完成！结果保存在: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()