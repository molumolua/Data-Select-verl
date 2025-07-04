from datasets import load_dataset
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import pandas as pd
# from verl.utils.reward_score.math import *
# from verl.utils.reward_score.math_dapo import normalize_final_answer
import json
import random
def load_json(path):
    with open (path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data
def load_jsonl(path):
    with open (path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data
def extract_answer(solution,extract_type="TEST"):
    # verl/verl/utils/reward_score/__init__.py
    answer=""
    if extract_type.startswith("think"):
        return {
            "ground_truth": solution# use raw solution and extract answer later in reward model
        }
   
    string_in_last_boxed = last_boxed_only_string(solution)
    if string_in_last_boxed is not None:
        if extract_type == "TEST":  # extract ground truth for math compute score
            answer = remove_boxed(string_in_last_boxed)
        elif extract_type == "dapo":  # extract ground truth for math_dapo compute score
            answer = normalize_final_answer(string_in_last_boxed)
    return {
        "ground_truth": answer
        }
def prepare_question(question,prompt_type="TEST"):
    prompt_list = []
    # Escape curly braces to avoid conflicts with format method
    if prompt_type == "dapo":
        user_prompt = f"Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n{question}\n\nRemember to put your answer on its own line after \"Answer:\""
        prompt_list.append(
            {"content": user_prompt, "role": "user"}
        )
    elif prompt_type.startswith("think"):
        system_prompt = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer, and put your final answer within \\boxed{{}} . The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>." 
        prompt_list.append(
            {"content":system_prompt, "role": "system"}
        )
        prompt_list.append(
            {"content": question, "role": "user"}
        )
    elif prompt_type == "TEST":
        system_prompt = "Please reason step by step, and put your final answer within \\boxed{{}}."
        prompt_list.append(
            {"content":system_prompt, "role": "system"}
        )
        prompt_list.append(
            {"content": question, "role": "user"}
        )

    return prompt_list

if __name__ == "__main__":
    keep_size=1000
    data_source = f"think_mui_math_{keep_size}"  # "think_MATH-500" or "dapo_MATH-500"
    repeat_time= 1
    random.seed(1)  # For reproducibility
    # 加载数据集的 train 部分
    # dataset = load_dataset("DigitalLearningGmbH/MATH-lighteval",split="train")
    # save_path = f"/data2/xucaijun/DAPO_verl/verl/data/think-MATH-{keep_size}.parquet"
    # dataset = load_dataset("HuggingFaceH4/MATH-500",split="test")
    # save_path = f"/data2/xucaijun/DAPO_verl/verl/data/{data_source}_{keep_size}_MATH-500-processed.parquet"
    # dataset = load_json("/data2/xucaijun/My-Math-Generator/outputs/7b_rejected_first_filter_1-4.json")
    # save_path = f"/data2/xucaijun/verl/data/{data_source}_7b_rejected_first_filter_1-4.parquet"
    # dataset = load_jsonl("/data2/xucaijun/My-Math-Generator/data/amc23/test.jsonl")
    # save_path = f"/data2/xucaijun/DAPO_verl/verl/data/{data_source}_amc23_test.parquet"

    # dataset =load_jsonl("/data2/xucaijun/My-Math-Generator/data/aime24/test.jsonl")
    # save_path = f"/data2/xucaijun/DAPO_verl/verl/data/{data_source}_aime24_test.parquet"

    # df = pd.read_parquet(f"/data2/xucaijun/DAPO_verl/verl/data/think-DeepMath-103K.parquet")
    # dataset = df.to_dict(orient="records") 

    dataset = load_jsonl(f"/data2/xucaijun/MUI-Eval/neuron_and_sae/greedy_sample/qwen2.5-7b/math_topk_{keep_size}.jsonl")
    # dataset = load_dataset("Open-Reasoner-Zero/orz_math_72k_collection_extended",split="train")
    save_path = f"/data2/xucaijun/DAPO_verl/verl/data/think-mui-MATH-{keep_size}.parquet"

    processed_data = []

    for i,item in enumerate(dataset):
        #question = item['problem']  # Extract the problem
        # solution = item['solution']  # Extract the solution
        # solution = f"\\boxed{{{item['answer']}}}"  # Add boxed to the solution
        # Generate the prompt and answer
        # question = item['0']["value"]
        # solution = f"\\boxed{{{item['1']['ground_truth']['value']}}}"

        question = item['question'][68:-11]
        solution = f"\\boxed{{{item['answer']}}}"


        prompt = prepare_question(question,prompt_type=data_source)
        answer = extract_answer(solution,extract_type=data_source)
        if i ==0 :
            print(f"Prompt: {prompt}")
            print(f"Answer: {answer}")
        if len(answer['ground_truth'])==0:
            print(f"Error: No ground truth found for solution: {solution}")
            continue
        # Append to the processed data
        processed_data.append({
            'prompt': prompt,
            'reward_model': answer,
            "data_source": data_source,
            "extra_info":{"index": i},
            "metric_list": [],
            "step_list":[],
            "entropy_list":[],
            "metric_sum":0
        })

    # dataset = load_json("/data2/xucaijun/DAPO_verl/verl/data/orz_math_57k_collection.json")

    # for i,item in enumerate(dataset):
    #     #question = item['problem']  # Extract the problem
    #     # solution = item['solution']  # Extract the solution
    #     # solution = f"\\boxed{{{item['answer']}}}"  # Add boxed to the solution
    #     # Generate the prompt and answer
    #     question = item[0]["value"]
    #     solution = f"\\boxed{{{item[1]['ground_truth']['value']}}}"
    #     prompt = prepare_question(question,prompt_type=data_source)
    #     answer = extract_answer(solution,extract_type=data_source)
    #     if i ==0 :
    #         print(f"Prompt: {prompt}")
    #         print(f"Answer: {answer}")
    #     if len(answer['ground_truth'])==0:
    #         print(f"Error: No ground truth found for solution: {solution}")
    #         continue
    #     # Append to the processed data
    #     processed_data.append({
    #         'prompt': prompt,
    #         'reward_model': answer,
    #         "data_source": data_source,
    #         "extra_info":{"index": i},
    #         "metric_list": [],
    #         "step_list":[],
    #         "entropy_list":[],
    #         "metric_sum":0
    #     })

    if keep_size:
        random.shuffle(processed_data)
        processed_data = processed_data[:keep_size]

    processed_data = processed_data * repeat_time
    processed_df = pd.DataFrame(processed_data)
    processed_df.to_parquet(save_path)

    print(f"Processed {len(processed_data)} dataset saved as Parquet.")
