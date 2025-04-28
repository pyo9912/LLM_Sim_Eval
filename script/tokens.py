import transformers
from transformers import AutoTokenizer
import json
import os
import argparse


BASE_DIR = "/home/user/junpyo/iEvaLM-CRS-main/save_5/output"

def load_json(file_path):
    """JSON 파일 로드"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, file_path):
    """JSON 데이터 저장"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_json_files_for_model(model, timelog, topk):
    """주어진 모델의 /output/{model}/ 경로에서 모든 turn(1~5) 파일을 읽어옴"""
    all_turns = {}
    success_turns = {}
    fail_turns = {}
    topk = topk
    timelog = timelog

    model_dir = os.path.join(BASE_DIR, model, timelog)
    
    turn = 1
    turn_dir = os.path.join(model_dir, str(turn))
    if not os.path.exists(turn_dir):
        print(f"Skipping {turn_dir}: Directory does not exist")

    all_file = os.path.join(turn_dir, f"all_samples_{model}_in_turn_{turn}.json")
    success_file = os.path.join(turn_dir, f"success_{topk}_samples_{model}_in_turn_{turn}.json")
    fail_file = os.path.join(turn_dir, f"fail_{topk}_samples_{model}_in_turn_{turn}.json")

    all_turns[turn] = load_json(all_file) if os.path.exists(all_file) else []
    success_turns[turn] = load_json(success_file) if os.path.exists(success_file) else []
    fail_turns[turn] = load_json(fail_file) if os.path.exists(fail_file) else []
    
    return all_turns, success_turns, fail_turns

def calculate_token_length(model_name, tokenizer_name, timelog, topk):
    """각 조건에 따른 평균 토큰 길이를 계산"""
    topk = topk
    timelog = timelog
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    all_turns, success_turns, fail_turns = load_json_files_for_model(model_name, timelog, topk)
    total = success_turns[1] + fail_turns[1]

    # 6가지 조건별 토큰 길이를 저장할 리스트
    lengths = {
        "given_turn": [],
        "simulated_turn": [],
        "total_length": [],
        "avg_length": [],
        "crs_avg_length": [],
        "user_avg_length": [],
        "simulator_avg_length": [],
        "simulator_crs_avg_length": [],
        "simulator_user_avg_length": []
    }

    for data in total:
        if not data:
            continue

        # for sample in data:
        for idx, context in enumerate(data['context']):
            tokenized = tokenizer(context['content'], truncation=False, padding=False, return_tensors="pt")
            token_length = tokenized.input_ids.shape[1]  # 토큰 개수
            
            # 샘플 당 turn 수 총합
            if idx < data['turn_id']:
                lengths["given_turn"].append(1)
            else:
                lengths['simulated_turn'].append(1)

            lengths["total_length"].append(token_length)

            # 샘플 당 token 수 총합
            if idx < data['turn_id']:
                lengths["avg_length"].append(token_length)
            else:
                lengths["simulator_avg_length"].append(token_length)

            # CRS token 수 총합
            if context['role'] == 'assistant' and 'entity' in context:
                lengths["simulator_crs_avg_length"].append(token_length)

            # USER token 수 총합
            if context['role'] == 'user' and 'entity' in context:
                lengths["simulator_user_avg_length"].append(token_length)

    # 평균 계산 함수 (빈 리스트일 경우 0 반환)
    def avg(lst):
        return sum(lst) / len(lst) if lst else 0

    # 결과 출력
    print(f"모델 {model_name}의 평균 발화 길이:")
    given_turn = sum(lengths['given_turn'])/len(total)
    simulated_turn = sum(lengths['simulated_turn'])/len(total)
    given_input_length = ((sum(lengths['avg_length'])/len(total))*(given_turn+simulated_turn))/2
    simulated_input_length = ((sum(lengths['simulator_avg_length'])/len(total))*simulated_turn)/2
    print(f"0. 평균 given 턴 수: {given_turn:.2f}")
    print(f"0. 평균 simulated 턴 수: {simulated_turn:.2f}")
    print(f"1. 평균 given input 길이: {given_input_length:.2f}")
    print(f"1. 평균 simulated input 길이: {simulated_input_length:.2f}")
    print(f"1. 평균 input 길이: {(given_input_length + simulated_input_length):.2f}")
    print(f"2. Simulator 샘플 당 발화 길이: {sum(lengths['simulator_avg_length'])/len(total):.2f}")
    print(f"2. Simulator CRS 샘플 당 발화 길이: {sum(lengths['simulator_crs_avg_length'])/len(total):.2f}")
    print(f"2. Simulator User 샘플 당 발화 길이: {sum(lengths['simulator_user_avg_length'])/len(total):.2f}")
    print(f"2. 평균 output 길이: {(sum(lengths['simulator_user_avg_length'])/(len(total))):.2f}")



if __name__ == '__main__':
    # model_name = 'chatgpt_our_prompt_topk_10'
    tokenizer_name = 'bert-base-uncased'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--topk', type=int)
    parser.add_argument('--timelog', type=str)
    args = parser.parse_args()
    topk = args.topk
    timelog = args.timelog
    model_name = args.model  # 사용할 모델 지정
    calculate_token_length(model_name, tokenizer_name, timelog=timelog, topk=topk)
