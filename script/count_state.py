import json
import re
from collections import Counter
import os
from copy import copy
import argparse
from itertools import product

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
    timelog = timelog
    topk = topk

    model_dir = os.path.join(BASE_DIR, model, timelog)
    
    for turn in range(1, 6):  # 1~5 턴 폴더 탐색
        turn_dir = os.path.join(model_dir, str(turn))
        if not os.path.exists(turn_dir):
            print(f"Skipping {turn_dir}: Directory does not exist")
            continue

        all_file = os.path.join(turn_dir, f"all_samples_{model}_in_turn_{turn}.json")
        success_file = os.path.join(turn_dir, f"success_{topk}_samples_{model}_in_turn_{turn}.json")
        fail_file = os.path.join(turn_dir, f"fail_{topk}_samples_{model}_in_turn_{turn}.json")

        all_turns[turn] = load_json(all_file) if os.path.exists(all_file) else []
        success_turns[turn] = load_json(success_file) if os.path.exists(success_file) else []
        fail_turns[turn] = load_json(fail_file) if os.path.exists(fail_file) else []
    
    return all_turns, success_turns, fail_turns

def load_processed_json_files_for_model(model, timelog, topk):
    """주어진 모델의 /output/{model}/ 경로에서 모든 turn(1~5) 파일을 읽어옴"""
    all_turns = {}
    success_turns = {}
    fail_turns = {}
    topk = topk
    timelog = timelog

    model_dir = os.path.join(BASE_DIR, model, timelog)
    
    for turn in range(1, 6):  # 1~5 턴 폴더 탐색
        turn_dir = os.path.join(model_dir, str(turn))
        if not os.path.exists(turn_dir):
            print(f"Skipping {turn_dir}: Directory does not exist")
            continue

        all_file = os.path.join(turn_dir, f"all_{topk}_samples_{model}_in_turn_{turn}.json")
        success_file = os.path.join(turn_dir, f"success_{topk}_samples_{model}_in_turn_{turn}.json")
        fail_file = os.path.join(turn_dir, f"fail_{topk}_samples_{model}_in_turn_{turn}.json")

        all_turns[turn] = load_json(all_file) if os.path.exists(all_file) else []
        success_turns[turn] = load_json(success_file) if os.path.exists(success_file) else []
        fail_turns[turn] = load_json(fail_file) if os.path.exists(fail_file) else []
    
    return all_turns, success_turns, fail_turns

def count_state(model, timelog, topk):
    timelog = timelog
    topk = topk
    all_turns, success_turns, fail_turns = load_processed_json_files_for_model(model, timelog, topk)
    total = success_turns[1] + fail_turns[1]
    success_total = success_turns[1] + success_turns[2] + success_turns[3] + success_turns[4] + success_turns[5]

    max_turn = 5  # 최대 턴 수
    state_patterns = {i: [] for i in range(1, max_turn + 1)}
    
    # 실제 등장한 state_list 저장
    for dialog in total:
        state_list = tuple(dialog.get("state_list", []))
        if state_list:
            state_patterns[len(state_list)].append(state_list)
    
    # 가능한 모든 조합 생성
    all_combinations = {i: [] for i in range(1, max_turn + 1)}
    for turn in range(1, max_turn + 1):
        all_combinations[turn] = [combo + ('rec',) for combo in product(['rec', 'chat'], repeat=turn - 1)]
    
    # 패턴 개수 세기
    pattern_counts = {turn: Counter(state_patterns[turn]) for turn in range(1, max_turn + 1)}
    
    print("State Pattern Counts:")
    
    # 가능한 모든 조합을 출력 (등장하지 않은 조합도 포함하여 0으로 출력)
    for turn in range(1, max_turn + 1):
        for pattern in all_combinations[turn]:
            count = pattern_counts[turn].get(pattern, 0)
            print(f"{', '.join(pattern)}: {count}")
    
    return pattern_counts

def count_state_prev(model, timelog, topk):
    timelog = timelog
    topk = topk
    all_turns, success_turns, fail_turns = load_processed_json_files_for_model(model, timelog, topk)

    state_patterns = []
    
    for dialog in all_turns[1]:
        state_list = tuple(dialog.get("state_list", []))  # 튜플로 변환하여 리스트 패턴을 키로 사용 가능하게 함
        if state_list:
            state_patterns.append(state_list)
    
    pattern_counts = Counter(state_patterns)  # 각 패턴의 개수를 세기
    
    print("State Pattern Counts:")
    
    # 단일 상태 패턴 출력
    for pattern, count in sorted(pattern_counts.items(), key=lambda x: len(x[0])):
        print(f"{', '.join(pattern)}: {count}")
    
    return pattern_counts


def count_avg_state_for_turn_samples(model, timelog, topk):
    """모델별로 turn 데이터를 보고 state (rec/chat) 의 평균 개수 카운트"""
    topk = topk
    all_turns, success_turns, fail_turns = load_json_files_for_model(model, timelog, topk)

    for turn in range(1, 6):
        if turn not in success_turns:
            continue
        
        cnt, rec_cnt, chat_cnt = 0.0, 0.0, 0.0
        success_samples = success_turns[turn]

        len_history_state_list = turn - 1
        for sample in success_samples:
            if len_history_state_list == 0:
                break
            else:
                history_state_list = sample['state_list'].copy()
                history_state_list = history_state_list[:-1]

                for history in history_state_list:
                    if history == 'rec':
                        rec_cnt += 1.0
                    elif history == 'chat':
                        chat_cnt += 1.0
                    cnt += 1.0
        
        if len_history_state_list == 0:
            print(f"turn: {turn}\nrec_avg: {rec_cnt}\nchat_avg: {chat_cnt}")
        else:
            print(f"turn: {turn}\nrec_avg: {(rec_cnt)}\nchat_avg: {(chat_cnt)}")
            # print(f"turn: {turn}\nrec_avg: {(rec_cnt/cnt)}\nchat_avg: {(chat_cnt/cnt)}")
            # print(f"turn: {turn}\nrec_avg: {(rec_cnt/cnt)*len_history_state_list}\nchat_avg: {(chat_cnt/cnt)*len_history_state_list}")
            
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--topk', type=int)
    parser.add_argument('--timelog', type=str)
    args = parser.parse_args()
    topk = args.topk
    model = args.model  # 사용할 모델 지정
    timelog = args.timelog

    # detect_rec_or_chat_for_model(model, timelog, topk=topk)
    # write_state_for_turn_samples(model, timelog, topk=topk)
    # count_avg_state_for_turn_samples(model, timelog, topk=topk)
    count_state(model, timelog, topk=topk)


