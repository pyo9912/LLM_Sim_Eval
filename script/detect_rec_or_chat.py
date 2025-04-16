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


def write_state(dialog_list):
    chat_list = []
    rec_words = ["i would recommend", "i recommend", "recommend you", "here are some", "you should watch", "is good movie", "is a great", "have you seen", "how about", "you might enjoy", "you might also enjoy", "you might like", "you might also like"]
    pattern_1 = r'\d+\.\s+\w+' # movie(2000)
    pattern_2 = r'"[^"]+"' # "movie(2000)"
    pattern_3 = r'[^>]+\s\(\d+\)\s*>\s*[^>]+\s\(\d+\)' # movie(2000) > movie(2001)
    pattern_4 = r'[^>]+\s\(\d+\)\s*>\s*[^,]+\s\(\d+\)' # movie(2000) , movie(2001)

    for dialog in dialog_list:
        context = dialog.get("context", [])
        turn_cnt = 0
        state_list = []
        
        for idx, turn in enumerate(context):
            if "rec_items" in turn:
                turn_cnt += 1
                # 매 turn 별 rec_response chat_response 측정
                next_user_uttr = context[idx+1]
                user_intent = next_user_uttr['intent'].split('intent: ')[-1]
                # content = turn.get("content", "").lower()
                # 추천 여부 판단 (추천 문구 포함 여부 확인)
                if 'inquiry' not in user_intent.lower() or 'i would recommend' in turn['content'].lower():
                    state_list.append("rec")
                else:
                    state_list.append("chat")
        dialog.update({"state_list":state_list})
    return dialog_list

def compute_recall(pred_list, label, k):
        return int(label in pred_list[:k])

def write_recall(dialog_list):
    def compute_recall(pred_list, label, k):
        return int(label in pred_list[:k])
    
    for dialog in dialog_list:
        rec_1_result_list = []
        rec_10_result_list = []
        rec_50_result_list = []
        for turn in dialog.get("context", []):
            rec_items = turn.get("rec_items", [])  # 추천된 아이템 목록
            rec_labels = dialog.get("rec_label", [])  # 실제 정답 레이블 목록
            
            if turn['role'] == 'user' or 'rec_items' not in turn:
                continue

            if rec_items:
                turn["rec_success@1"] = False
                turn["rec_success@10"] = False
                turn["rec_success@50"] = False

            rec_success = False
            for rec_label in rec_labels:
                if rec_label in rec_items[:1]:
                    turn["rec_success@1"] = True
                    turn["rec_success@10"] = True
                    turn["rec_success@50"] = True
                    rec_1_result_list.append(True)
                    rec_10_result_list.append(True)
                    rec_50_result_list.append(True)
                    rec_success = True
                    break  # 더 이상 확인할 필요 없음
                elif rec_label in rec_items[:10]:
                    turn["rec_success@10"] = True
                    turn["rec_success@50"] = True
                    rec_1_result_list.append(False)
                    rec_10_result_list.append(True)
                    rec_50_result_list.append(True)
                    rec_success = True
                    break
                elif rec_label in rec_items[:50]:
                    turn["rec_success@50"] = True
                    rec_1_result_list.append(False)
                    rec_10_result_list.append(False)
                    rec_50_result_list.append(True)
                    rec_success = True
                    break
            if not rec_success:
                rec_1_result_list.append(False)
                rec_10_result_list.append(False)
                rec_50_result_list.append(False)
        dialog.update({"rec_success@1":rec_1_result_list})    
        dialog.update({"rec_success@10":rec_10_result_list})    
        dialog.update({"rec_success@50":rec_50_result_list})    
    return dialog_list

def detect_rec_or_chat(dialog_list, turn_accuracy):
    rec_list = []
    chat_list = []

    for dialog in dialog_list:
        context = dialog.get("context", [])
        turn_cnt = 0
        for idx, turn in enumerate(context):
            if "rec_items" in turn:
                turn_cnt += 1
                # 매 turn 별 rec_response chat_response 측정
                if turn_cnt == turn_accuracy:
                    next_user_uttr = context[idx+1]
                    user_intent = next_user_uttr['intent'].split('intent: ')[-1]
                    # content = turn.get("content", "").lower()
                    # 추천 여부 판단 (추천 문구 포함 여부 확인)
                    if 'inquiry' not in user_intent.lower() or 'i would recommend' in turn['content'].lower():
                        rec_list.append(dialog)
                    else:
                        chat_list.append(dialog)
    # print(f"rec_cnt: {len(rec_list)}")
    # print(f"chat_cnt: {len(chat_list)}")
    # print(f"all_cnt: {len(rec_list)+len(chat_list)}")
    return rec_list, chat_list

def detect_rec_or_chat_prev(dialog_list, turn_accuracy):
    rec_list = []
    chat_list = []
    rec_words = ["i would recommend", "i recommend", "recommend you", "here are some", "you should watch", "is good movie", "is a great", "have you seen", "how about", "you might enjoy", "you might also enjoy", "you might like", "you might also like"]
    pattern_1 = r'\d+\.\s+\w+' # movie(2000)
    pattern_2 = r'"[^"]+"' # "movie(2000)"
    pattern_3 = r'[^>]+\s\(\d+\)\s*>\s*[^>]+\s\(\d+\)' # movie(2000) > movie(2001)
    pattern_4 = r'[^>]+\s\(\d+\)\s*>\s*[^,]+\s\(\d+\)' # movie(2000) , movie(2001)

    for dialog in dialog_list:
        context = dialog.get("context", [])
        turn_cnt = 0
        for turn in context:
            if "rec_items" in turn:
                turn_cnt += 1
                # 매 turn 별 rec_response chat_response 측정
                if turn_cnt == turn_accuracy:
                    content = turn.get("content", "").lower()
                    rec_word_check = False
                    # rec_words 가 포함된 경우
                    for rw in rec_words:
                        if rw in content:
                            rec_word_check = True
                            break 
                    # 추천 여부 판단 (추천 문구 포함 여부 확인)
                    if (rec_word_check or
                        re.search(pattern_1, content) or
                        re.search(pattern_2, content) or
                        re.search(pattern_3, content) or
                        re.search(pattern_4, content)):
                        rec_list.append(dialog)
                    
                    else:
                        chat_list.append(dialog)
                    break
    # print(f"rec_cnt: {len(rec_list)}")
    # print(f"chat_cnt: {len(chat_list)}")
    # print(f"all_cnt: {len(rec_list)+len(chat_list)}")
    return rec_list, chat_list
        
def write_state_for_turn_samples(model, timelog, topk):
    """모델별로 turn 데이터를 처리하고 저장"""
    topk = topk
    timelog = timelog
    all_turns, success_turns, fail_turns = load_json_files_for_model(model, topk)

    for turn in range(1, 6):
        if turn not in all_turns:
            continue  # 해당 turn이 없으면 건너뜀

        # 전체 데이터 처리
        new_dialog_list = write_state(all_turns[turn])
        # new_dialog_list = write_recall(new_dialog_list)
        save_json(new_dialog_list, os.path.join(BASE_DIR, model, timelog, str(turn), f"all_{topk}_samples_{model}_in_turn_{turn}.json"))

        # 성공 데이터 처리
        new_dialog_list = write_state(success_turns[turn])
        # new_dialog_list = write_recall(new_dialog_list)
        save_json(new_dialog_list, os.path.join(BASE_DIR, model, timelog, str(turn), f"success_{topk}_samples_{model}_in_turn_{turn}.json"))

        # 실패 데이터 처리
        new_dialog_list = write_state(fail_turns[turn])
        # new_dialog_list = write_recall(new_dialog_list)
        save_json(new_dialog_list, os.path.join(BASE_DIR, model, timelog, str(turn), f"fail_{topk}_samples_{model}_in_turn_{turn}.json"))

def detect_rec_or_chat_for_model(model, timelog, topk):
    """주어진 모델의 데이터를 rec/chat으로 분류"""
    topk=topk
    timelog = timelog
    rec_turns, success_turns, fail_turns = load_json_files_for_model(model, timelog, topk)

    print(f"\nProcessing Model: {model}")
    
    print("\nAll Samples")
    for turn in range(1, 6):
        if turn in rec_turns:
            detect_rec_or_chat(rec_turns[turn], turn)

    print("\nSuccess Samples")
    for turn in range(1, 6):
        if turn in success_turns:
            detect_rec_or_chat(success_turns[turn], turn)

    print("\nFail Samples")
    for turn in range(1, 6):
        if turn in fail_turns:
            detect_rec_or_chat(fail_turns[turn], turn)
            

def write_state_for_turn_samples(model, timelog, topk):
    """모델별로 turn 데이터를 처리하고 저장"""
    topk = topk
    all_turns, success_turns, fail_turns = load_json_files_for_model(model, timelog, topk)

    for turn in range(1, 6):
        if turn not in all_turns:
            continue  # 해당 turn이 없으면 건너뜀

        # 전체 데이터 처리
        new_dialog_list = write_state(all_turns[turn])
        # new_dialog_list = write_recall(new_dialog_list)
        save_json(new_dialog_list, os.path.join(BASE_DIR, model, timelog, str(turn), f"all_{topk}_samples_{model}_in_turn_{turn}.json"))

        # 성공 데이터 처리
        new_dialog_list = write_state(success_turns[turn])
        # new_dialog_list = write_recall(new_dialog_list)
        save_json(new_dialog_list, os.path.join(BASE_DIR, model, timelog, str(turn), f"success_{topk}_samples_{model}_in_turn_{turn}.json"))

        # 실패 데이터 처리
        new_dialog_list = write_state(fail_turns[turn])
        # new_dialog_list = write_recall(new_dialog_list)
        save_json(new_dialog_list, os.path.join(BASE_DIR, model, timelog, str(turn), f"fail_{topk}_samples_{model}_in_turn_{turn}.json"))

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--topk', type=int)
    parser.add_argument('--timelog', type=str)
    args = parser.parse_args()
    topk = args.topk
    model = args.model  # 사용할 모델 지정
    timelog = args.timelog

    detect_rec_or_chat_for_model(model, timelog, topk=topk)
    write_state_for_turn_samples(model, timelog, topk=topk)
    # count_avg_state_for_turn_samples(model, timelog, topk=topk)
    # count_state(model, timelog, topk=topk)


