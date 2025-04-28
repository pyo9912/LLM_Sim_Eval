import json
import argparse
import re
import os
from copy import copy
from tqdm import tqdm

import sys
sys.path.append("..")

from src.model.metric import RecMetric

# datasets = ['redial_eval', 'opendialkg_eval']
datasets = ['redial_eval']
# models = ['kbrd', 'barcor', 'unicrs', 'chatgpt']
# models = ['kbrd', 'unicrs']
# models = ['chatgpt_our_prompt_topk_10']
# models = ['chatgpt-4o-7']
models = []


# compute rec recall
def rec_eval(args, turn_num, mode, turn_accuracy, topk, timelog):
    # 250227 JP sample 저장용
    for dataset in datasets:
        with open(f"../data/{dataset.split('_')[0]}/entity2id.json", 'r', encoding="utf-8") as f:
            entity2id = json.load(f)
        
        for model in models:
            all_samples = []
            turn_samples = []
            success_1_samples, fail_1_samples = [], []
            success_10_samples, fail_10_samples = [], []
            success_50_samples, fail_50_samples = [], []
            metric = RecMetric([1, 10, 25, 50])
            persuatiness = 0
            save_path = f"/home/user/junpyo/iEvaLM-CRS-main/save_{turn_num}/{mode}/{model}/{dataset}/{topk}/{timelog}" # data loaded path
            result_path = f"/home/user/junpyo/iEvaLM-CRS-main/save_{turn_num}/result/{mode}/{model}/{timelog}"
            os.makedirs(result_path, exist_ok=True)
            if os.path.exists(save_path) and len(os.listdir(save_path)) > 0:
                path_list = os.listdir(save_path)
                print(f"turn_num: {turn_num}, mode: {mode} model: {model} dataset: {dataset}", len(path_list))
                
                for path in tqdm(path_list):
                    with open(f"{save_path}/{path}", 'r', encoding="utf-8") as f:
                        data = json.load(f)
                        # if mode == 'chat':
                            # persuasiveness_score = data['persuasiveness_score']
                            # persuatiness += float(persuasiveness_score)
                        PE_dialog = data['simulator_dialog']
                        rec_label = data['rec']
                        rec_label = [entity2id[rec] for rec in rec_label if rec in entity2id]
                        contexts = PE_dialog['context']
                        # Original iEvaLM setting --> 가장 최근 대화의 추천 정확도 == 최종 정확도
                        for r in rec_label:
                            rec = [r]

                            if turn_accuracy == 0:
                                for context in contexts[::-1]:
                                    if 'rec_items' in context:
                                        if 'ieval' in model:
                                            rec_items = context['rec_items']
                                        else:
                                            rec_items = [entity2id[i] for i in context['rec_items'] if i in entity2id]
                                        metric.evaluate(rec_items, rec)
                                        output = data['simulator_dialog'].copy()
                                        output.update({"rec_label":rec})
                                        output.update({"rec_labels":rec_label})
                                        all_samples.append(output)
                                        break
                            # 250226 JP --> turn 별 추천 정확도 확인용
                            else:
                                turn_cnt = 0
                                check = False   # turn_accuracy 보다 simulated turn이 적은 경우 탐지용
                                for idx, context in enumerate(contexts):
                                    if 'rec_items' in context:
                                        next_user_uttr = contexts[idx+1]
                                        user_intent = next_user_uttr['intent'].split('intent: ')[-1]
                                        turn_cnt += 1
                                        if turn_cnt == turn_accuracy:
                                            if 'inquiry' not in user_intent.lower():
                                                rec_items = [entity2id[i] for i in context['rec_items'] if i in entity2id]
                                                # if 'inquiry' not in user_intent.lower():
                                                metric.evaluate(rec_items, rec)
                                                    # check = True
                                                recall_1 = int(any(item in rec for item in rec_items[:1]))
                                                recall_10 = int(any(item in rec for item in rec_items[:10]))
                                                recall_50 = int(any(item in rec for item in rec_items[:50]))
                                                
                                                output = data['simulator_dialog'].copy()
                                                output.update({"rec_label": rec})
                                                output.update({"rec_labels":rec_label})
                                                
                                                if topk==1:
                                                    if recall_1 > 0:
                                                        success_1_samples.append(output)
                                                    else:
                                                        fail_1_samples.append(output)
                                                elif topk==10:
                                                    if recall_10 > 0:
                                                        success_10_samples.append(output)
                                                    else:
                                                        fail_10_samples.append(output)
                                                elif topk==50:
                                                    # if context.get('rec_success', False):
                                                    if recall_50 > 0:
                                                        success_50_samples.append(output)
                                                    else:
                                                        fail_50_samples.append(output)
                                                
                                                turn_samples.append(output)
                                                check = True
                                                break
                                            else:
                                                output = data['simulator_dialog'].copy()
                                                output.update({"rec_label": rec})
                                                output.update({"rec_labels":rec_label})
                                                if topk==1:
                                                    fail_1_samples.append(output)
                                                elif topk==10:
                                                    fail_10_samples.append(output)
                                                elif topk==50:
                                                    fail_50_samples.append(output)
                                if not check:
                                    metric.evaluate([], rec)
                        
                report = metric.report()
                
                print('r1:', f"{report['recall@1']:.3f}", 'r10:', f"{report['recall@10']:.3f}", 'r25:', f"{report['recall@25']:.3f}", 'r50:', f"{report['recall@50']:.3f}", 'count:', report['count'])
                if mode == 'chat':
                    persuativeness_score = persuatiness / len(path_list)
                    print(f"{persuativeness_score:.3f}")
                    report['persuativeness'] = persuativeness_score
                
                with open(f"{result_path}/{dataset}.json", 'w', encoding="utf-8") as w:
                    w.write(json.dumps(report))
    
            output_dir = f"/home/user/junpyo/iEvaLM-CRS-main/save_{turn_num}/output/{model}/{timelog}/{turn_accuracy}"
            os.makedirs(output_dir, exist_ok=True)
            all_samples_path = f"/home/user/junpyo/iEvaLM-CRS-main/save_{turn_num}/output/all_samples_train_{model}.json"
            turn_samples_path = f"{output_dir}/all_samples_{model}_in_turn_{turn_accuracy}.json"
            success_1_samples_path = f"{output_dir}/success_1_samples_{model}_in_turn_{turn_accuracy}.json"
            fail_1_samples_path = f"{output_dir}/fail_1_samples_{model}_in_turn_{turn_accuracy}.json"
            success_10_samples_path = f"{output_dir}/success_10_samples_{model}_in_turn_{turn_accuracy}.json"
            fail_10_samples_path = f"{output_dir}/fail_10_samples_{model}_in_turn_{turn_accuracy}.json"
            success_50_samples_path = f"{output_dir}/success_50_samples_{model}_in_turn_{turn_accuracy}.json"
            fail_50_samples_path = f"{output_dir}/fail_50_samples_{model}_in_turn_{turn_accuracy}.json"
            if args.write:
                if turn_accuracy == 0:
                    with open(all_samples_path, 'w', encoding="utf-8") as f:
                        json.dump(all_samples, f, ensure_ascii=False, indent=2)
                    print(f"Matched samples saved to {all_samples_path}\n num_samples: {len(all_samples)}")
                else:
                    with open(turn_samples_path, 'w', encoding="utf-8") as f:
                        json.dump(turn_samples, f, ensure_ascii=False, indent=2)
                    print(f"Matched samples saved to {turn_samples_path}\n num_samples: {len(turn_samples)}")
                    if topk == 1:
                        with open(success_1_samples_path, 'w', encoding="utf-8") as f:
                            json.dump(success_1_samples, f, ensure_ascii=False, indent=2)
                        print(f"Matched samples saved to {success_1_samples_path}\n num_samples: {len(success_1_samples)}")
                        with open(fail_1_samples_path, 'w', encoding="utf-8") as f:
                            json.dump(fail_1_samples, f, ensure_ascii=False, indent=2)
                        print(f"Matched samples saved to {fail_1_samples_path}\n num_samples: {len(fail_1_samples)}")
                    elif topk == 10:
                        with open(success_10_samples_path, 'w', encoding="utf-8") as f:
                            json.dump(success_10_samples, f, ensure_ascii=False, indent=2)
                        print(f"Matched samples saved to {success_10_samples_path}\n num_samples: {len(success_10_samples)}")
                        with open(fail_10_samples_path, 'w', encoding="utf-8") as f:
                            json.dump(fail_10_samples, f, ensure_ascii=False, indent=2)
                        print(f"Matched samples saved to {fail_10_samples_path}\n num_samples: {len(fail_10_samples)}")
                    elif topk == 50:
                        with open(success_50_samples_path, 'w', encoding="utf-8") as f:
                            json.dump(success_50_samples, f, ensure_ascii=False, indent=2)
                        print(f"Matched samples saved to {success_50_samples_path}\n num_samples: {len(success_50_samples)}")
                        with open(fail_50_samples_path, 'w', encoding="utf-8") as f:
                            json.dump(fail_50_samples, f, ensure_ascii=False, indent=2)
                        print(f"Matched samples saved to {fail_50_samples_path}\n num_samples: {len(fail_50_samples)}")
                        
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--turn_num', type=int)
    parser.add_argument('--mode', type=str)
    parser.add_argument('--turn_accuracy', type=int, default=0)
    parser.add_argument('--topk', type=int, default=50)
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--model', type=str)
    parser.add_argument('--timelog', type=str)
    args = parser.parse_args()
    models.append(args.model)
    rec_eval(args, args.turn_num, args.mode, args.turn_accuracy, args.topk, args.timelog)