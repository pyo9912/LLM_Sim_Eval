import json
import os
import argparse

def load_json(file_path):
    """Load JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, file_path):
    """Save JSON data to file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def sort_json_by_dialog_id(directory):
    """Find all JSON files in the directory and sort by dialog_id."""
    json_files = []

    # 모든 JSON 파일 찾기
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    """Sort JSON data by dialog_id in each file."""
    for file in json_files:
        data = load_json(file)
        if isinstance(data, list):
            # data.sort(key=lambda x: int(x.get("dialog_id", 0)))  # Sort by dialog_id
            data.sort(key=lambda x:(x['dialog_id'], x['turn_id']))
            save_json(data, file)
            print(f"Sorted and saved: {file}")
        else:
            print(f"Skipping {file}: Data is not a list")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--turn_num', type=int)
    parser.add_argument('--mode', type=str)
    parser.add_argument('--turn_accuracy', type=int, default=0)
    parser.add_argument('--topk', type=int, default=50)
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--model', type=str)
    parser.add_argument('--timelog', type=str)
    args = parser.parse_args()
    file_dir = f"/home/user/junpyo/iEvaLM-CRS-main/save_5/output/{args.model}/{args.timelog}"
    sort_json_by_dialog_id(file_dir)