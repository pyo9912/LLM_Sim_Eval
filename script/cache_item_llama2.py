import json
import os
import random
import typing
from argparse import ArgumentParser
from chat import get_model_args

import openai
from loguru import logger
from tenacity import _utils, Retrying, retry_if_not_exception_type
from tenacity.stop import stop_base
from tenacity.wait import wait_base

from parser import parse_args
from src.model.LLAMA2 import LLAMA2


def get_exist_item_set():
    exist_item_set = set()
    for file in os.listdir(save_dir):
        user_id = os.path.splitext(file)[0]
        exist_item_set.add(user_id)
    return exist_item_set


if __name__ == '__main__':

    args = parse_args()
    args.dataset = 'redial'
    
    model_args = get_model_args(args)

    llama2 = LLAMA2(**model_args)
    
    from platform import system as sysChecker
    if sysChecker() == 'Linux':
        args.home = os.path.dirname(os.path.dirname(__file__))
    elif sysChecker() == "Windows":
        args.home = ''
    
    openai.api_key = args.api_key
    print(f"API_KEY = {args.api_key}")

    batch_size = args.batch_size
    dataset = args.dataset

    save_dir = os.path.join(args.home, f'save/embed/item/{dataset}/llama2')
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(args.home, f'data/{dataset}/id2info.json'), encoding='utf-8') as f:
        id2info = json.load(f)

    # redial
    if dataset == 'redial':
        info_list = list(id2info.values())
        item_texts = []
        for info in info_list:
            item_text_list = [
                f"Title: {info['name']}", f"Genre: {', '.join(info['genre']).lower()}",
                f"Star: {', '.join(info['star'])}",
                f"Director: {', '.join(info['director'])}", f"Plot: {info['plot']}"
            ]
            item_text = '; '.join(item_text_list)
            item_texts.append(item_text)
        attr_list = ['genre', 'star', 'director']

    # opendialkg
    if dataset == 'opendialkg':
        item_texts = []
        for info_dict in id2info.values():
            item_attr_list = [f'Name: {info_dict["name"]}']
            for attr, value_list in info_dict.items():
                if attr != 'title':
                    item_attr_list.append(f'{attr.capitalize()}: ' + ', '.join(value_list))
            item_text = '; '.join(item_attr_list)
            item_texts.append(item_text)
        attr_list = ['genre', 'actor', 'director', 'writer']

    id2text = {}
    for item_id, info_dict in id2info.items():
        attr_str_list = [f'Title: {info_dict["name"]}']
        for attr in attr_list:
            if attr not in info_dict:
                continue
            if isinstance(info_dict[attr], list):
                value_str = ', '.join(info_dict[attr])
            else:
                value_str = info_dict[attr]
            attr_str_list.append(f'{attr.capitalize()}: {value_str}')
        item_text = '; '.join(attr_str_list)
        id2text[item_id] = item_text

    item_ids = set(id2info.keys()) - get_exist_item_set()
    while len(item_ids) > 0:
        logger.info(len(item_ids))

        # redial
        if dataset == 'redial':
            batch_item_ids = random.sample(tuple(item_ids), min(batch_size, len(item_ids)))
            batch_texts = [id2text[item_id] for item_id in batch_item_ids]

        # opendialkg
        if dataset == 'opendialkg':
            batch_item_ids = random.sample(tuple(item_ids), min(batch_size, len(item_ids)))
            batch_texts = [id2text[item_id] for item_id in batch_item_ids]

        # batch_embeds = annotate(llama2, batch_texts)
        batch_embeds = llama2.get_emb(batch_texts)
        for idx, embed in enumerate(batch_embeds):
            item_id = batch_item_ids[idx]
            with open(f'{save_dir}/{item_id}.json', 'w', encoding='utf-8') as f:
                json.dump(embed, f, ensure_ascii=False)

        item_ids -= get_exist_item_set()
