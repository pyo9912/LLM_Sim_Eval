import json
import os
import sys

import numpy as np
import torch
# import wandb
from nltk.translate.bleu_score import sentence_bleu
from transformers import AutoTokenizer

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import GenerationConfig, LlamaForCausalLM
from peft import PeftModel

import re
import time
import datetime
from copy import deepcopy

import nltk
import openai
import tiktoken
import numpy as np
import typing

from src.model.CHATGPT import CHATGPT

from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity
from accelerate.utils import set_seed
from tenacity import _utils, Retrying, retry_if_not_exception_type
from tenacity.stop import stop_base
from tenacity.wait import wait_base
from thefuzz import fuzz


# chat_recommender_instruction_ours = '''Pretend you are a conversational recommender system. I will provide you a dialog between a user and the system.\n\nHere is the dialog.\n{instruction}\n\nCreate a response that the system should provide.\n\n### Response:'''
# chat_recommender_instruction_ours_noft = '''You are a recommender engaging in a conversation with the user to provide recommendations. You must follow the instructions below during the chat:
# If you have sufficient confidence in the user's preferences, you should recommend 10 items the user is most likely to prefer without any explanations. The recommendation list can contain items that were already mentioned in the dialog. The format of the recommendation list is: no. title (year).
# If you do not have sufficient confidence in the user's preferences, you should ask the user about their preferences.

# Here is the dialog.
# {instruction}

# Create the suitable response following my instructions.

# ### Response:'''

response_split = '### Response:'

instruction_ours = """You are a recommender engaging in a conversation with the user to provide recommendations. 
You must follow the instructions below during the chat:
If you have sufficient confidence in the user's preferences, you should recommend 10 items the user is most likely to prefer without any explanations. 
The recommendation list can contain items that were already mentioned in the dialog. The format of the recommendation list is: no. title (year).
If you do not have sufficient confidence in the user's preferences, you should ask the user about their preferences."""

instruction_rec = """You are a recommender engaging in a conversation with the user to provide recommendations. 
You must recommend 10 items the user is most likely to prefer without any explanations. 
The recommendation list can contain items that were already mentioned in the dialog. The format of the recommendation list is: no. title (year)."""

# os.environ["TOKENIZERS_PARALLELISM"] = "false"

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

class my_wait_exponential(wait_base):
    def __init__(
        self,
        multiplier: typing.Union[int, float] = 1,
        max: _utils.time_unit_type = _utils.MAX_WAIT,  # noqa
        exp_base: typing.Union[int, float] = 2,
        min: _utils.time_unit_type = 0,  # noqa
    ) -> None:
        self.multiplier = multiplier
        self.min = _utils.to_seconds(min)
        self.max = _utils.to_seconds(max)
        self.exp_base = exp_base

    def __call__(self, retry_state: "RetryCallState") -> float:
        if retry_state.outcome == openai.error.Timeout:
            return 0

        try:
            exp = self.exp_base ** (retry_state.attempt_number - 1)
            result = self.multiplier * exp
        except OverflowError:
            return self.max
        return max(max(0, self.min), min(result, self.max))


class my_stop_after_attempt(stop_base):
    """Stop when the previous attempt >= max_attempt."""

    def __init__(self, max_attempt_number: int) -> None:
        self.max_attempt_number = max_attempt_number

    def __call__(self, retry_state: "RetryCallState") -> bool:
        if retry_state.outcome == openai.error.Timeout:
            retry_state.attempt_number -= 1
        return retry_state.attempt_number >= self.max_attempt_number


def annotate(conv_str):
    request_timeout = 6
    for attempt in Retrying(
        reraise=True, retry=retry_if_not_exception_type((openai.error.InvalidRequestError, openai.error.AuthenticationError)),
        wait=my_wait_exponential(min=1, max=60), stop=(my_stop_after_attempt(8)), before_sleep=my_before_sleep
    ):
        with attempt:
            response = openai.Embedding.create(
                model='text-embedding-ada-002', input=conv_str, request_timeout=request_timeout
            )
        request_timeout = min(30, request_timeout * 2)

    return response


class LLAMA2:
    def __init__(self, *args, **kwargs):
        self.conv_model = kwargs['conv_model']
        self.saved_model_path = f"/home/user/junpyo/iEvaLM-CRS-main/src/{self.conv_model}"
        self.args = kwargs['args']
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.base_model)
        

        self.model = self.prepare_model()
                # set instruction prompt
        # if self.args.crs_prompt == 'ours_noft':
        #     self.instruction = chat_recommender_instruction_ours_noft
        # else:
        #     self.instruction = chat_recommender_instruction_ours


        # self.kg_dataset = kwargs['kg_dataset']
        # self.kg_dataset_path = f"../data/{self.kg_dataset}"
        # with open(f"{self.kg_dataset_path}/entity2id.json", 'r', encoding="utf-8") as f:
        #     self.entity2id = json.load(f)
        # with open(f"{self.kg_dataset_path}/id2info.json", 'r', encoding="utf-8") as f:
        #     self.id2info = json.load(f)
            
        # self.id2entityid = {}
        # for id, info in self.id2info.items():
        #     if info['name'] in self.entity2id:
        #         self.id2entityid[id] = self.entity2id[info['name']]
        
        # self.item_embedding_path = f"../save/embed/item/{self.kg_dataset}/llama2"
        
        # item_emb_list = []
        # id2item_id = []
        # for i, file in tqdm(enumerate(os.listdir(self.item_embedding_path))):
        #     item_id = os.path.splitext(file)[0]
        #     if item_id in self.id2entityid:
        #         id2item_id.append(item_id)

        #         with open(f'{self.item_embedding_path}/{file}', encoding='utf-8') as f:
        #             embed = json.load(f)
        #             item_emb_list.append(embed)

        # self.id2item_id_arr = np.asarray(id2item_id)
        # self.item_emb_arr = np.asarray(item_emb_list)
        self.item_mapping = CHATGPT(*args, **kwargs)

    def prepare_model(self,
                      base_model: str = "",
                      load_8bit: bool = False,
                      lora_weights: str = "",
                      server_name: str = "0.0.0.0",  # Allows to listen on all interfaces by providing '0.
                      share_gradio: bool = False, ):
        print('prepare new model for evaluating')
        base_model = self.args.base_model

        if self.args.peft_weights != '':
            peft_weights = os.path.join(self.saved_model_path, self.args.peft_weights)
        else:
            peft_weights = ''
            
        assert (
            base_model
        ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

        if self.args.bf:
            dtype = torch.bfloat16
        else:
            dtype = torch.float16

        if device == "cuda":
            model = LlamaForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=load_8bit,
                torch_dtype=dtype,  #
            ).to("cuda")

            if peft_weights != "":
                model = PeftModel.from_pretrained(
                    model,
                    peft_weights,
                    torch_dtype=dtype,
                )
        else:
            raise ValueError

        self.tokenizer.add_special_tokens({'pad_token': '<pad>'})
        model.resize_token_embeddings(len(self.tokenizer))
        model.config.pad_token_id = self.tokenizer.pad_token_id
        model.config.bos_token_id = self.tokenizer.bos_token_id
        model.config.eos_token_id = self.tokenizer.eos_token_id
        self.tokenizer.add_eos_token = False
        return model


    def evaluate(self,
                 input_ids,
                 attention_mask,
                 model,
                 input=None,
                 temperature=0.1,
                 top_p=0.75,
                 top_k=40,
                 num_beams=1,  # todo: beam 1개로 바꿔보기
                 max_new_tokens=512,
                 **kwargs):
        generation_config = GenerationConfig(
            num_beams=num_beams,
            num_return_sequences=num_beams,
            output_logits=True,
            output_scores=True,
            return_dict_in_generate=True,
            **kwargs,
        )

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                return_dict_in_generate=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences
        output = self.tokenizer.batch_decode(s, skip_special_tokens=True)[0]
        # generated_responses = output.split(response_split)[-1].split('System: ')[-1].strip()
        generated_responses = output[output.rfind('[/INST]'):].split('[/INST]')[-1].replace('\n','').strip()

        return generated_responses
  
    def get_conv(self, conv_dict):
        

        context = conv_dict['context']        
        context_list = []
        # context_list = [{'role': 'system', 'content': instruction}]

        # for i, text in enumerate(context):
        #     if len(text) == 0:
        #         continue
        #     if i % 2 == 0:
        #         role_str = 'user'
        #     else:
        #         role_str = 'assistant'
        #     # context_list.append(f"{role_str}: {text}")
        #     context_list.append({'role': role_str, 'content': text})

        for i, text in enumerate(context):
            # if len(text) == 0:
            #     continue
            if i % 2 == 0:
                role_str = 'user'
            else:
                role_str = 'assistant'
            # context_list.append(f"{role_str}: {text}")
            context_list.append({"role": role_str, "content": text})
        
        context_list = context_list[-5:]
        if self.args.crs_prompt == 'only_rec':
            instruction = instruction_rec
        else:
            instruction = instruction_ours
        context_list.insert(0, {'role': 'system', 'content': instruction})

        # print(context_list)        
        # full_prompt = self.instruction.format(instruction='\n'.join(context_list[-5:]))
        full_prompt = self.tokenizer.apply_chat_template(context_list, add_generation_prompt=True, tokenize=False)

        self.model.eval()

        inputs = self.tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True).to("cuda")
        input_ids = inputs["input_ids"].to("cuda")
        attention_mask = inputs["attention_mask"].to("cuda")
        responses = self.evaluate(input_ids, attention_mask, self.model, max_new_tokens=self.args.max_new_tokens, num_beams=self.args.num_beams)

        return None, responses


    def get_emb(self, batch_texts):

        batch_texts_tokens = self.tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True)
        input_ids = batch_texts_tokens["input_ids"].to("cuda")
        attention_mask = batch_texts_tokens["attention_mask"].to("cuda")
        self.model.eval()

        with torch.no_grad():
            last_hidden_state = self.model(input_ids, attention_mask, output_hidden_states=True).hidden_states[-1]
            last_token_embedding = last_hidden_state[:, -1, :]

        return last_token_embedding.cpu().tolist()


    ### CHATGPT.py 와 동일하게 동작 (매핑)
    def get_rec(self, conv_dict, response=None):
        item_rank_arr, rec_labels = self.item_mapping.get_rec(conv_dict, response)
        # rec_labels = [self.entity2id[rec] for rec in conv_dict['rec'] if rec in self.entity2id]
        
        # context = conv_dict['context']
        # if response is not None:
        #     context = conv_dict['context'] + [response]
        #     # context = [response]

        # context_list = [] # for model
        
        # for i, text in enumerate(context):
        #     if len(text) == 0:
        #         continue
        #     if i % 2 == 0:
        #         role_str = 'user'
        #     else:
        #         role_str = 'assistant'
        #     context_list.append({
        #         'role': role_str,
        #         'content': text
        #     })
        
        # conv_str = ""
        
        # for context in context_list[-4:]:
        #     # conv_str += f"{context['role']}: {context['content']} "
        #     conv_str += f"{context['content']}"
            
        # # conv_embed = annotate(conv_str)['data'][0]['embedding']
        # conv_embed = self.get_emb(conv_str)
        # conv_embed = np.asarray(conv_embed).reshape(1, -1)
        
        # sim_mat = cosine_similarity(conv_embed, self.item_emb_arr)
        # rank_arr = np.argsort(sim_mat, axis=-1).tolist()
        # rank_arr = np.flip(rank_arr, axis=-1)[:, :50]
        # item_rank_arr = self.id2item_id_arr[rank_arr].tolist()
        # item_rank_arr = [[self.id2entityid[item_id] for item_id in item_rank_arr[0]]]

        return item_rank_arr, rec_labels
    
    ### attr form 용 --> free form에는 필요없음
    def get_choice(self, gen_inputs, options, state, conv_dict):
        pass
