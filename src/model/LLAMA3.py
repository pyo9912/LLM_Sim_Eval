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


instruction_ours = """You are a recommender engaging in a conversation with the user to provide recommendations. 
You must follow the instructions below during the chat:
If you have sufficient confidence in the user's preferences, you should recommend 10 items the user is most likely to prefer without any explanations. 
The recommendation list can contain items that were already mentioned in the dialog. The format of the recommendation list is: no. title (year).
If you do not have sufficient confidence in the user's preferences, you should ask the user about their preferences."""

instruction_rec = """You are a recommender engaging in a conversation with the user to provide recommendations. 
You must recommend 10 items the user is most likely to prefer without any explanations. 
The recommendation list can contain items that were already mentioned in the dialog. The format of the recommendation list is: no. title (year)."""

def format_llama3_prompt(dialog: list[dict]) -> str:
    """
    dialog: [{'role': 'user' | 'assistant', 'content': str}, ...]
    instruction: task framing instruction string
    Returns: LLaMA3-style prompt string including the instruction
    """
    # prompt = "<|begin_of_text|>\n"

    # Instruction을 첫 번째 user 메시지로 삽입
    prompt = "<|start_header_id|>system<|end_header_id|>\n" + instruction.strip() + "<|eot_id|>\n\n"

    # 실제 멀티턴 대화
    for message in dialog:
        role = message["role"]
        content = message["content"].strip()
        tag = "<|start_header_id|>user<|end_header_id|>" if role == "user" else "<|start_header_id|>assistant<|end_header_id|>"
        prompt += f"{tag}\n{content}<|eot_id|>\n"

    # 모델이 assistant로 응답할 수 있도록 마무리
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n"
    return prompt




# dialog = [
#     {"role": "user", "content": "안녕! 영화 추천해줘."},
#     {"role": "assistant", "content": "안녕하세요! 어떤 장르 좋아하시나요?"},
#     {"role": "user", "content": "액션 영화 좋아해."}
# ]

# prompt = format_llama3_prompt(dialog, instruction)
# print(prompt)

# os.environ["TOKENIZERS_PARALLELISM"] = "false"

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


class LLAMA3:
    def __init__(self, *args, **kwargs):
        self.conv_model = kwargs['conv_model']
        self.saved_model_path = f"/home/user/junpyo/iEvaLM-CRS-main/src/{self.conv_model}"
        self.args = kwargs['args']
        # self.args.base_model = "meta-llama/Meta-Llama-3-8B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.base_model)
        
        self.model = self.prepare_model()

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
            # model = LlamaForCausalLM.from_pretrained(
            #     base_model,
            #     load_in_8bit=load_8bit,
            #     torch_dtype=dtype,  #
            # ).to("cuda")

            model = LlamaForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=True,
                device_map="auto"
            )

            if peft_weights != "":
                model = PeftModel.from_pretrained(
                    model,
                    peft_weights,
                    torch_dtype=dtype,
                )
        else:
            raise ValueError

        # self.tokenizer.add_special_tokens({'pad_token': '<pad>'})
        # model.resize_token_embeddings(len(self.tokenizer))
        # model.config.pad_token_id = self.tokenizer.pad_token_id
        # model.config.bos_token_id = self.tokenizer.bos_token_id
        # model.config.eos_token_id = self.tokenizer.eos_token_id
        # self.tokenizer.add_eos_token = False
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
                pad_token_id=self.tokenizer.eos_token_id
            )
        s = generation_output.sequences
        output = self.tokenizer.batch_decode(s, skip_special_tokens=True)[0]
        
        # input_lens = torch.sum(attention_mask, dim=-1).cpu().tolist()
        # generated_responses = [output[idx][l:].strip() for idx, l in enumerate(input_lens)]
        generated_responses = output[output.rfind('assistant\n'):].split('assistant\n')[-1].replace('\n','').strip()
        return generated_responses
  
    def get_conv(self, conv_dict):
        

        context = conv_dict['context']        
        context_list = []

        # context_list = [{'role': 'system', 'content': instruction}]
        for i, text in enumerate(context):
            if len(text) == 0:
                continue
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
        # full_prompt = self.instruction.format(instruction=dialog)
        # full_prompt = format_llama3_prompt(dialog)
        full_prompt = self.tokenizer.apply_chat_template(context_list, add_generation_prompt=True, tokenize=False)
        self.model.eval()

        inputs = self.tokenizer(full_prompt, return_tensors="pt").to("cuda")
         
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
        return item_rank_arr, rec_labels
    
    ### attr form 용 --> free form에는 필요없음
    def get_choice(self, gen_inputs, options, state, conv_dict):
        pass
