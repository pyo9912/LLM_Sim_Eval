import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', type=str)
    parser.add_argument('--dataset', type=str, default='redial_eval', choices=['redial_eval', 'opendialkg_eval'])
    parser.add_argument('--turn_num', type=int, default=5)
    parser.add_argument('--crs_model', type=str, default='chatgpt') #, choices=['kbrd', 'barcor', 'unicrs', 'chatgpt'])
    parser.add_argument('--chatgpt_model', type=str, default='gpt-4o') # gpt-4o
    parser.add_argument('--timelog', type=str)

    # interaction cross check
    parser.add_argument('--interaction_cross', action='store_true')
    parser.add_argument('--interaction_model', type=str, default='chatgpt')
    parser.add_argument('--interaction_log', type=str)

    # prompt select
    parser.add_argument('--user_prompt', type=str, default='ours')
    parser.add_argument('--crs_prompt', type=str, default = 'original')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--kg_dataset', type=str, default='redial', choices=['redial', 'opendialkg'])

    # model_detailed
    parser.add_argument('--hidden_size', type=int)
    parser.add_argument('--entity_hidden_size', type=int)
    parser.add_argument('--num_bases', type=int, default=8)
    parser.add_argument('--context_max_length', type=int)
    parser.add_argument('--entity_max_length', type=int)

    # model
    parser.add_argument('--rec_model', type=str)
    parser.add_argument('--conv_model', type=str)

    # conv
    parser.add_argument('--tokenizer_path', type=str)
    parser.add_argument('--encoder_layers', type=int)
    parser.add_argument('--decoder_layers', type=int)
    parser.add_argument('--text_hidden_size', type=int)
    parser.add_argument('--attn_head', type=int)
    parser.add_argument('--resp_max_length', type=int)

    # prompt
    parser.add_argument('--model', type=str)
    parser.add_argument('--text_tokenizer_path', type=str)
    parser.add_argument('--text_encoder', type=str)
    parser.add_argument('--topk',type=int, default=50)

    # peft
    parser.add_argument('--sft', type=bool, default=False)
    parser.add_argument('--bf', type=bool, default=False)
    parser.add_argument('--fp16_trainarg', type=bool, default=False)
    parser.add_argument('--quantization', type=str, default="8bit")
    parser.add_argument('--peft', type=str, default="lora")
    parser.add_argument('--local_rank', type=int, default=-1)

    # llama
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_new_tokens', type=int, default=500)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--peft_weights', type=str, default="")
    parser.add_argument('--base_model', type=str, default='meta-llama/Llama-2-7b-chat-hf')


    args = parser.parse_args()

    from platform import system as sysChecker
    if sysChecker() == 'Linux':
        args.home = os.path.dirname(os.path.dirname(__file__))
    elif sysChecker() == "Windows":
        args.home = ''
    print(args.home)

    return args