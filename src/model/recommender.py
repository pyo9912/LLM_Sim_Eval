import sys
sys.path.append("..")

from src.model.KBRD import KBRD
from src.model.BARCOR import BARCOR
from src.model.UNICRS import UNICRS
from src.model.CHATGPT import CHATGPT
from src.model.LLAMA2 import LLAMA2
from src.model.LLAMA3 import LLAMA3

name2class = {
    'kbrd': KBRD,
    'barcor': BARCOR,
    'unicrs': UNICRS,
    'chatgpt': CHATGPT,
    'llama2': LLAMA2,
    'llama3': LLAMA3
}

class RECOMMENDER():
    def __init__(self, crs_model, *args, **kwargs) -> None:
        if 'chatgpt' in crs_model:
            crs_model_name = 'chatgpt'
        elif 'llama2' in crs_model:
            crs_model_name = 'llama2'
        elif 'llama3' in crs_model:
            crs_model_name = 'llama3'
        elif 'unicrs' in crs_model:
            crs_model_name = 'unicrs'
        elif 'kbrd' in crs_model:
            crs_model_name = 'kbrd'
        else:
            crs_model_name = crs_model
        model_class = name2class[crs_model_name]
        self.crs_model = model_class(*args, **kwargs)
        self.crs_model_name = crs_model_name
        
    def get_rec(self, conv_dict, response=None):
        return self.crs_model.get_rec(conv_dict, response)
    
    def get_conv(self, conv_dict):
        return self.crs_model.get_conv(conv_dict)
    
    def get_choice(self, gen_inputs, option, state, conv_dict=None):
        return self.crs_model.get_choice(gen_inputs, option, state, conv_dict)