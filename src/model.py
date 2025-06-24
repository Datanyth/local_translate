import os
from transformers import AutoTokenizer, LlamaForCausalLM
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, LlamaForCausalLM, BitsAndBytesConfig
import time 
from torch.nn.attention import SDPBackend, sdpa_kernel
import torch
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TranslateModel:

    def __init__(self, prompt_template_fn, model_id:str, max_length_token:int = 12800, use_4bit: bool = False, use_8bit: bool = False):
        

        self.model_id = model_id
        self.max_length_token = max_length_token
        self.prompt_template_fn = prompt_template_fn
        
        try:
            if use_4bit:
                bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.float16)
            elif use_8bit:
                bnb_config = BitsAndBytesConfig(load_in_8bit=True, bnb_8bit_compute_dtype=torch.float16)
            else:
                bnb_config = None
            self.model = LlamaForCausalLM.from_pretrained(
                self.model_id, quantization_config=bnb_config, device_map="auto"
                ).to(device= 'cuda', dtype = torch.float16)
        except Exception as e:
            logger.info(f'Got error {e} when loading model translate ')
            quit()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        

    def translate(self, query:str, src_language:str, trg_language:str):
        prompt = self.prompt_template_fn(query, src_language, trg_language)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device= 'cuda')
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            generate_ids = self.model.generate(inputs.input_ids, max_length=self.max_length_token)
            output= self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return output




