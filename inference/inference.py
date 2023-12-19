from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import os
import json
import pandas as pd
import argparse
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import sys

class ListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return generate_prompt(self.original_list[i])


def generate_prompt(example):
    input = example['input']
    instruction = example['instruction']
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""

def evaluate(
        sample_prompt,
        args,
        tokenizer=None,
        model=None
    ):
        temperature = args.temperature
        top_p = args.top_p
        top_k = args.top_k
        num_beams = args.num_beams
        repetition_penalty = args.repetition_penalty
        no_repeat_ngram_size = args.no_repeat_ngram_size
        # prompt = generate_prompt(json_sample)        
        # inputs = tokenizer(sample_prompt, return_tensors="pt")
        inputs = tokenizer(sample_prompt, truncation=True, max_length=args.max_new_tokens, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            # forced_bos_token_id=tokenizer.bos_token_id
            # forced_eos_token_id=tokenizer.eos_token_id
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=args.max_new_tokens,
            )
        s=generation_output.sequences
        output = tokenizer.batch_decode(s, skip_special_tokens=True)
        return output[0]

def load_dataset(args):
    with open(args.infer_ids_path, 'r') as f: 
        test_ids = f.readlines()
    test_ids=[int(number.strip()) for number in test_ids]
    print(f"# of test ids: {len(test_ids)}")
    # load json file
    with open(args.dataset_path, 'r') as f:
        dataset = json.load(f)
    test_json_samples = [sample for sample in dataset if sample['QuestionId'] in test_ids]

    print(f"# of data: {len(test_json_samples)}")
    return test_json_samples

def postprocess(output):
    if "### Response:" not in output:
        print('failed to generate response due to trunction')
        return None
    output = output.split("### Response:")[1].strip()
    output = output.split('</s>')[0].strip()
    return output

def single_inference(args, model, tokenizer):
    test_json_samples = load_dataset(args)
    for idx, json_sample in enumerate(tqdm(test_json_samples)):
        file_save_path = os.path.join(args.save_path, f'{json_sample["QuestionId"]}.json')
        if os.path.exists(file_save_path):
            continue
        output = evaluate(sample_prompt=generate_prompt(json_sample), args=args, tokenizer=tokenizer, model=model)
        result=postprocess(output)
        if result is None:
            continue
        
        with open(file_save_path, 'w') as f:
            f.write(json.dumps(dict(QuestionId=json_sample["QuestionId"], instruction=json_sample['instruction'], question=json_sample['input'], answer=json_sample['output'], llama=result), indent=4))            
        print(result)


def batch_inference(args, inference_pipeline, tokenizer):
    test_json_samples = load_dataset(args)
    test_dataset=ListDataset(test_json_samples)

    pipelineIterator = inference_pipeline(
        test_dataset,
        do_sample=False,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
        
    for idx, (json_sample, result) in enumerate(tqdm(zip(test_dataset, pipelineIterator))):    
        result = result[0]['generated_text']
        result=postprocess(result)
        file_save_path = os.path.join(args.save_path, f'{json_sample["QuestionId"]}.json')
        with open(file_save_path, 'w') as f:
            f.write(json.dumps(dict(QuestionId=json_sample["QuestionId"], instruction=json_sample['instruction'], question=json_sample['input'], answer=json_sample['output'], llama=result), indent=4))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str)
    parser.add_argument('--infer_ids_path', type=str)
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--load_8bit", type=bool, default=False)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.75)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--num_beams", type=int, default=3)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()    
    os.makedirs(args.save_path, exist_ok=True)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    try:
        if torch.backends.mps.is_available():
            device = "mps"
    except:
        pass


    assert args.base_model, (
        "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    )
    # args.model_path = 'gpt2'
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False, padding_side="left", truncation=True, max_length=args.max_new_tokens)
    # tokenizer.use_fast = False
    # tokenizer.padding_side = "left"
    # tokenizer.truncation=True
    # tokenizer.max_length=args.max_new_tokens
    device_map = None
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            load_in_8bit=args.load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    model.tie_weights()

    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not args.load_8bit:
        model.half()

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    # tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    # tokenizer.padding_side = "left" 
    # tokenizer.pad_token = tokenizer.eos_token # to avoid an error

    # tokenizer.pad_token_id = 0  # unk
    # tokenizer.bos_token_id = 1
    # tokenizer.eos_token_id = 2

    if args.batch_size==1:
        print("single inference")
        # not batching
        single_inference(args, model, tokenizer)        
    else:
        print("batch inference")
        # batching
        inference_pipeline = pipeline(
            "text-generation", 
            tokenizer=tokenizer, 
            model=model, 
            device_map=device_map, 
            batch_size=args.batch_size, 
            torch_dtype=torch.float16)
                    
        batch_inference(args, inference_pipeline, tokenizer)
    

