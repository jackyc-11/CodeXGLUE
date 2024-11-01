from openai import OpenAI
from typing import Iterable, Dict
import gzip
import json
import os
from codebleu import calc_codebleu
from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration
import anthropic

def generate_one_completion_gpt(prompt):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    system_prompt = "You are a helpful coding assistant. Write clean, functional code. Just the code without any explanation or ticks"
    
    client = OpenAI()

    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": prompt,
            }
        ]
    )
    return completion.choices[0].message.content.strip()

def generate_one_completion_llama(prompt):
    # Load the LLaMA model and tokenizer from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")  # Example model
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_new_tokens=300, do_sample=True, temperature=0.7)
    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_code

def generate_one_completion_claude(prompt):
    claude_api_key = os.getenv("CLAUDE_API_KEY")
    client = anthropic.Client(api_key=claude_api_key)
    
    response = client.completions.create(
        model="claude-v1",  # Use the appropriate model (e.g., claude-v1, claude-v2)
        prompt=f"{anthropic.HUMAN_PROMPT} {prompt}{anthropic.AI_PROMPT}",
        max_tokens_to_sample=300,
        temperature=0.7,
    )
    
    generated_code = response['completion'].strip()
    return generated_code

def generate_one_completion(prompt, model="gpt"):
    if model == "gpt":
        return generate_one_completion_gpt(prompt)
    elif model == "llama":
        return generate_one_completion_llama(prompt)
    elif model == "claude":
        return generate_one_completion_claude(prompt)
    else:
        raise ValueError("Unsupported model")

ROOT = os.path.dirname(os.path.abspath(__file__))
HUMAN_EVAL = os.path.join(ROOT, "..", "data", "HumanEval.jsonl.gz")
def read_problems(evalset_file: str = HUMAN_EVAL) -> Dict[str, Dict]:
    return {task["task_id"]: task for task in stream_jsonl(evalset_file)}

def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, 'rt') as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)

def write_jsonl(filename: str, data: Iterable[Dict], append: bool = False):
    """
    Writes an iterable of dictionaries to jsonl
    """
    if append:
        mode = 'ab'
    else:
        mode = 'wb'
    filename = os.path.expanduser(filename)
    if filename.endswith(".gz"):
        with open(filename, mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode='wb') as gzfp:
                for x in data:
                    gzfp.write((json.dumps(x) + "\n").encode('utf-8'))
    else:
        with open(filename, mode) as fp:
            for x in data:
                fp.write((json.dumps(x) + "\n").encode('utf-8'))

if __name__ == "__main__":
    data = []
    with open("Text-Code/text-to-code/dataset/concode/dev.json", "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))

    references = []
    predictions = []

    for i in range(100):
        prompt = data[i]["nl"]
        reference_code = data[i]["code"]
        
        generated_code = generate_one_completion(prompt)

        references.append(reference_code)
        predictions.append(generated_code)

    result = calc_codebleu(references, predictions, "python")

    print("CodeBLEU result:")
    print(result)

    ################################################################
    ################################################################

    # num_problems = 164
    # num_samples_per_problem = 3
    # model = "gpt"
    # exp_output = f"results/{model}_{num_problems}p_{num_samples_per_problem}s.jsonl"

    # # read problems
    # problems = read_problems()
    # print("Problems read...")

    # references = []
    # predictions = []

    # problems_pairs = list(problems.items())[:num_problems] # (task_id, problem)
    # for i, (task_id, problem) in enumerate(problems_pairs):
    #     prompt = problem["prompt"]
    #     reference_code = problem["canonical_solution"]
        
    #     # Generate multiple samples for each problem
    #     for _ in range(num_samples_per_problem):
    #         generated_code = generate_one_completion(prompt, model)
    #         references.append(reference_code)
    #         predictions.append(generated_code)

    # # Calculate CodeBLEU for the generated predictions
    # result = calc_codebleu(references, predictions, "python")

    # print("CodeBLEU result:")
    # print(result)


    # samples = [
    #     dict(task_id=problem[1]['task_id'], completion=generate_one_completion(problem[1]["prompt"], model))
    #     for problem in problems_pairs
    #     for _ in range(num_samples_per_problem)
    # ]

    # print("Samples generated...")
    # write_jsonl(exp_output, samples)