import os, json
import time
import random
from openai import OpenAI
from tqdm import tqdm
from typing import List, Union, Dict
from copy import deepcopy

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), organization=os.getenv("OPENAI_ORG"))

# Define a function that adds a delay to a Completion API call
def delayed_completion(delay_in_seconds: float = 1, max_trials: int = 1, **kwargs):
    """Delay a completion by a specified amount of time."""

    # Sleep for the delay
    time.sleep(delay_in_seconds)

    # Call the Completion API and return the result
    output, error = None, None
    for _ in range(max_trials):
        try:
            output = client.chat.completions.create(**kwargs)
            break
        except Exception as e:
            error = e
            pass
    return output, error

class OpenAIGenerationBase:
    def __init__(
        self,
        # customized inputs
        data_list: List[Union[str, Dict]],
        prompt_paths: List[str],
        # save path
        results_json_path: str,
        overwrite: bool = False,
        # generation parameters
        system_msg="You are a helpful assistant.",
        model_name="gpt-3.5-turbo-0613",
        save_every: int = 50,
        delay: float = 1,
        max_trials: int = 10,
        temperature: float = 0,
        max_token: int = 1024,
        seed: int = 0,
        budget: int = None,
        # others
        shuffle_data_before_generate: bool = False,
        debug_mode: int = None,
        **kwargs
    ) -> None:
        """
        The key features are:
            1. It allows saving outputs every several inference, results can be picked up from last termination.
            2. It takes customized prepare_data and post_process functions.
            3. There is a consecutive prompting that takes a list of prompts.

        Arguments:
            data_list: a list of data, elements can be dict or string
            prompt_paths: is a list of paths to prompt txt files.
            results_json_path: the results will be saved into this json path,
            overwrite: whether to overwrite previous generations.
            system_msg: system message.
            model_name: openai model name. Please provide the specific date in order to be more reproducible
            save_every: results will be saved every several generations in order to avoid intermediate termination
            delay: this is to reduce the throughput because of the token limit per min from openai
            max_trials: max trials before termination
            temperature: LLM generation temperature. Lower indicates lower variations. Recommend 0 for reproducibility.
            max_token: maximum tokens for LLM.
            seed: there will be randomness during shuffling or debugging mode
            shuffle_data_before_generate: shuffle the data before generation.
            debug_mode: this will sample a few data for example. Provide a number for sampling.
        
        Return:
            None
        """
        self.data_list = data_list
        print(f"Receive {len(data_list)} data")
        if isinstance(self.data_list[0], str):
            self.data_list = [{"text": d} for d in self.data_list]
        else:
            assert isinstance(self.data_list[0], dict)
        self.prompts = []
        for prompt_path in prompt_paths:
            with open(prompt_path, 'r') as f:
                self.prompts.append({
                    "name": prompt_path.split("/")[-1].split(".")[0],
                    "content": ''.join(f.readlines())
                })
        self.promptname2idx = {prompt['name']: idx for idx, prompt in enumerate(self.prompts)}
        print(f"Receive {len(self.prompts)} prompts")
        self.seed = seed
        self.results_json_path = results_json_path
        assert self.results_json_path.endswith(".json"), "`results_json_path` extension should be json"
        print("Results will be saved in: ", self.results_json_path)
        self.overwrite = overwrite
        if self.overwrite:
            print("Results will be overwritten!!!")
        else:
            if os.path.exists(self.results_json_path):
                print("Continue to generate")
        self.system_msg = system_msg
        self.model_name = model_name
        print("Using ", self.model_name)
        self.save_every = save_every
        self.delay = delay
        self.max_trials = max_trials
        self.temperature = temperature
        self.max_token = max_token
        self.seed = seed
        self.budget = budget
        if self.budget is not None:
            print(f"Using budget {self.budget}")
        self.shuffle_data_before_generate = shuffle_data_before_generate
        self.debug_mode = debug_mode
        if self.debug_mode is not None:
            print(f"Enabling debugging mode, will sample {self.debug_mode} data")
    
    def prepare_data(self, datum: dict, prompt_name: str = None):
        """
        This function is for data preparation.
        """
        if prompt_name is None:
            prompt_name = self.prompts[0]['name']
        prepared = deepcopy(self.prompts[self.promptname2idx[prompt_name]]['content'])
        for k in datum:
            karg = f"[{k.upper()}]"
            if isinstance(datum[k], str) and karg in prepared:
                prepared = prepared.replace(karg, datum[k])
        return prepared
    
    def post_process(self, completion: str, prompt_name: str = None):
        """
        This function is a placeholder post-processing for output completions.
        It usually include parsing such as regex or converting formats.
        The base function only takes the full content without parsing.
        """
        content = completion.choices[0].message.content.strip()
        result = content
        return content, result

    def generate(self) -> None:
        """
        This function takes in the raw data input and then generate LLM responses.
        It will only execute once.
        """
        
        random.seed(self.seed)

        folder_name = "/".join(self.results_json_path.split("/")[:-1])
        os.makedirs(folder_name, exist_ok=True)
        if not os.path.exists(self.results_json_path) or self.overwrite:
            prepared_data = deepcopy(self.data_list)
            if self.shuffle_data_before_generate:
                random.shuffle(prepared_data)
        else:
            with open(self.results_json_path, 'r') as f:
                prepared_data = json.load(f)

        for d in prepared_data:
            if "prepared" not in d:
                d["prepared"] = self.prepare_data(d)

        if self.debug_mode is not None:
            prepared_data = random.sample(prepared_data, self.debug_mode)
        
        pbar = tqdm(enumerate(prepared_data), total=len(prepared_data) if self.budget is None else self.budget)
        for idx, datum in pbar:
            if idx == 0:
                print(datum['prepared'])
            if 'prediction' in datum and datum['prediction'] is not None:
                continue
            if self.budget is not None and idx > self.budget:
                continue
            
            messages = [
                {"role": "system", "content": self.system_msg},
                {"role": "user", "content": datum['prepared']}
            ]

            completion, error = delayed_completion(delay_in_seconds=self.delay, max_trials=self.max_trials, model=self.model_name, messages=messages, max_tokens=self.max_token, temperature=self.temperature)

            if completion is None:
                pbar.set_description(f"Saving data after {idx + 1} inference.")
                with open(self.results_json_path, 'w') as f:
                    json.dump(prepared_data, f, indent=4)
                print(error)
                # debug the error
                breakpoint()
            else:
                content, result = self.post_process(completion)
                prepared_data[idx]['content'] = content
                prepared_data[idx]['prediction'] =  result
            
            if idx % self.save_every == 0 and idx > 0:
                pbar.set_description(f"Saving data after {idx + 1} inference.")
                with open(self.results_json_path, "w") as f:
                    json.dump(prepared_data, f, indent=4)
        
        with open(self.results_json_path, "w") as f:
            json.dump(prepared_data, f, indent=4)
        
        print("Done")
    
    def generate_consecutive(self) -> None:
        """
        This function will consecutively generate texts.
        """
        
        random.seed(self.seed)

        folder_name = "/".join(self.results_json_path.split("/")[:-1])
        os.makedirs(folder_name, exist_ok=True)
        if not os.path.exists(self.results_json_path) or self.overwrite:
            prepared_data = deepcopy(self.data_list)
            if self.shuffle_data_before_generate:
                random.shuffle(prepared_data)
        else:
            with open(self.results_json_path, 'r') as f:
                prepared_data = json.load(f)
        
        for d in prepared_data:
            if "prepared" not in d or len(d["prepared"]) != self.prompts:
                d["prepared"] = [self.prepare_data(d, prompt['name']) for prompt in self.prompts]
        
        if self.debug_mode is not None:
            prepared_data = random.sample(prepared_data, self.debug_mode)
        
        pbar = tqdm(enumerate(prepared_data), total=len(prepared_data) if self.budget is None else self.budget)
        for idx, datum in pbar:
            if idx == 0:
                print(datum['prepared'])
            if 'prediction' in datum and datum['prediction'] is not None:
                continue
            if self.budget is not None and idx > self.budget:
                continue
            
            messages = [
                {"role": "system", "content": self.system_msg},
            ]
            prepared_data[idx]['content'] = []
            prepared_data[idx]['prediction'] = []
            for current_user_input, prompt in zip(datum['prepared'], self.prompts):
                messages.append({
                    "role": "user", "content": current_user_input
                })

                completion, error = delayed_completion(delay_in_seconds=self.delay, max_trials=self.max_trials, model=self.model_name, messages=messages, max_tokens=self.max_token, temperature=self.temperature)

                if completion is None:
                    pbar.set_description(f"Saving data after {idx + 1} inference.")
                    with open(self.results_json_path, 'w') as f:
                        json.dump(prepared_data, f, indent=4)
                    print(error)
                    # debug the error
                    breakpoint()
                else:
                    content, result = self.post_process(completion, prompt['name'])
                    prepared_data[idx]['content'].append(content)
                    prepared_data[idx]['prediction'].append(result)

                    assistent_message = completion.choices[0].message
                    messages.append(assistent_message)
                
            if idx % self.save_every == 0 and idx > 0:
                pbar.set_description(f"Saving data after {idx + 1} inference.")
                with open(self.results_json_path, "w") as f:
                    json.dump(prepared_data, f, indent=4)
        
        with open(self.results_json_path, "w") as f:
            json.dump(prepared_data, f, indent=4)
        
        print("Done")
    