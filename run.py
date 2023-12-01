import os, json, fire
from datasets import load_dataset
from openai_tools import OpenAIGenerationBase

def load_data_banking77(split):
    dataset = load_dataset("PolyAI/banking77", revision="96a234bd25c04939c4a79213eb764ae90e4d0d81")
    data = dataset[split]
    label_names = data.features['label'].names
    data = list(data)
    for d in data:
        d['intent'] = label_names[d['label']]
    return data

class OpenAIGenerationCustom(OpenAIGenerationBase):
    def post_process(self, completion: str, prompt_name: str = None):
        content, _ = super().post_process(completion, prompt_name)
        # try loading a json prompt first
        try:
            if content.startswith("```json\n"):
                content = content.split("```json\n")[1]
                content = content.split("\n```")[0]
            result = json.loads(content)
        except:
            result = content
        return content, result

def main(
    generate_config_path: str,
    overwrite: bool = False,
    debug_mode: int = None,
    budget: int = None
):
    # config
    with open(generate_config_path, 'r') as f:
        generate_config = json.load(f)
    
    # prompt
    prompt_paths = [os.path.join("prompts", prompt) for prompt in generate_config['prompts']]
    
    # data
    data = load_data_banking77('test')
    
    generator = OpenAIGenerationCustom(data, prompt_paths, overwrite=overwrite, debug_mode=debug_mode, budget=budget, **generate_config)

    generator.generate_consecutive()

if __name__ == "__main__":
    fire.Fire(main)