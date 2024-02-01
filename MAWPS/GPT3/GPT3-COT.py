import openai
import json
import tqdm

openai.api_key = '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'

def load_json(file):
    with open(file,'r',encoding='utf-8') as f:
        data = json.load(f)
    return data
    
class GPT3:
    def __init__(self, config):
        self.config = config
        self.user = config['user']
        self.model = config['model']
        self.temperature = config['temperature']
        if config['ascii'] == 'ch':
            self.ensure_ascii = False
        else:
            self.ensure_ascii = True
        self.filename = config['save_file']
        
    def ask_gpt(self, prompt_text):
        rsp = openai.Completion.create(
            engine=self.model,
            prompt=prompt_text,
            max_tokens=100,
            temperature=self.temperature
        )
        response_txt = rsp.choices[0].text.strip()
        return response_txt

def configure():
    filename = f'configs/config.json'
    with open(filename, encoding="utf-8") as F:
        config = json.load(F)
    return config

def main_ans(case_q, case_a):
    config = configure()    
    model = GPT3(config)
    data_file = r"C:\Users\Administrator\Desktop\fold4_dev.json"
    # r"C:\Users\Administrator\Desktop\new_work\data\fold2\dev.json"
    data = load_json(data_file)
    new_data = []
    new_data_file = r"C:\Users\Administrator\Desktop\fold4_dev1.json"
    for i in tqdm.tqdm(range(len(data))):
        d = data[i]
        if "stat" in d and d["stat"] == 1:
            continue
        try:
            prompt = "Q: " + case_q + "A: " + case_a + "Q: " + d["question"]
            res = model.ask_gpt(prompt)
            d["GPT3_answer"] = res
            d["stat"] = 1
        except:
            d["stat"] = -1
        # prompt = "Q: " + case_q + "A:" + case_a + "Q: " + d["question"]
        # res = model.ask_gpt(prompt)
        # d["GPT3_answer"] = res
        # d["stat"] = 1

        new_data.append(d)
    with open(new_data_file, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)

case_q = "Bryan took a look at his books as well . If Bryan has 56.0 books in each of his 9.0 bookshelves , how many books does he have in total ?"
case_a = "9.0 multiplied by 56.0 equals 504.0. The answer is 504.0."
main_ans(case_q, case_a)