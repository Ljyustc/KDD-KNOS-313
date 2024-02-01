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
            temperature=self.temperature
        )
        response_txt = rsp.choices[0].text.strip()
        return response_txt

def configure():
    filename = f'configs/config.json'
    with open(filename, encoding="utf-8") as F:
        config = json.load(F)
    return config

def main():  
    data_file = r"C:\Users\Administrator\Desktop\new_work\data\fold4\dev.json"
    data = load_json(data_file)
    exis_data = [r"C:\Users\Administrator\Desktop\fold4_dev.json",
                 r"C:\Users\Administrator\Desktop\fold4_dev1.json"]
    data_dict = {}
    for f in exis_data:
        e_d = load_json(f)
        for d in e_d:
            if d["stat"] == 1:
                data_dict[d["question"]] = d["GPT3_answer"]
    new_data = []
    new_data_file = r"C:\Users\Administrator\Desktop\fold4_devs.json"
    for i in tqdm.tqdm(range(len(data))):
        d = data[i]
        if d["question"] in data_dict:
            d["stat"] = 1
            d["GPT3_answer"] = data_dict[d["question"]]
        else:
            d["stat"] = -1
        new_data.append(d)
    with open(new_data_file, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)

def main1():  
    data_file = r"C:\Users\Administrator\Desktop\dev_cor.json"
    data_file1 = r"C:\Users\Administrator\Desktop\new_work\data\MAWPS\fold0\dev.json"
    data = load_json(data_file)
    data1 = load_json(data_file1)
    
    data_dict = {}
    cor_dict = {}
    for d in data:
        data_dict[d["question"]] = d["GPT3_answer"]
        cor_dict[d["question"]] = d["GPT3_correct"]
    new_data = []
    new_data_file = r"C:\Users\Administrator\Desktop\fold0_devs.json"
    for i in tqdm.tqdm(range(len(data1))):
        d = data1[i]
        if d["question"] in data_dict:
            d["GPT3_answer"] = data_dict[d["question"]]
            d["GPT3_correct"] = cor_dict[d["question"]]
        new_data.append(d)
    with open(new_data_file, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)

def main2():  
    data_file = r"C:\Users\Administrator\Desktop\new_work\data\MAWPS\fold0\dev.json"
    data = load_json(data_file)
    exis_data = [r"C:\Users\Administrator\Desktop\fold0_dev.json",
                 r"C:\Users\Administrator\Desktop\fold0_dev1.json",
                 r"C:\Users\Administrator\Desktop\fold0_dev2.json"]
    data_dict = {}
    for f in exis_data:
        e_d = load_json(f)
        for d in e_d:
            if d["exp_stat"] == 1:
                data_dict[d["question"]] = d["GPT3_explanation"]
    new_data = []
    new_data_file = r"C:\Users\Administrator\Desktop\devs.json"
    for i in tqdm.tqdm(range(len(data))):
        d = data[i]
        new_d = {"question":d["question"], "equation":d["equation"], "answer":d["answer"], "GPT3_answer":d["GPT3_answer"], "GPT3_correct":d["GPT3_correct"]}
        if d["question"] in data_dict:
            new_d["GPT3_explanation"] = data_dict[d["question"]]
        else:
            print(d)
        new_data.append(new_d)
    with open(new_data_file, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)

main()