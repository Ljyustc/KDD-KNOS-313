# coding: utf-8
import openai
import time
import logging
import json

api_keys = [
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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'
]
api_bases = ["https://api.openai-proxy.com/v1",
             "https://api.openai-forward.com/v1",
             "https://cloudflare.worker.openai-forward.com/v1",
             "https://cloudflare.page.openai-forward.com/v1",
             "https://render.openai-forward.com/v1"]
class GPT3:
    def __init__(self, config):
        self.config = config
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
    def ask_chatgpt(self, prompt_text):
        rsp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_text}
        ]
        )
        response_txt = rsp.choices[0].message.content.strip()
        return response_txt
    
def configure():
    filename = f'./GPT3/configs/config.json'
    with open(filename, encoding="utf-8") as F:
        config = json.load(F)
    return config
