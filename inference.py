from transformers import AutoTokenizer, AutoModelForCausalLM

final_output = "./outputs"

tokenizer = AutoTokenizer.from_pretrained(final_output, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(final_output, trust_remote_code=True)

import torch

# プロンプトの準備
prompt = "### Question:\n埼玉の県庁所在地は何市？\n\n### Answer:\n"

# 推論の実行
inputs = tokenizer(prompt, return_tensors="pt")
generated_tokens = model.generate(
    **inputs,
    max_new_tokens=64,
    pad_token_id=tokenizer.pad_token_id,
)[0]
generated_text = tokenizer.decode(generated_tokens)
print(generated_text)