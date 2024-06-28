from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers import pipeline

# pipe = pipeline("text-generation", model="mistralai/Mixtral-8x7B-v0.1")

model_id = "mistralai/Mixtral-8x7B-v0.1"

# model_id = "mistralai/Mistral-7B-v0.3"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)



text = "Hello my name is"
inputs = tokenizer(text, return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))