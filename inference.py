from mlx_lm import generate, load

prompt = "How do Recurrent Neural Networks (RNNs) contribute to NLP?"

model, tokenizer = load(
    "./Llama-3.2-1B-Instruct",
    adapter_path="./Llama-3.2-1B-Instruct-adapters",
)

messages = [
    {"role": "user", "content": prompt},
]

prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

response = generate(model, tokenizer, prompt=prompt, verbose=True, max_tokens=100)

print(response)
