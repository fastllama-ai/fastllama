import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import sys
import threading

# Colors
BLUE = "\033[94m"
GREEN = "\033[92m"
RESET = "\033[0m"

name = "jarvis"

spinner_frames = [".    ", "..   ", "...  ", ".... ", "....."]

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always calls the user sir. Your name is " + name + ".",
    },
    {
        "role": name,
        "content": "I am a friendly chatbot named "+ name +".",
    }
]

def slow_print(text, speed=0.05):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(speed)
    print()

print("Fastllama online")

print("\nPress CTRL + C to exit.")

try:
    while True:
        user_input = input(f"{BLUE}user: {RESET}")
        messages.append({"role": "user", "content": user_input})

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        done_event = threading.Event()
        output_container = {}

        def generate_model():
            output_container['outputs'] = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False
            )
            done_event.set()

        thread = threading.Thread(target=generate_model)
        thread.start()

        spinner_idx = 0
        while not done_event.is_set():
            sys.stdout.write(f"\r{GREEN}Generating {spinner_frames[spinner_idx]}{RESET}  ")
            sys.stdout.flush()
            spinner_idx = (spinner_idx + 1) % len(spinner_frames)
            time.sleep(0.1)

        thread.join()
        sys.stdout.write("\r" + " " * 20 + "\r")  # Clear line

        outputs = output_container['outputs']
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("|>\n")[-1].strip()

        slow_print(f"{GREEN}" + name + f": {response}{RESET}", speed=0.05)

        messages.append({"role": name, "content": response})
except KeyboardInterrupt:
    print(f"\n{BLUE}Exiting FastLlama.{RESET}")
    sys.exit(0)
