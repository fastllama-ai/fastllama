import os
import time
import datetime
from datetime import timedelta
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import threading
import sys

# Colors
BLUE = "\033[94m"
GREEN = "\033[92m"
RESET = "\033[0m"
name = "jarvis"

spinner_frames = [".    ", "..   ", "...  ", ".... ", "....."]

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Get system information
def get_system_info():
    user_info = os.getlogin()  # Gets current logged in user
    os_name = os.name  # OS name (e.g., 'posix' or 'nt' for Windows)
    now = datetime.datetime.now()
    current_time = now.strftime("%H:%M:%S")
    current_date = now.strftime("%Y-%m-%d")
    timezone_offset = time.altzone // 60 if time.localtime().tm_isdst > 0 else time.timezone // 60
    return {
        'user_info': user_info,
        'os_name': os_name,
        'current_time': current_time,
        'current_date': current_date,
        'timezone_offset': timezone_offset
    }

# Update system message with real-time info
def update_system_message():
    system_info = get_system_info()
    return {
        "role": "system",
        "content": f"Current system details:\nUser: {system_info['user_info']}\nOperating System: {system_info['os_name']}\nDate: {system_info['current_date']}\nTime: {system_info['current_time']}\nTimezone offset: {system_info['timezone_offset']} minutes"
    }

# Spinner while loading model
def load_model():
    global model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    done_event.set()

# Slow type printing
def slow_type(text, color=GREEN, delay=0.02):
    for char in text:
        sys.stdout.write(f"{color}{char}{RESET}")
        sys.stdout.flush()
        time.sleep(delay)
    print()

# Start loading model with spinner
done_event = threading.Event()
print(f"{GREEN}Booting", end="")
thread = threading.Thread(target=load_model)
thread.start()

spinner_idx = 0
while not done_event.is_set():
    sys.stdout.write(f"\r{GREEN}Booting {spinner_frames[spinner_idx]}{RESET}")
    sys.stdout.flush()
    spinner_idx = (spinner_idx + 1) % len(spinner_frames)
    time.sleep(0.2)

thread.join()
sys.stdout.write("\r" + " " * 30 + "\r")  # Clear line after loading
print(f"{GREEN}Model loaded successfully!{RESET}")

# Initial messages list
messages = [
    {"role": "system", "content": "You are a friendly chatbot named jarvis."},
    {"role": "assistant", "content": "I am Jarvis, at your service."},
]

# Replace old system message
def update_messages():
    system_message = update_system_message()
    messages[0] = system_message

# Chat loop
try:
    while True:
        update_messages()
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
        sys.stdout.write("\r" + " " * 30 + "\r")  # Clear line

        outputs = output_container['outputs']
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("|>\n")[-1].strip()

        # Slow type response
        slow_type(f"{name}: {response}")

        messages.append({"role": "assistant", "content": response})

except KeyboardInterrupt:
    print(f"\n{BLUE}Exiting FastLlama.{RESET}")
    sys.exit(0)
