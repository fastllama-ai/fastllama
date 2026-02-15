import os
import time
import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import threading
import sys

# Colors
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"
name = "assistant"
spinner_frames = [".    ", "..   ", "...  ", ".... ", "....."]

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Get system information safely
def get_system_info():
    try:
        user_info = os.getlogin()
    except OSError:
        user_info = os.environ.get("USERNAME") or os.environ.get("USER") or "unknown"
    now = datetime.datetime.now()
    return {
        "user_info": user_info,
        "os_name": os.name,
        "current_time": now.strftime("%I:%M:%S %p"),  # 12-hour format with AM/PM
        "current_date": now.strftime("%Y-%m-%d")
    }


# Update system message
def update_system_message():
    info = get_system_info()
    return {
        "role": "system",
        "content": (
            f"[SYSTEM] Current system details:\n"
            f"User: {info['user_info']}\n"
            f"OS: {info['os_name']}\n"
            f"Date: {info['current_date']}\n"
            f"Time: {info['current_time']}\n"
            "This information is from the system and is not visible to the user.\n"
            "Assistant will always have access to real-time information."
        )

    }

# Spinner helper
def run_spinner(event, message="Loading", interval=0.2):
    idx = 0
    while not event.is_set():
        sys.stdout.write(f"\r{GREEN}{message} {spinner_frames[idx]}{RESET}")
        sys.stdout.flush()
        idx = (idx + 1) % len(spinner_frames)
        time.sleep(interval)
    sys.stdout.write("\r" + " " * 50 + "\r")  # clear line

# Display GPU info
def print_gpu_info():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        print(f"{YELLOW}Using device: {torch.cuda.get_device_name(0)}{RESET}")
        print(f"{YELLOW}CUDA version: {torch.version.cuda}{RESET}")
        print(f"{YELLOW}Total VRAM: {vram_total:.2f} GB{RESET}")
    else:
        device = torch.device("cpu")
        print(f"{YELLOW}CUDA not available, using CPU{RESET}")
    return device

# Load model
def load_model(device):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None
    )
    model.eval()
    return model

# Slow print
def slow_type(text, color=GREEN, delay=0.02):
    for char in text:
        sys.stdout.write(f"{color}{char}{RESET}")
        sys.stdout.flush()
        time.sleep(delay)
    print()

# Chat messages
messages = [
    {"role": "system", "content": "You are a friendly chatbot named assistant."},
    {"role": "assistant", "content": "I am assistant, at your service."},
]

# Update system message in conversation
def update_messages():
    messages[0] = update_system_message()

# -------- Main program --------
device = print_gpu_info()

done_event = threading.Event()
print(f"{GREEN}Booting model", end="")
spinner_thread = threading.Thread(target=lambda: run_spinner(done_event, "Booting"))
spinner_thread.start()

# Load model (optimized for GPU if available)
model = load_model(device)
done_event.set()
spinner_thread.join()

print(f"{GREEN}Model loaded successfully!{RESET}\n")

# Chat loop
try:
    while True:
        user_input = input(f"{BLUE}user: {RESET}")
        messages.append({"role": "user", "content": user_input})
        update_messages()

        # Prepare input prompt
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

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
        run_spinner(done_event, "Generating", interval=0.1)
        thread.join()

        outputs = output_container['outputs']
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("|>\n")[-1].strip()

        slow_type(f"{name}: {response}")
        messages.append({"role": "assistant", "content": response})

except KeyboardInterrupt:
    print(f"\n{BLUE}Exiting assistant.{RESET}")
    sys.exit(0)
