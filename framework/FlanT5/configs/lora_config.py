from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    TaskType
)

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q",
    "v",
]
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    fan_in_fan_out=False,
    task_type="CAUSAL_LM",
)