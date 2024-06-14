import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Replace 'username/model-name' with the correct path from the Model Hub
model_name= "Writer/palmyra-med-20b"
adapters_name = 'Sampe12/tuned-palmyra-med'

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    max_memory= {i: '24000MB' for i in range(torch.cuda.device_count())},
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
    ),
)
model = PeftModel.from_pretrained(model, adapters_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Function to generate text based on prompt
def generate_text(prompt):
    input_text = (
        "A medical assistant responds to a query about symptoms. "
        "The assistant provides detailed and informative answers. "
        f"PATIENT: {prompt}. Based on these symptoms, what could be the possible diagnosis and treatment plan?"
        "ASSISTANT:"
    )

    model_inputs = tokenizer(input_text.format(prompt=prompt), return_tensors="pt").to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    gen_conf = {
        "temperature": 0.7,
        "repetition_penalty": 1.0,
        "max_new_tokens": 512,
        "do_sample": True,
    }

    out_tokens = model.generate(**model_inputs, **gen_conf)

    response_ids = out_tokens[0][len(model_inputs.input_ids[0]) :]
    output = tokenizer.decode(response_ids, skip_special_tokens=True)

    return output