from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import pipeline


def load_model(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return model


def load_tokenizer(tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    return tokenizer


def generate_text_roberta(sequence):

    fill_mask = pipeline(
        "fill-mask",
        model="./Models/AzBERTo/",
        tokenizer="./Models/AzBERTo/"
    )

    res = fill_mask(sequence)

    return '\n'.join(item['sequence'] for item in res)


def generate_text_gpt_2(sequence):
    model = load_model("./Models/GPT2_1M/")
    tokenizer = load_tokenizer("./Models/GPT2_1M/")
    ids = tokenizer.encode(f'{sequence}', return_tensors='pt')

    final_outputs = model.generate(
        ids,
        do_sample=True,
        max_length=50,
        pad_token_id=model.config.eos_token_id,
        top_k=100,
        top_p=0.95,
    )

    generated_text = tokenizer.decode(final_outputs[0], skip_special_tokens=True)

    return generated_text.split('\n')[0] + "."
