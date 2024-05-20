import fire
from llama_cpp import Llama
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import GenerationConfig

SYSTEM_PROMPT = "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."
SYSTEM_TOKEN = 1788
USER_TOKEN = 1404
BOT_TOKEN = 9225
LINEBREAK_TOKEN = 13

ROLE_TOKENS = {
    "user": USER_TOKEN,
    "bot": BOT_TOKEN,
    "system": SYSTEM_TOKEN
}


def get_message_tokens(tokenizer, model, role, content):
    message_tokens = tokenizer.encode(content)
    message_tokens.insert(1, ROLE_TOKENS[role])
    message_tokens.insert(2, LINEBREAK_TOKEN)
    message_tokens.append(tokenizer.eos_token)
    return message_tokens


def get_system_tokens(tokenizer, model):
    system_message = {
        "role": "system",
        "content": SYSTEM_PROMPT
    }
    return get_message_tokens(tokenizer, model, **system_message)


def interact(
    model_path,
    n_ctx=2000,
    top_k=30,
    top_p=0.9
):

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to("cpu")
    model.eval()

    generation_config = GenerationConfig.from_pretrained(model_path)
    
    history=""

    while True:
        user_message = input("User: ")
        if user_message=='`':
            from stt import stt
            print("sst")
            print("="*80)
            stt()
            print("sst done")
            print("="*80)
            f = open("prompt.txt", "r")
            prompt=f.read()
            print(prompt)
            print("="*80)
            user_message=prompt
        from emotion import emotion
        emo=emotion("../emotion/TinyLlama-Q2_K.gguf", history, "<User:\""+user_message+"\">")
        print(emo)
        print("="*80)
        history+="\t User:\""+user_message+"\" "
        user_message = user_message + "\t("+emo+")"
        print("User:", user_message)
        
        data = tokenizer(user_message, return_tensors="pt")
        data = {k: v.to(model.device) for k, v in data.items()}
        generator = model.generate(
            **data,
            generation_config=generation_config,
            max_length= n_ctx,
            top_k=top_k,
            top_p=top_p
        )
        history+="\t Bot:\""
        for token in generator:
            token_str = tokenizer.decode(token.tolist())
            if token == tokenizer.eos_token:
                break
            print(token_str, end="", flush=True)
            history+=token_str
        history+="\" "
        print()

if __name__ == "__main__":
    fire.Fire(interact)
