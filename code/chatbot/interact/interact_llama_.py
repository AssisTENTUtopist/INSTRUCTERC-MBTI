import fire
from llama_cpp import Llama

SYSTEM_PROMPT = "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."


def get_message_tokens(model, role, content):
    content = f"{role}\n{content}\n</s>"
    content = content.encode("utf-8")
    message_tokens = model.tokenize(content, special=True)
    return message_tokens


def get_system_tokens(model):
    system_message = {
        "role": "system",
        "content": SYSTEM_PROMPT
    }
    return get_message_tokens(model, **system_message)


def interact(
    model_path,
    n_ctx=2000,
    top_k=30,
    top_p=0.9,
    temperature=0.2,
    repeat_penalty=1.1
):
    model = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_parts=1,
    )

    system_tokens = get_system_tokens(model)
    tokens = system_tokens
    model.eval(tokens)

    history=""

    while True:
        user_message = input("Пользователь: ")
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
        emo=emotion("../emotion/TinyLlama-Q2_K.gguf", history, "<Пользователь:\""+user_message+"\">")
        print(emo)
        print("="*80)
        history+="\t Пользователь:\""+user_message+"\" "
        user_message = user_message + "\t("+emo+")"
        print("Пользователь:", user_message)
        
        message_tokens = get_message_tokens(model=model, role="Пользователь", content=user_message)
        role_tokens = model.tokenize("Сайга\n".encode("utf-8"), special=True)
        tokens += message_tokens + role_tokens
        output = model.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {"role": "user", "content": tokens},
            ],
            response_format={
                "type": "json_object",
            },
            temperature=0.7,
        )
        history+="\t Сайга:\""
        token=output["choices"][0]["text"]
        tokens.append(token)
        print(token, end="", flush=True)
        history+=token
        history+="\" "
        print()

if __name__ == "__main__":
    fire.Fire(interact)
