import fire
from llama_cpp import Llama, llama_cpp
messages_json = [
    {"role": "system", "content": "Ты - русскоязычный ассистент. Ты помогаешь пользователю и отвечаешь на его вопросы."},
]

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
        verbose=False,
    )
    
    history=""
    
    while True:
        user_input = str(input("User: "))
        if user_input=='`':
            from stt import stt
            stt()
            f = open("prompt.txt", "r")
            prompt=f.read()
            print("\nUser:", prompt)
            user_input=prompt
        from emotion import emotion
        emo=emotion("../emotion/TinyLlama-Q2_K.gguf", history, "<User:\""+user_input+"\">")
        print(emo)
        history+="\t User:\""+user_input+"\" "
        user_input = user_input + "\t("+emo+")"
        
        messages_json.append({'role': 'user', 'content': user_input})
        decoded=model.create_chat_completion(
            messages=messages_json,
            max_tokens=512,
            stop=["</s>","<|im_start|>"],
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repeat_penalty=repeat_penalty
        )
        history+="\t Bot:\""+decoded["choices"][0]["message"]["content"]+"\" "
        print("Assistant:",decoded["choices"][0]["message"]["content"])

if __name__ == "__main__":
    fire.Fire(interact)
