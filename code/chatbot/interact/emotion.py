import fire
from llama_cpp import Llama

def edit_distance(s1, s2):
    """
    Calculate the editing distance between two strings
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
    return dp[m][n]
    
def optimize_output(output, label_set=["neutral", "positive", "negative"]):
    min_distance = float('inf')
    optimized_output = None
    for label in label_set:
        distance = edit_distance(output, label)
        if distance < min_distance:
            min_distance = distance
            optimized_output = label
    return optimized_output

def emotion(
    model_path,
    history,
    prompt,
    max_tokens=256,
    n_ctx=1024
):
    llm = Llama(
        model_path=model_path,
        verbose=False
    )
    
    full_prompt="Теперь ты эксперт анализа тональностей и эмоций. Следующий разговор, заключенный между '### ###', содержит несколько участников. ### " + history + prompt + " ### Пожалуйста, выбери эмоциональную метку для " + prompt + " из [neutral, positive, negative]. Это очень важно для моей карьеры."
    while len(full_prompt)>n_ctx:
        try:
            history=history[history[6:].index("User:")+5:]
        except:
            history=''
        full_prompt="Теперь ты эксперт анализа тональностей и эмоций. Следующий разговор, заключенный между '### ###', содержит несколько участников. ### " + history + prompt + " ### Пожалуйста, выбери эмоциональную метку для " + prompt + " из [neutral, positive, negative]. Это очень важно для моей карьеры."
    out = llm(
      full_prompt, # Prompt
      max_tokens=max_tokens,
    )
    return optimize_output(out["choices"][0]["text"])

if __name__ == "__main__":
    fire.Fire(emotion)
