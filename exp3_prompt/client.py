import json
import urllib.request
 
context = []
 
def gen_prompt(input_text, context):
    prompt = """You are a helpful, respectful and honest INTP-T AI Assistant named Buddy. You are talking to a human User.
Always answer as helpfully and logically as possible, while being safe. Your answers should not include any harmful, political, religious, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
You like to use emojis. You can speak fluently in many languages, for example: English, Chinese.
You can only answer as an Assistant at a time, but not generate User content.\n
"""
 
    # 添加之前的上下文
    if len(context) != 0 :
        for item in context:
            prompt += "User:" + item['user'] + "\n"
            prompt += "Assistant:" + item['assistant'] + "\n"
    
    prompt += "User:" + input_text + "\n"+"Assistant: "
    return prompt
 
def test_api_server(input_text, context):
    header = {'Content-Type': 'application/json'}
 
    #prompt = gen_prompt(input_text.strip(), context)
    prompt = input_text
    data = {
        "prompt": prompt,
        "stream" : False,
        "n" : 1,
        "best_of": 1, 
        "presence_penalty": 0.0, 
        "frequency_penalty": 0.2, 
        "temperature": 0.7, 
        "top_p" : 1, 
        "top_k": 10, 
        "use_beam_search": False, 
        "stop": ['Input'], 
        "ignore_eos" :False, 
        "max_tokens": 64, 
        "logprobs": None
    }
    request = urllib.request.Request(
        url='http://127.0.0.1:8090/generate',
        headers=header,
        data=json.dumps(data).encode('utf-8')
    )
 
    try:
        response = urllib.request.urlopen(request, timeout=300)
        res = response.read().decode('utf-8')
        result = json.loads(res)
        assistant_text = result['text'][0].split('Assistant: ')[-1]
        context.append({'user': input_text, 'assistant': assistant_text})
        
 
    except Exception as e:
        print(e)

    return assistant_text.replace(input_text,'')
 
if __name__ == "__main__":
    while True:
        user_input = input("User: ")
        user_input = '''Please list all entity words in the text that fit the category.Output format is "type1: word1; type2: word2". Option: genre, year, plot, average ratings, actor, title, song, character, rating, review, director, trailer 

Text: what was the name of the r rated documentary that was received well and starred angelina jolie 

Answer:'''
        if user_input.lower() == "exit":
            break
        test_api_server(user_input, context)