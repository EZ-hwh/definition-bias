import openai
import time

openai.api_key = '' # your openai api key

def chatgpt(query):
    query_session = [{"role":"user", "content": query}]
    #time.sleep(60)
    resp = openai.ChatCompletion.create(
                model='gpt-4-1106-preview',
                messages=query_session,
                temperature=0.1,
                max_tokens=4096,
                top_p=1,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                request_timeout=60
            )
    ret = resp.choices[0]['message']['content']
    return ret

if __name__ == '__main__':
    print((chatgpt('How to use python code to calculate 1+1?')))