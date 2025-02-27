import ollama

prompt = "The below in enclosed single backtick is an output from a model intepreting a conversation.\n" + \
        "It may contain duplicates due to the nature of the model detection. Your job is to output " + \
        "an approximate phrase that the person is trying to say." 

def get_response(text):
    full_text = prompt + '`' + text + '`'
    response = ollama.chat(model='llama3.2', messages=[
    { 'role': "system", "content": "Just output the guessed phrase."},
    {'role': 'user', 'content': full_text}
    ])

    # print(response['message']['content'])
    return response['message']['content']
