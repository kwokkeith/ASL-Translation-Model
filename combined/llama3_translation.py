from ollama import ChatResponse, chat
import subprocess

prompt = "The below in enclosed single backtick is an output from a model intepreting a conversation.\n" + \
        "It may contain duplicates due to the nature of the model detection. The inputs may contain \n" +\
        "phrases and individual alphabets. Your job is to output " + \
        "an approximate phrase that the person is trying to say." 

# def get_response(text):
#     full_text = prompt + '`' + text + '`'
#     response = ollama.chat(model='llama3', messages=[
#     { 'role': "system", "content": "Just output the guessed phrase."},
#     {'role': 'user', 'content': full_text}
#     ])

#     # print(response['message']['content'])
#     return response['message']['content']

# grammar = 'root::= optional-greeting WS? (question | statement) \n' + \
#             'optional-greeting ::= greeting-word |  \n' + \
#             'greeting-word ::= "Hello" \n' + \
#             'question ::= interrogative-pronoun WS verb WS possessive-pronoun WS noun "?" \n' + \
#             'statement ::= possessive-pronoun WS noun WS verb WS proper-noun (WS proper-noun)* \n' + \
#             'interrogative-pronoun ::= "What" \n' + \
#             'possessive-pronoun ::= "My" | "Your" \n' + \
#             'verb ::= "is" \n' + \
#             'noun ::= "name" \n' + \
#             'proper-noun ::= capital-word+ \n' + \
#             'capital-word ::= capital-letter+ \n' + \
#             'capital-letter ::= [A-Z] \n' + \
#             'lowercase-letter ::= [a-z] \n' + \
#             'WS ::= " " | "\t" | "\n"'

def get_response(text):
    result = subprocess.run(["./llama-cli/llama-cli", 
            "-m",
            "./model/ggml-model-Q4_K_M.gguf",
            "--jinja",
            "--single-turn",
            "-sys",
            "You are an interpreter trying to convert ASL phrases and fingerspelled names into English sentences. You must strictly follow the grammar provided",
            "-p",
            text,
            "--grammar-file",
            "./grammar/english.gbnf"],
            capture_output=True,
            text=True)
    s = result.stdout
    
    # Find the index of '[end of text]'
    end_marker = "[end of text]"
    end_index = s.find(end_marker)
    
    if end_index != -1:
        # Get the substring before '[end of text]'
        before_end = s[:end_index]
    
        # Find the last newline before '[end of text]'
        last_newline_index = before_end.rfind('\n')
    
        # Extract the line just before '[end of text]'
        line_before_end = before_end[last_newline_index + 1:].strip()
    
        # return the sentence
        return line_before_end
    else:
        print("[end of text] not found.")
        return ""

