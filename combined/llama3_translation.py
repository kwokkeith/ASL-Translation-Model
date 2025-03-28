from ollama import ChatResponse, chat
import subprocess
import os

cpu_count = os.cpu_count()

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
            "./grammar/english.gbnf",
            "-t", cpu_count],
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

