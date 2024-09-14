import gradio as gr
from transformers import pipeline
import torch

pipe = pipeline("text-generation", "distilgpt2", torch_dtype=torch.bfloat16, device_map="auto")
roles = {
    "Dog": "Woof",  
    "Cat": "Meow"   
}

def respond(message, history, role):
    prompt = f"You are a {role}. Respond with different numbers of '{roles[role]}' based on the input, and add your emotion in parentheses."
    generated_response = pipe(prompt + f"\nUser: {message}\nAssistant:", max_new_tokens=50)[0]['generated_text']
    response = generated_response.split("Assistant:")[-1].strip()
    return response

demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Radio(choices=list(roles.keys()), label="Choose a role", value="Dog"),
    ],
)

if __name__ == "__main__":
    demo.launch()
