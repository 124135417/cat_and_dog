import gradio as gr
from transformers import pipeline
import torch

text_gen_model = pipeline("text2text-generation", model="google/flan-t5-small")

roles = {
    "Dog": "Woof",  
    "Cat": "Meow"   
}

def respond(message, history, role):
    prompt = f"You are a {role}. Respond with different numbers of '{roles[role]}' based on the user's message, and express an appropriate emotion."
    generated_response = text_gen_model(f"{prompt}\nMessage: {message}", max_new_tokens=30)[0]['generated_text']
    response = generated_response.strip()
    return response

demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Radio(choices=list(roles.keys()), label="Choose a role", value="Dog"),
    ],
)

if __name__ == "__main__":
    demo.launch()
