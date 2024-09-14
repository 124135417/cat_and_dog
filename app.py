import gradio as gr
from transformers import pipeline
import torch

"""
For more information on `huggingface_hub` Inference API support, please check the docs: https://huggingface.co/docs/huggingface_hub/v0.22.2/en/guides/inference
"""
pipe = pipeline("text-generation", "microsoft/Phi-3-mini-4k-instruct", torch_dtype=torch.bfloat16, device_map="auto")

roles = {
    "Dog": "You are a dog. You respond with different numbers of 'Woof.', and you will add your emotion on the end of the message, inside parenthesis.",
    "Cat": "You are a cat. You respond with different numbers of 'Meow.', and you will add your emotion on the end of the message, inside parenthesis."
}

def respond(message, history, role):
    system_message = roles[role]

    prompt = f"{system_message}\nUser: {message}\nAssistant:"
    output = pipe(prompt, max_new_tokens=100)

    response = output[0]['generated_text'].split("Assistant:")[-1].strip()

    return response

"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""
demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Radio(choices=list(roles.keys()), label="Choose a role", value="Dog"),
    ],
)

if __name__ == "__main__":
    demo.launch()