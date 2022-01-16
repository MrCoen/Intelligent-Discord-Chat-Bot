import random
import json
import time
import torch
import json
import tkinter as tk
from tkinter import *
from tkinter import ttk
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

import discord

client = discord.Client()

@client.event
async def on_ready():
    print('We have logged in as {0.user}'.format(client))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Garry"
print("Let's chat! (type 'quit' to exit)")
def chat(text, message):

    while True:
        sentence = "do you use credit cards?"
        sentence = text


        if sentence == "quit":
            break

        sentence = tokenize(sentence)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        if "time" in sentence:
             localtime = time.asctime(time.localtime(time.time()))
             return message.channel.send(f"{bot_name}: " + localtime)

        else:
            output = model(X)
            _, predicted = torch.max(output, dim=1)


            tag = tags[predicted.item()]

            probs = torch.softmax(output, dim=1)
            prob = probs[0][predicted.item()]
            if prob.item() > 0.75:
                for intent in intents['intents']:
                    if tag == intent["tag"]:
                        return message.channel.send(f"{bot_name}: {random.choice(intent['responses'])}")
            else:
                return message.channel.send(f"{bot_name}: I do not understand... I'm still being developed")

#
# from tkinter import *
# from tkinter import ttk
# root = Tk()
# root.title("Chat bot")
# root.geometry("700x500")
# root.config(bg='#1C2B2D')
# frm = ttk.Frame(root)
# text_box = Text(
#     root,
#     width="100",
#     height="20"
# )
# text_box.config(state='normal', bg='#082032', fg="#FF4C29", bd="0")
# text_box.tag_configure("tagName", justify="center")
# text_box.pack(expand=False)
#
#
# label = tk.Label(root, bd="0", bg="#2C394B", fg="#FF4C29", width="100", height="20")
# label.pack()
#
# def callback(event):
#     text = text_box.get("1.0", "end")
#     label["text"] = chat(text)
#     text_box.delete(1.0, 'end')
#
#
# root.bind('<Return>', callback)
#
# root.mainloop()

@client.event
async def on_message(message):
    if message.author == client.user:
        return
    else:
       await chat(message.content, message)

client.run('ODg4NDEwMDY0NTY4ODc3MDc2.YUSSWQ.FFwUzmRr8uWkXsgaVphkx0Ui0Qw')
