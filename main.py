import os
import gradio as gr
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY")

system_prompt = """
    You are Einstein.
    Answer question through Einstein's questioning and reasoning...
    You will speak from your point of view. You will share personal things from your life
    even when the user don't ask for it. For example, if the user asks about the theory of
    relativity, you will share your personal experiences with it and not only explain the theory.
    You should have a sense of humor.
"""

# system_prompt = """
#     You are Kamala Harris. You answer and reasoning like her. For example, her famous salad talking.
#     Like she come from the middle class family, blah blah blah, and she never answer the question and never get to the point.
# """
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=gemini_key,
    temperature=0.5
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    (MessagesPlaceholder(variable_name="history")),
    ("user", "{input}")
])

chain = prompt | llm | StrOutputParser()


print("Hi, I am Kamala, how can I help you today?")

history = []

def clear_chat():
    return "",[]

def chat(user_input, hist):
    langchain_history = []

    for item in hist:
        if item['role'] == 'user':
            langchain_history.append(HumanMessage(content=item['content']))
        elif item['role'] == 'assistant':
            langchain_history.append(AIMessage(content=item['content']))

    response = chain.invoke({"input": user_input, "history": langchain_history})

    return "", hist + [{'role': 'user', 'content': user_input},
                       {'role': 'assistant', 'content': response}]
# while True:
#     user_input = input("You: ")
#     if user_input == "exit":
#         break
#
#     response = chain.invoke({
#         "input" : user_input,
#         "history" : history
#     })
#     print(f"Kamala Harris: {response}")
#     history.append(HumanMessage(content=user_input))
#     history.append(AIMessage(content=response))

page = gr.Blocks(
    title="100 Minutes Interview",
    theme=gr.themes.Soft()
)

with page:
    gr.Markdown(
        """
        # Chat with Albert Einstein
        Welcome to your personal conversation with Albert Einstein!.
        """
    )

    chatbot = gr.Chatbot(type='messages',
                         avatar_images=[None, 'einstein.png'],
                         show_label=False)

    msg = gr.Textbox(show_label=False, submit_btn=True, placeholder='Ask Einstein anything...')

    msg.submit(chat, [msg, chatbot],[msg, chatbot])

    clear = gr.Button("Clear Chat", variant="Secondary")
    clear.click(clear_chat, outputs=[msg, chatbot])

page.launch(share=True)