"""最陽春的聊天介面：拿來「先和生成式 AI 討論人設」用。模型走 aisuite -> 本機 Ollama。"""
import sys
import aisuite as ai
import gradio as gr

MODEL = sys.argv[1] if len(sys.argv) > 1 else "ollama:qwen2.5:14b"
client = ai.Client(provider_configs={"ollama": {"timeout": 600}})

def chat(message, history):
    msgs = [{"role": "system", "content": "請用台灣習慣的繁體中文回覆，回答要具體。"}]
    for m in history:
        if isinstance(m, dict) and m.get("role") in ("user", "assistant"):
            msgs.append({"role": m["role"], "content": m["content"] if isinstance(m["content"], str) else str(m["content"])})
    msgs.append({"role": "user", "content": message})
    r = client.chat.completions.create(model=MODEL, messages=msgs, temperature=0.7)
    return r.choices[0].message.content

demo = gr.ChatInterface(chat, title=f"人設討論用聊天室（{MODEL}）", chatbot=gr.Chatbot(height=820))
demo.launch(server_name="127.0.0.1", server_port=7861, share=False, inline=False)
