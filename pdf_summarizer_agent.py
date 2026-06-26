import os
import json
import PyPDF2
from dotenv import load_dotenv
import openai
load_dotenv(".env")
api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    print("❌ 未找到 DASHSCOPE_API_KEY")
    exit(1)

print(f"✅ API Key 已加载: {api_key[:20]}...")

client = openai.OpenAI(
    api_key=api_key,
    base_url="https://llm-prh5rydio6k6g481.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
)

#工具函数

def read_pdf_content(file_path: str, max_pages: int = 20) -> str:
    try:
        if not os.path.exists(file_path):
            return f"❌ 文件不存在: {file_path}"
        text = []
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            total = len(reader.pages)
            pages = min(max_pages, total)
            for i in range(pages):
                page = reader.pages[i]
                txt = page.extract_text()
                if txt.strip():
                    text.append(f"--- 第{i+1}页 ---\n{txt.strip()}")
                else:
                    text.append(f"--- 第{i+1}页 ---\n[无文本]")
        full = "\n\n".join(text)
        if total > max_pages:
            full += f"\n\n... (共{total}页，仅显示前{max_pages}页)"
        return full if full.strip() else "❌ 未提取到文本"
    except Exception as e:
        return f"❌ 读取失败: {str(e)}"

def summarize_text(text: str, max_length: int = 500) -> str:
    if len(text) < 100:
        return "⚠️ 文本太短，无需总结:\n" + text
    try:
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": f"你是文档摘要专家，用中文总结，控制在{max_length}字内。输出：1.主题 2.核心观点 3.关键数据 4.总结"},
                {"role": "user", "content": f"总结：\n\n{text[:8000]}"}
            ],
            temperature=0.3,
            max_tokens=800
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ 总结失败: {str(e)}"

def list_pdf_files(directory: str = ".") -> str:
    try:
        if not os.path.exists(directory):
            return f"❌ 目录不存在: {directory}"
        files = [f for f in os.listdir(directory) if f.lower().endswith('.pdf')]
        if not files:
            return "📂 没有PDF文件"
        result = f"📂 目录 '{directory}' 下的PDF:\n"
        for i, f in enumerate(files, 1):
            size = os.path.getsize(os.path.join(directory, f)) / 1024
            result += f"  {i}. {f} ({size:.1f} KB)\n"
        return result
    except Exception as e:
        return f"❌ 错误: {str(e)}"

#工具定义

tools = [
    {
        "type": "function",
        "function": {
            "name": "read_pdf_content",
            "description": "读取PDF文件内容",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "PDF文件路径"},
                    "max_pages": {"type": "integer", "description": "最大页数，默认20", "default": 20}
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_text",
            "description": "对文本进行智能摘要",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "待总结文本"},
                    "max_length": {"type": "integer", "description": "摘要长度，默认500", "default": 500}
                },
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_pdf_files",
            "description": "列出目录下的PDF文件",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {"type": "string", "description": "目录路径，默认当前", "default": "."}
                }
            }
        }
    }
]

tool_map = {
    "read_pdf_content": read_pdf_content,
    "summarize_text": summarize_text,
    "list_pdf_files": list_pdf_files
}

#Agent 核心循环（带对话记忆）

# 消息历史保存在函数外部，跨轮次保留
conversation_history = []

def process_query(user_input: str) -> str:
    global conversation_history

    # 将用户输入加入历史
    conversation_history.append({"role": "user", "content": user_input})
    # 如果历史太长，只保留最近 10 条
    if len(conversation_history) > 10:
        conversation_history = conversation_history[-10:]

    messages = conversation_history.copy()

    for _ in range(5):
        try:
            response = client.chat.completions.create(
                model="qwen-plus",
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.7
            )
        except Exception as e:
            return f"❌ LLM 调用失败: {str(e)}"

        assistant_msg = response.choices[0].message

        if assistant_msg.tool_calls:
            # 将助手消息加入历史
            messages.append(assistant_msg.model_dump())
            conversation_history.append(assistant_msg.model_dump())

            for call in assistant_msg.tool_calls:
                tool_name = call.function.name
                args = json.loads(call.function.arguments)
                print(f"🔧 调用工具: {tool_name}({args})")
                result = tool_map[tool_name](**args)

                # 将工具结果加入历史
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result
                }
                messages.append(tool_msg)
                conversation_history.append(tool_msg)
            continue

        # 没有工具调用，将最终回复加入历史并返回
        conversation_history.append(assistant_msg.model_dump())
        return assistant_msg.content

    return "⚠️ 超过尝试次数"

def chat_loop():
    print("\n" + "=" * 50)
    print("📄 PDF 智能摘要助手（带对话记忆）")
    print("=" * 50)
    print("💡 命令示例:")
    print("  '有哪些PDF文件'")
    print("  '读取 文件.pdf'")
    print("  '总结 文件.pdf'")
    print("  'quit' 退出")
    print("=" * 50)

    while True:
        user_input = input("\n👤 你: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("👋 再见")
            break
        if not user_input.strip():
            continue

        print("🤖 思考中...")
        answer = process_query(user_input)
        print(f"🤖: {answer}")

if __name__ == "__main__":
    chat_loop()