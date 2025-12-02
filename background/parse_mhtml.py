import email
from email import policy
from bs4 import BeautifulSoup
import sys

def parse_mhtml(file_path):
    with open(file_path, 'rb') as f:
        msg = email.message_from_binary_file(f, policy=policy.default)

    html_content = ""
    
    # 遍历 MHTML 的各个部分寻找 HTML 内容
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == "text/html":
                # 获取解码后的 payload
                payload = part.get_payload(decode=True)
                if payload:
                    html_content += payload.decode('utf-8', errors='ignore')
    else:
        if msg.get_content_type() == "text/html":
             payload = msg.get_payload(decode=True)
             if payload:
                html_content = payload.decode('utf-8', errors='ignore')

    if not html_content:
        print("No HTML content found.")
        return

    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 尝试提取对话内容，Gemini 的对话通常在特定的结构中
    # 这里简单提取所有文本，或者尝试寻找特定的类名
    # 根据用户提供的片段，内容似乎在 message-content 类中
    messages = soup.find_all(class_='message-content')
    
    if not messages:
        # 如果找不到特定类，回退到提取正文文本，但可能会有很多杂音
        print("No specific message content found, extracting generic text...")
        print(soup.get_text()[:2000]) # 打印前2000字符
    else:
        for i, msg_div in enumerate(messages):
            text = msg_div.get_text(separator='\n', strip=True)
            print(f"--- Message {i+1} ---")
            print(text)
            print("\n")

if __name__ == "__main__":
    file_path = "/Users/andrewlee/Desktop/Projects/LLMBayesian/_Gemini - 直接体验 Google AI 黑科技.mhtml"
    try:
        parse_mhtml(file_path)
    except Exception as e:
        print(f"Error: {e}")
