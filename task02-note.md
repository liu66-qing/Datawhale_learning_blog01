[Untitled1 (1).md](https://github.com/user-attachments/files/22443722/Untitled1.1.md)
### Blog0919
### 以下基于Datawhale教程文档进行我个人的一点补充

### 2.0使用大模型途径
##### 1.云端API调用
##### 2.本地化部署:隐私

### 2.1云端大模型调用
使用 openai 库的方式来调用云端大模型

##### step1:获取API Key：
sk-guogzwqpoxuwhaxqsqpzodqrcwekkeytgtbueorojsniqmhd

##### step2:调用大模型
安装openai库，用于调用大模型


```python
!pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```


```python
!pip install -q openai
```

### 调用云端API大模型


```python
from openai import OpenAI
client=OpenAI(api_key="sk-guogzwqpoxuwhaxqsqpzodqrcwekkeytgtbueorojsniqmhd",
              base_url="https://api.siliconflow.cn/v1")
response=client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[{'role':'user','content':"你好！"}],
    max_tokens=1024,
    temperature=0.7,
    stream=False
)
print(response.choices[0].message.content)
```

    
    
    你好！很高兴见到你😊 有什么我可以帮你的吗？无论是解答问题、提供建议，还是闲聊，我都很乐意与你交流！


#### * temperature（温度）：
范围0-1: 越小越保守，越大越抽象
#### * stream（流式输出）：
stream=True（流式输出）：一个字一个字蹦出来
stream=False（非流式输出）：全生成完一次性给结果

#### 封装成函数方便复用


```python
def chat_with_model(user_input: str, history: list = None, temperature: float = 0.7, system_prompt: str = None) -> str:
    #初始化 OpenAI 客户端就可以用云端 API 服务调用 AI 模型
    client=OpenAI(api_key="sk-guogzwqpoxuwhaxqsqpzodqrcwekkeytgtbueorojsniqmhd",
              base_url="https://api.siliconflow.cn/v1")
    # 初始化历史记录：存储之前的对话记录
    if history is None:
        history = []
    # 构建消息列表：要发送给 AI 模型的完整对话列表
    messages = []
    # 添加系统提示词：给 AI 模型看的“我希望你以某种特定身份或风格回答”
    if system_prompt:
        messages.append({"role": "system", "content": "黑人rapper"})
    # 添加历史对话
    for msg in history:
        messages.append(msg)
     # 添加当前用户输入
    messages.append({"role": "user", "content": user_input})
    # 调用API获取响应
    response = client.chat.completions.create(
        model="Qwen/Qwen3-8B",
        messages=messages,
        temperature=temperature
    )
    # 返回模型回复的文本
    return response.choices[0].message.content
    #调用函数
    print(chat_with_model("嘿bro"))
```


```python
-> str： 是个标记，说明这个函数最后会返回一个字符串
system_prompt：可以替换成想要的AI人设
```

 ### 2.2本地部署与调用


```python
from modelscope import snapshot_download

model_dir = snapshot_download('Qwen/Qwen3-4B-Thinking-2507', cache_dir='/root/autodl-tmp/model', revision='master')
```

    Downloading Model from https://www.modelscope.cn to directory: /root/autodl-tmp/model/Qwen/Qwen3-4B-Thinking-2507



```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 设置模型本地路径
model_name = "/root/autodl-tmp/model/Qwen/Qwen3-4B-Thinking-2507"

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",  # 自动选择合适的数据类型
    device_map="cuda:0",    # 自动选择可用设备(CPU/GPU)
    trust_remote_code=True
)

# 准备模型输入
prompt = "你好，请介绍一下自己"
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # 选择是否打开深度推理模式
)
# 将输入文本转换为模型可处理的张量格式
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# 生成文本
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768  # 设置最大生成token数量
)
# 提取新生成的token ID
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# 解析思考内容
try:
    # rindex finding 151668 (</think>)
    # 查找结束标记"</think>"的位置
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

# 解码思考内容和最终回答
thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

# 打印结果
print("thinking content:", thinking_content)
print("content:", content)
```

    `torch_dtype` is deprecated! Use `dtype` instead!



    Loading checkpoint shards:   0%|          | 0/3 [00:00<?, ?it/s]


    thinking content: 嗯，用户让我介绍一下自己。首先，我需要确定用户想要了解的是哪个“自己”。因为“我”可以指代很多不同的实体，比如人类、AI模型、软件程序等等。不过，考虑到用户是在和我（Qwen）对话，所以这里应该是指Qwen这个AI模型。
    
    接下来，我得回忆一下Qwen的基本信息。Qwen是通义千问系列中的一个模型，由阿里云研发。我应该先说明自己的身份，比如我是通义千问系列中的一个大语言模型，然后介绍我的主要功能和特点。
    
    用户可能想知道我有什么能力，比如回答问题、创作文字、编程、逻辑推理、多语言支持等。需要具体一点，但不要太过技术化。另外，可能用户还想知道我的应用场景，比如教育、娱乐、工作等。
    
    还要注意语气要友好、自然，避免太机械。可能需要分点说明，但用户要求的是“介绍”，所以可能用一段话或者几个简短的点。不过用户没有指定格式，所以可能用连贯的段落。
    
    另外，要避免错误信息。比如，Qwen是通义千问系列，而通义千问是阿里云的模型，不是通义实验室的。需要确认这一点。通义实验室是阿里云的，所以Qwen是阿里云研发的。这点要准确。
    
    可能用户有深层需求，比如想测试我的能力，或者想了解我是否适合他们的需求。所以介绍时可以提到我可以帮助用户解决各种问题，比如写故事、写公文、写邮件、写剧本等，这样用户知道我的适用范围。
    
    还要注意不要用太专业的术语，保持口语化。比如“多语言支持”可以换成“支持多种语言”。
    
    需要检查有没有遗漏的重要信息。比如，Qwen的版本，是否最新？不过用户可能不需要这么细，所以可能暂时不提版本号。
    
    另外，可能用户会问我的训练数据截止时间，但用户没问，可能不需要主动提，除非介绍里提到。不过原问题只是“介绍一下自己”，所以可能不需要。
    
    总结下来，回答应该包括：
    
    1. 我是通义千问系列中的一个大语言模型，由阿里云研发。
    2. 主要功能：回答问题、创作文字、编程、逻辑推理、多语言支持等。
    3. 应用场景：教育、娱乐、工作等。
    4. 语气友好，鼓励用户提问。
    
    可能需要避免太长的句子，分点但不用序号。比如用“我可以……”的句式。
    
    现在组织语言：
    
    你好！我是Qwen，是通义千问系列中的一个大语言模型，由阿里云研发。我能够回答问题、创作文字（比如写故事、写公文、写邮件、写剧本等）、进行逻辑推理、编程，以及表达观点等。我支持多种语言，可以帮助用户在学习、工作、娱乐等场景中获得帮助。如果你有任何问题或需要协助，随时告诉我哦！
    
    检查有没有错误：通义千问是系列，Qwen是其中的一个模型，可能需要说明Qwen是通义千问系列中的一个模型，但可能用户不太清楚系列，所以直接说“通义千问系列中的一个大语言模型”应该没问题。
    
    另外，是否要提到“通义实验室”？可能不需要，因为阿里云是通义实验室的母公司，但用户可能更熟悉阿里云。所以保持简单。
    
    可能用户会混淆Qwen和通义千问，所以需要明确Qwen就是通义千问系列中的模型，比如“我是通义千问系列中的一个大语言模型，Qwen是通义千问系列中的一个模型”。
    
    不过中文里通常说“通义千问”就是Qwen，所以可能直接说“我是通义千问（Qwen）”。
    
    不过用户可能已经知道Qwen是通义千问的英文名，所以需要确认。
    
    比如，通义千问的英文名是Qwen，所以可以说“我是通义千问（Qwen）”。
    
    不过用户可能更熟悉中文名，所以可能先说中文名再提英文名。
    
    比如：“你好！我是通义千问（Qwen），是阿里云研发的超大规模语言模型。”
    
    这样更清晰。
    
    再调整一下：
    
    你好！我是通义千问（Qwen），由阿里云研发的超大规模语言模型。我能够回答问题、创作文字（如写故事、公文、邮件、剧本等）、进行逻辑推理、编程，以及多语言支持。无论你是学习、工作还是娱乐，我都能尽力提供帮助。有什么问题或需要协助的，随时告诉我哦！
    
    这样应该比较全面，又不冗长。
    
    需要检查有没有重复或者冗余的地方。比如“创作文字”后面举例，可能用顿号分隔，但中文里用“、”分隔。比如“写故事、写公文、写邮件、写剧本等”。
    
    另外，“超大规模语言模型”可能需要确认是否正确，不过Qwen确实是超大规模的。
    
    可能用户会想知道我的训练数据截止时间，但问题中没有问，所以暂时不提。
    
    最后，保持友好和鼓励的语气，比如“有什么问题或需要协助的，随时告诉我哦！”
    
    这样应该符合用户的需求。
    </think>
    content: 你好！我是通义千问（Qwen），由阿里云研发的超大规模语言模型。我能够帮助你回答问题、创作文字（比如写故事、公文、邮件、剧本等）、进行逻辑推理、编程，还能支持多种语言。无论是学习、工作还是生活中的小问题，我都可以尽力帮你解决！如果你有任何需求或想试试看，随时告诉我哦～ 😊



```python
import os
model_path = "/root/autodl-tmp/model/Qwen/Qwen3-4B-Thinking-2507"
print(os.path.exists(model_path))  # 应该返回 True
```

    True



```python
# 准备模型输入
prompt = "你好，请介绍一下自己"
messages = [
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,#不把文本转换成 token直接保持为字符串
    add_generation_prompt=True,
    enable_thinking=True # 选择是否打开深度推理模式
)
print(text)
```

    <|im_start|>user
    你好，请介绍一下自己<|im_end|>
    <|im_start|>assistant
    <think>
    


##### <|im_start|> 和 <|im_end|> ：是 Qwen 模型常用的对话标记，表示消息的开始和结束
##### add_generation_prompt=True ：在末尾加一个提示（ assistant ）

#### 封装成函数


```python
def chat_with_local_model(model, tokenizer, user_input: str, history: list = None, temperature: float = 0.7, max_new_tokens: int = 1024, enable_thinking: bool = True) -> dict:
    
    # 初始化历史记录
    if history is None:
        history = []
    # 构建消息列表
    messages = []
    # 添加历史对话
    for msg in history:
        messages.append(msg)
    # 添加当前用户输入
    messages.append({"role": "user", "content": user_input})

    # 应用聊天模板
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking  # 选择是否打开深度推理模式
    )
    
    # 将输入文本转换为模型可处理的张量格式
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # 生成文本
    generated_ids = model.generate(    #生成token ID 序列，需用 tokenizer.decode 转成文字
        **model_inputs,   #用 ** 给model_inputs解包
        max_new_tokens=1024,  # 设置最大生成token数量
        temperature=0.7,
        do_sample=True if temperature > 0 else False  # 当temperature > 0时启用采样
    )
    
    # 提取新生成的token ID
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    
    # 解析思考内容
    try:
        # rindex finding 151668 (</think>)
        # 查找结束标记"</think>"的位置
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0
    
    # 解码思考内容和最终回答
    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    
    return {
        "thinking_content": thinking_content,
        "content": content
    }

# 使用示例
if __name__ == "__main__":
    
    # 单轮对话示例
    result = chat_with_local_model(
        model=model,
        tokenizer=tokenizer,
        user_input="你好，请介绍一下自己",
        temperature=0.7
    )
    
    print("\n==================== 单轮对话结果 ====================")
    print("thinking content:", result["thinking_content"])
    print("content:", result["content"])
    
    # 多轮对话示例
    history = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！我是Qwen，一个AI助手。"}
    ]
    
    result = chat_with_local_model(
        model=model,
        tokenizer=tokenizer,
        user_input="你能帮我写一首诗吗？",
        history=history,
        temperature=0.8
    )
    
    print("\n==================== 多轮对话结果 ====================")
    print("thinking content:", result["thinking_content"])
    print("content:", result["content"])
```

to(model.device)：把模型移到 GPU，可以解决数据和模型不在同一设备的问题

### 2.3使用 vLLM 进行高性能部署

transformers 本地部署计算效率有限，vLLM 这样的高性能推理框架可以提升模型吞吐量和响应速度


```python
from openai import OpenAI

client = OpenAI(api_key="xxx", 
                base_url="http://127.0.0.1:8000/v1")
response = client.chat.completions.create(
    model="Qwen3-4B",
    messages=[
        {'role': 'user', 'content': "你好哇"}
    ],
    max_tokens=512,
    temperature=0.7,
    stream=False
)
print(response.choices[0].message)
```
