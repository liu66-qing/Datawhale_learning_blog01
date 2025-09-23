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




## 作业用到了requests，以下是我额外补充学习的内容：
### 1.Requests库的使用


```python
!pip install requests   
```


```python
import   进口 requests   进口的请求
```

#### 发出GET请求

##### 最常用的 HTTP 方法之一是GET，它从指定资源检索数据

##### 调用 requests.get（）


```python
requests.get   得到("https://api.github.com")
```

#### 检查响应


```python
import   进口 requests   进口的请求
response = requests.get   得到("https://api.github.com")
```

#### 使用状态代码


```python
response.status_code
```




    200



##### 如果在布尔上下文 （如条件语句 ）中使用 Response 实例，则当status codes小于 400 时，它的计算结果为 True，否则计算结果为 False。


```python
if response:
    print("Success!")
else:
    raise Exception(f"Non-success status code: {response.status_code}")
```

    Success!

这个方法认为200-399的所有状态代码都是成功的，这个方法只能告诉大致成功没，想要更细致的处理需要单独判断每个状态码

##### 也可以不用if收订检查，.raise_for_status()会自动判断状态码，如果不是200-399会抛出异常


```python
import   进口 requests   进口的请求
from requests.exceptions import   进口 HTTPError

URLS = ["https://api.github.com", "https://api.github.com/invalid"]

for url in   在 URLS:
    try:
        response = requests.get   得到(url)
        response.raise_for_status()
    except HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"Other error occurred: {err}")
    else:
        print("Success!")
```

#### 访问响应内容

GET 请求的响应里，通常会包含一些有价值的信息，这部分内容就叫做 payload（有效载荷/消息体）.使用 Response 的属性和方法，您可以查看各种格式的有效负载


```python
#bytes
#返回原始字节数据（图片、文件）
response.content
```


```python
#str
#如果没指定编码（比如UTF-8），requests 会根据响应头（headers）猜测
response.text
#先指定再解码
response.encoding = "utf-8"
print(response.text)
```


```python
#dict
response.json()
```

#### 查看响应标头

响应标头可以提供信息，例如响应有效负载的内容类型以及缓存响应的时间。要查看这些标头 .headers：


```python
>>> import   进口 requests

>>> response >>> response = requests.get（"https://api.github.com"）= requests.get   得到("https://api.github.com")
>>> response.headers
{'Server': 'github.com',
...
'X-GitHub-Request-Id': 'AE83:3F40:2151C46:438A840:65C38178'}
```




    {'Server': 'github.com',
     'X-GitHub-Request-Id': 'AE83:3F40:2151C46:438A840:65C38178'}



.headers 属性返回一个类似字典的对象

若要查看响应有效负载的内容类型，可以访问 “Content-Type”：不区分大小写


```python
response.headers["content-type"]
```




    'application/json; charset=utf-8'



### 2.自定义GET请求

这节介绍如何在 GET 请求里加上查询参数，从而得到定制化的响应

#### 自定义响应头headers 参数


```python
headers = {"Accept": "application/json"}
```

举例：GitHub 搜索高亮匹配项

```python
import   进口 requests   进口的请求
response = requests.get   得到(
    "https://api.github.com/search/repositories",
    params={"q": '"real python"'},
    headers={"Accept": "application/vnd.github.text-match+json"},
)

```

params={"q": '"real python"'} → 搜索关键词 "real python"
Accept header 告诉服务器：客户端希望接收 text-match 格式的 JSON

### 3.使用其他HTTP方法

除了 GET 之外，其他流行的 HTTP 方法还包括 POST、PUT、DELETE、HEAD、PATCH 和 OPTIONS。与 get（） 类似


```python
import   进口 requests   进口的请求

requests.get   得到("https://httpbin.org/get")

requests.post("https://httpbin.org/post", data={"key": "value"})

requests.put("https://httpbin.org/put", data={"key": "value"})

requests.delete("https://httpbin.org/delete")

requests.head("https://httpbin.org/get")

requests.patch("https://httpbin.org/patch", data={"key": "value"})

requests.options("https://httpbin.org/get")

```




    <Response [200]>


POST → 新建数据

PUT → 更新数据

DELETE → 删除数据

HEAD → 只要响应头，不要响应体

PATCH → 局部更新

OPTIONS → 询问服务器支持哪些方法
上面用的 requests.get()、requests.post() 等，都是 高层函数。它们其实都是对 requests.request() 这个底层函数的封装。


```python
requests.request("GET", "https://httpbin.org/get")
```


```python
等价于
requests.get   得到("https://httpbin.org/get")
```

### 4.发送请求数据

根据 HTTP 协议的规范，某些请求方法（如 POST、PUT 和较少使用的 PATCH）会通过**消息体（message body）**来传递数据，而不是像 GET 请求那样通过 URL 中的查询字符串（query string，例如 ?key=value）传递数据

对于 POST、PUT 和 PATCH 请求，数据会放在消息体中发送，具体格式取决于服务器的要求（比如表单格式或 JSON 格式）。

##### (1)发送表单数据（application/x-www-form-urlencoded），可以用  字典或元组列表

可以通过 data 参数来传递消息体中的数据,data参数支持  字典、元组列表、字节、文件对象

```python
#方式一：字典
import   进口 requests   进口的请求
response = requests.post("https://httpbin.org/post", data={"key": "value"})
```

* 这里通过 requests.post() 向 https://httpbin.org/post 发送一个 POST 请求Requests 会自动将这个字典编码为 application/x-www-form-urlencoded 格式，并放入请求的消息体中#方式二：元组列表
  response = requests.post("https://httpbin.org/post", data=[("key", "value")])
  (2)发送 JSON 数据（application/json）
  使用 Requests 提供的 json 参数，而不是 data 参数

```python
response = requests.post("https://httpbin.org/post", json={"key": "value"})
json_response = response.json()
print(json_response["data"])  
print(json_response["headers"]["Content-Type"]) '
```

### 5.检查准备好的请求

在 Python 的 Requests 库中，当你发起一个 HTTP 请求（比如 POST、GET 等）时，Requests 不会直接将请求发送到服务器，而是先对请求进行“准备”。这个准备过程会生成一个 PreparedRequest 对象，它包含了请求的所有细节准备过程包括：
验证头信息的有效性（比如确保 Content-Type 正确）。
序列化数据（比如将 Python 字典转换为 JSON 字符串）。
编码数据（比如将数据转换为适合网络传输的格式，如字节串）。
访问 PreparedRequest 对象
在发送请求并收到响应后，你可以通过 Response 对象的 .request 属性访问对应的 PreparedRequest 对象。用来检查

```python
#查看请求的头信息
print   打印(response.request.headers["Content-Type"])  # 输出：'application/json'

#查看请求的 URL
print   打印(response.request.url)  # 输出：'https://httpbin.org/post'

#查看请求的消息体
print   打印(response.request.body)  # 输出：b'{"key": "value"}'
```

### 6.使用身份认证

认证（Authentication）让服务器知道“你是谁”，确保只有授权用户才能访问特定资源Requests 库通过 auth 参数支持多种认证方式，常见的有 HTTP Basic Authentication 和自定义认证

#### （1）HTTP Basic Authentication


```python
import   进口 requests   进口的请求
response = requests.get   得到("https://httpbin.org/basic-auth/user/passwd", auth=("user", "passwd"))
print   打印(response.status_code)  # 输出：200
print   打印(response.request.headers["Authorization"])  # 输出：'Basic dXNlcjpwYXNzd2Q='
```

    200
    Basic dXNlcjpwYXNzd2Q=


#### （2）其他内置认证方式

HTTPDigestAuth：用于更安全的摘要认证（Digest Authentication）。
HTTPProxyAuth：用于代理认证。

### 7.与服务器安全通信

当处理敏感数据（如密码、个人信息）时，必须通过加密连接确保安全。HTTP 安全通信使用 TLS（传输层安全性），它是 SSL（安全套接字层）的升级版，提供更强的加密和效率。尽管如此，程序员常将 TLS 称为 SSL

##### 提高性能

让 Requests 更高效
（1）设置请求超时
默认情况下，Requests 会无限等待响应，可能导致程序卡住。使用 timeout 参数设置超时时间，防止阻塞


```python
response = requests.get   得到("https://api.github.com", timeout=1)  # 1秒超时
```


```python
response = requests.get   得到("https://api.github.com", timeout=(3.05, 5))  # 连接3.05秒，读取5秒
```

超时会抛出 ConnectTimeout（连接失败）或 ReadTimeout（读取失败），可用 try-except 捕获：

```python
try:
    response = requests.get   得到("https://api.github.com", timeout=(3.05, 5))
except requests.exceptions.Timeout:
    print   打印("The request timed out")
```

（2）使用 Session 对象重用连接
Session 对象可以跨请求保持参数（如认证、头信息）和连接，提高性能。


```python
from requests import   进口 Session
from custom_token_auth import   进口 TokenAuth

TOKEN = "<YOUR_GITHUB_PA_TOKEN>"
with Session() as session:
    session.auth = TokenAuth(TOKEN)
    response1 = session.get   得到("https://api.github.com/user")
    response2 = session.get   得到("https://api.github.com/user")
```

（3）重试失败的请求  Requests 默认不重试失败请求，但可以通过自定义 传输适配器 实现自动重试

（4）Requests 不支持异步请求。如果需要异步，推荐使用 AIOHTTP 或 HTTPX（后者与 Requests 语法兼容）

