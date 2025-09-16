### 以下基于Datawhale教程文档进行我个人的一点补充

------

### 1.1.1NLP传统任务

###### 传统NLP模型工作模式就是专业，单一。

我觉得在这种模式下，想要实现一个联合型项目（比如从文本中识别关键词并翻译）工作量大需要训练好几个模型。使用时需要在特定任务的数据集上再训练模型——微调

### 1.1.2大模型

###### 大模型：参数规模大，训练数据量大。通用，在训练时让它根据前文预测下一个token

通过prompting来解决任务，基本不需要额外训练——提示 

------

### 1.2.1环境配置

代码前面有 ！表示是命令行指令在终端运行

```python
!nvcc --version#查看当前的 CUDA 版本
```

```python
!python -c "import torch; print(torch.__version__)"#查看当前的 Pytorch 版本
```

```python
!python -c "import torch; print(torch.cuda.is_available())"#测试一下 CUDA 是否可用：
```

### 1.2.2 配置模型下载和运行环境

##### 平台

###### Hugging Face : https://huggingface.co/

ModelScope : https://www.modelscope.cn/

###### Hugging Face和ModelScope两个平台都有大量预训练模型、数据集和工具

##### 镜像源

在下载模型前可以先配置pip镜像源，这样在国内可以快速安装python包

```python
# 配置 pip 镜像源
!pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
#永久配置 pip 的下载源为清华镜像
```

### 1.2.3Hugging Face 下载模型

##### 安装依赖

###### 安装 Hugging Face 提供的 huggingface_hub 库，需要里面的huggingface-cli 命令行工具来下载上传模型。

```python
!pip install -U huggingface_hub
```

##### 下载模型

```Python
import os

# 设置国内下载镜像地址
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'#给程序临时设置环境变量 HF_ENDPOINT

# 下载模型
os.system('huggingface-cli download --resume-download Qwen/Qwen3-0.6B --local-dir /root/autodl-tmp/model/Qwen/Qwen3-0.6B')
```

--resume-download：断点续传

`Qwen/Qwen3-0.6B`：千问，这里替换成其他模型名称

`--local-dir your_path`：这里替换成本地路径，Linux要写绝对路径，Windows 要写相对路径

### 1.2.4Modelscope下载模型

##### 安装依赖

```python
!pip install -q modelscope transformers accelerate
#accelerate配合transformers可以让模型推理更快
```

##### snapshot_download下载模型

###### `modelscope`中的`snapshot_download`函数下载模型，第一个参数为模型名称，参数`cache_dir`为模型的下载路径(绝对路径)

```python
from modelscope import snapshot_download, AutoModel, AutoTokenizer
# AutoModel自动选择合适的模型类   AutoTokenizer自动加载分词器转变成token

model_dir = snapshot_download('Qwen/Qwen3-0.6B', cache_dir='/root/autodl-tmp/model', revision='master')
```

------

### 1.3 transformers 加载模型并推理

###### 我把教程大段代码分块理解

##### 导入 transformers 库 &并加载模型

```python
# 导入必要的transformers库组件
from transformers import AutoModelForCausalLM, AutoTokenizer
#这里的AutoTokenizer会覆盖掉modelscope里那个

# 设置模型本地路径
model_name = "/root/autodl-tmp/model/Qwen/Qwen3-0.6B"

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
#trust_remote_code=True允许加载并执行作者提供的代码

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",  # 自动选择合适的数据类型
    device_map="auto",    # 自动选择可用设备(CPU/GPU)
    trust_remote_code=True
)
```

* AutoTokenizer： 是通用接口，会根据给的模型名称自动选择对应的分词器。起到分词器作用的是tokenizer

  * 用法：编码：tokenizer.encode(文字)  返回‘List’     

    ​			解码：tokenizer.decode()

  * 也可以指定

    "pt" → PyTorch 张量（最常用）  tokenizer(text, return_tensors="pt") 

    "tf" → TensorFlow 张量

    "np" → NumPy 数组

* AutoModelForCausalLM：加载因果语言模型，CausalLM从左至右根据前一个文字预测后一个文字



##### 构造输入消息

```python
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
```

* `apply_chat_template`：转成模型能读懂的格式

* tokenize=False : 返回 字符串        tokenize=True :返回 token id
* add_generation_prompt=True 会在文本末尾 自动加上 <|assistant|>，模型看到 `<|assistant|>` 才会开始生成文本



##### 调用模型生成

```Python
# 将输入文本转换为模型可处理的张量格式
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# 生成文本
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768  # 设置最大生成token数量
)
```

* 用 generate() 生成新 token就是模型的输出，token数量足够多才能生成长文本

##### 提取新生成的 token

```python
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
```

* 输入部分切掉，只保留 新生成的token

##### 输出结果

```python
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


