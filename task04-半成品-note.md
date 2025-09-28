## BLOG 0926
9.26在高铁上用手机写的很粗糙，在28号回校后补充完整
## 微调大模型
提示词局限：
1. 有限上下文LLM难理解复杂任务要求
2. 通用LLM不具备领域知识
3. 需要较小的LLM

## 4.1 环境配置

将使用 datasets、transformers、peft 等框架完成从微调数据构造到高效微调模型的整体微调流程
需要安装的第三方库
```
!pip install -q datasets pandas peft
!pip install transformers
```

## 4.2 微调数据集构造

### 4.2.1 有监督微调（SFT）

#### 训练LLM： 预训练——有监督训练——人类反馈强化学习

预训练：预训练语料提供海量知识SFT： 问题答案都给模型，让模型照着答案学解问题（将输入和输出同时给模型，让他根据输出不断去拟合从输入到输出的逻辑）

#### 传统NLP：针对每一个任务对模型进行微调，比如情感分类，构造很多输入文本和其情感判断的数据，让模型去学会如何判断输入文本的情感。

#### LLM：指令微调，型能够泛化地处理多种类型的指令，而不是只针对某个特定任务（如情感分类）进行优化。

使用指令数据对模型SFT，指令数据集应该遵循以下三个键：
```
{
    "instruction": "即输入的用户指令",
    "input": "执行该指令可能需要的补充输入，没有则置空",
    "output": "即模型应该给出的回复"
}
```
为了让模型学会与预训练阶段不同的工作方式（即从简单的语言预测转变为理解和响应用户指令），在 SFT 的过程中，往往会针对性设置特定格式

LLaMA 的 SFT 格式为

### Instruction:\n{{content}}\n\n### Response:\n

这里的 content 不是只包含 instruction（用户任务描述，比如“翻译成英文”），而是把 instruction 和 input（补充输入，比如具体的句子“今天天气真好”）拼接成一个完整的指令

### 4.2.2 构造微调数据集

### 1. 数据加载检查
```
# 加载第三方库
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer

# 将JSON文件转换为CSV文件
df = pd.read_json('huanhuan.json')
ds = Dataset.from_pandas(df)

ds[0]
```
**tip:用的是远程服务器,所以需要把 huanhuan.json 上传到当前工作目录，然后用相对路径读取**

### 2. 数据处理函数

Dataset → Qwen 指令格式 → tokenized → 训练数据
不同LLM的指令格式不同，首先查看Qwen-3-4B 的指令格式:
```
# 加载模型 tokenizer 
tokenizer = AutoTokenizer.from_pretrained('model/Qwen/Qwen3-4B-Instruct-2507', trust_remote=True)

# 打印一下 chat template
messages = [
    {"role": "system", "content": "===system_message_test==="},
    {"role": "user", "content": "===user_message_test==="},
    {"role": "assistant", "content": "===assistant_message_test==="},
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enablre_thinking=True
)
print(text)
```
apply_chat_template：把 messages 按照 Qwen 预定义的对话模板（messages）转成模型需要的字符串格式
```
基于上文打印的指令格式，完成数据集处理函数
```

```
def process_func(example):
    MAX_LENGTH = 1024 # 设置最大序列长度为1024个token
    input_ids, attention_mask, labels = [], [], [] # 初始化返回值
    # 适配chat_template
    instruction = tokenizer(
        f"<s><|im_start|>system\n现在你要扮演皇帝身边的女人--甄嬛<|im_end|>\n" 
        f"<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n"  
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n",  
        add_special_tokens=False   
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    # 将instructio部分和response部分的input_ids拼接，并在末尾添加eos token作为标记结束的token
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    # 注意力掩码，表示模型需要关注的位置
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    # 对于instruction，使用-100表示这些位置不计算loss（即模型不需要预测这部分）
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]  
    if len(input_ids) > MAX_LENGTH:  # 超出最大序列长度截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
    ```
    ```
   # 使用上文定义的函数对数据集进行处理
tokenized_id = ds.map(process_func, remove_columns=ds.column_names)
tokenized_id
    ```
    ```
    # 最终模型的输入
print(tokenizer.decode(tokenized_id[0]['input_ids']))
```

## 4.3 高效微调-LoRA
模型微调方法：
全量微调：在SFT过程中更新全部参数，成本高
高效微调：通过向模型插入新的层，在微调时仅更新新层少量参数
         LoRA微调：低秩矩阵层插入
                  只更新插入的低矩阵参数  
                  模型学习简单任务   
                  推理时合并LoRA参数：在实际使用模型进行预测时，LoRA 插入的低秩矩阵参数会被与原始模型的参数合并


###### 用peft库来实现LoRA微调
```
#首先配置 LoRA 参数
from peft import LoraConfig, TaskType, get_peft_model

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, # 任务类型为 CLM，即 SFT 任务的类型
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # 目标模块，即需要进行 LoRA 微调的模块
    inference_mode=False, # 训练模式
    r=8, # Lora 秩，即 LoRA 微调的维度
    lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1# Dropout 比例
)
config
```
```
import torch
# 加载基座模型
model = AutoModelForCausalLM.from_pretrained('model/Qwen/Qwen3-4B-Instruct-2507', device_map="auto",torch_dtype=torch.bfloat16)
# 开启模型梯度检查点能降低训练的显存占用
model.enable_input_require_grads()
# 通过下列代码即可向模型中添加 LoRA 模块
model = get_peft_model(model, config)
config
```
```
# 查看 lora 微调的模型参数
model.print_trainable_parameters()
```
使用Swanable进行训练监测，查看训练的loss情况，GPU利用率
Swanlab是一个专注于机器学习实验追踪和可视化的MLOps工具，将各种关键指标数据实时展示在美观的网页上安装Swanlab
```
!pip install swanlab
```

```
# 配置 swanlab
import swanlab
from swanlab.integration.transformers import SwanLabCallback

swanlab.login(api_key='your api key', save=False)

# 实例化SwanLabCallback
swanlab_callback = SwanLabCallback(
    project="Qwen3-4B-lora", 
    experiment_name="Qwen3-4B-experiment"
)
```

```
swanlab 的 api key 通过登录官网注册账号获得：https://swanlab.cn/
```

```
from swanlab.integration.transformers import SwanLabCallback

# 配置训练参数
args = TrainingArguments(
    output_dir="./output/Qwen3_4B_lora", # 输出目录
    per_device_train_batch_size=16, # 每个设备上的训练批量大小
    gradient_accumulation_steps=2, # 梯度累积步数
    logging_steps=10, # 每10步打印一次日志
    num_train_epochs=3, # 训练轮数
    save_steps=100, # 每100步保存一次模型
    learning_rate=1e-4, # 学习率
    save_on_each_node=True, # 是否在每个节点上保存模型
    gradient_checkpointing=True, # 是否使用梯度检查点
    report_to="none", # 不使用任何报告工具
)
```
```
# 然后使用 trainer 训练即可
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[swanlab_callback]
)
```
###### LoRA 微调仅保存微调后的 LoRA 参数，因此推理微调模型需要加载 LoRA 参数并合并
```
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

mode_path = 'model/Qwen/Qwen3-4B-Instruct-2507'# 基座模型参数路径
lora_path = './output/Qwen3_4B_lora/checkpoint-351' # 这里改成你的 lora 输出对应 checkpoint 地址

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(mode_path)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(mode_path, device_map="auto",torch_dtype=torch.bfloat16, trust_remote_code=True)

# 加载lora权重
model = PeftModel.from_pretrained(model, model_id=lora_path)
```
```
prompt = "你是谁？"
inputs = tokenizer.apply_chat_template(
                                    [{"role": "user", "content": "假设你是皇帝身边的女人--甄嬛。"},{"role": "user", "content": prompt}],
                                    add_generation_prompt=True,
                                    tokenize=True,
                                    return_tensors="pt",
                                    return_dict=True,
                                    enable_thinking=False
                                )
inputs = inputs.to("cuda")


gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
with torch.no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

