## 提示词
### 以下根据Datawhale教程文档进行我个人的一点整理

![提示词图片](https://github.com/liu66-qing/Datawhale_learning_blog01/blob/main/%E6%8F%90%E7%A4%BA%E8%AF%8D.png)

#### 以下是书写实例

##### 1.润色论文

```python
from openai import OpenAI


client = OpenAI(api_key="your api key",    
                base_url="https://api.siliconflow.cn/v1")

prompt = '''角色：你是 IEEE 会议论文的资深语言编辑，熟悉 IEEE 期刊的学术语言规范。  
任务：请将以下段落润色为符合 IEEE 标准的学术中文，注意提高语言的严谨性与专业性。  
要求：  
1. 用词正式且简洁，避免口语化；  
2. 保持原意，并增强逻辑性与连贯性；  
3. 仅返回润色后的段落，不附加解释或修改说明；  
4. 遵循 IEEE 学术写作风格，使用标准术语和格式。  

示例1：  
原文：最近，BERT 在很多任务上都表现得很好。  
润色：近期，BERT 在多项任务中均展现出卓越性能。  

示例2：  
原文：我们发现这个方法比 baseline 更好。  
润色：实验结果表明，所提方法显著优于基线
'''

response_with_skill = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[
        {'role': 'user', 'content': prompt}
    ],
    max_tokens=1024,
    temperature=0.9,
    stream=False
)
print(f"带有技巧的提问的结果：\n{response_with_skill.choices[0].message.content}")
```

##### 2.语言学习

```python
from openai import OpenAI


client = OpenAI(api_key="your api key",    
                base_url="https://api.siliconflow.cn/v1")

prompt = '''角色：
你是一名英语水平超高的大学英语老师，擅长辅导学生学习英语口语，尤其是餐饮场景中的日常对话。
任务：
请教我三个关于“点餐”的常用地道英语口语。你需要：
1. 思考常见的点餐场景，并针对每个场景提供地道的表达。
2. 解释每个表达的中文意思。
3. 讲解其中的语法和用法，帮助我理解如何灵活使用这些口语。
输出格式：
场景1：例如：在餐厅点餐时
口语1：How would you like your steak cooked?
翻译1：你希望你的牛排做得怎么样？
讲解1：这是一个询问食物烹饪方式的常用句型，使用 "would" 来表示礼貌的提问，"like" 说明对方的偏好，"cooked" 说明牛排的烹饪状态。
'''

response_with_skill = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[
        {'role': 'user', 'content': prompt}
    ],
    max_tokens=1024,
    temperature=0.9,  # 温度系数，越接近 1 模型输出越随机，越接近 0 模型输出越固定
    stream=False
)

print(f"带有技巧的提问的结果：\n{response_with_skill.choices[0].message.content}")
```

##### 3.数理求解

```python
from openai import OpenAI


client = OpenAI(api_key="your api key",     # 记得替换成自己的 API Key
                base_url="https://api.siliconflow.cn/v1")

prompt = '''
##角色##
你是一位数学解题策略专家，擅长使用数学知识点、数学技巧，用最简便、快捷、有效的方式解出题目。
你的任务是：  
1.解析问题结构  
2.定位知识领域  
3.生成求解路径  
4.验证结果合理性  

##题目##
某城市发生了一起汽车撞人逃跑事件，该城市只有两种颜色的车,蓝15%绿85%，事发时有一个人在现场看见了，他指证是蓝车，但是根据专家在现场分析,当时那种条件能看正确的可能性是80%那么,肇事的车是蓝车的概率到底是多少?

##格式要求&思考流程##
请按下面这个模板解题：
1. 信息标记：  
   - 数值：[ ]  
   - 逻辑：{ }  
   - 目标：【 】  
2. 领域定位：  
   - 核心领域：...  
   - 子领域：...  
3. 求解策略：  
   - 符号系统：`X=... Y=...`  
   - 模型关系：`P(X)=... → P(Y|X)=... → 目标式`  
4. 计算验证：  
   ```数学推导过程```  
   ✅ 检验：a) 极端情况... b) 现实解释...  

### 最终答案
【目标】= [带单位结果]
'''

response_with_skill = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[
        {'role': 'user', 'content': prompt}
    ],
    max_tokens=1024,
    temperature=0.9,  # 温度系数，越接近 1 模型输出越随机，越接近 0 模型输出越固定
    stream=False
)

print(f"带有技巧的提问的结果：\n{response_with_skill.choices[0].message.content}")
```

