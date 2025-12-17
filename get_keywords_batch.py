import pandas as pd
import asyncio
from openai import AsyncOpenAI  # 导入异步客户端
import json
import os


class GetKeywords:
    def __init__(self, base_url, api_key, model_name, thinking=False, max_concurrent=5):
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.thinking = thinking
        self.max_concurrent = max_concurrent  # 最大并发数
        # 构建系统提示词模板
        self.system_message = self._build_system_message()
        # 初始化异步OpenAI客户端
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    def _build_system_message(self) -> str:
        """
        构建系统提示词，从问题中抽取出专业术语和认为需要澄清的术语
        - 样例问题：近期监管和公司制度对销售适当性有哪些新增规定
        - 样例输出：
        {
            "专业术语": [
                {
                    "销售适当性": "销售适当性是指金融机构在销售过程中必须确保其产品和服务符合客户的风险承受能力和其他相关情况。"
                }
            ],
            "澄清术语": [{"近期": "近三个月"}],
        }
        """
        example_q = "近期监管和公司制度对销售适当性有哪些新增规定"
        example_a = {
            "专业术语": [
                {
                    "销售适当性": "销售适当性是指金融机构在销售过程中必须确保其产品和服务符合客户的风险承受能力和其他相关情况。"
                }
            ],
            "澄清术语": [{"近期": "近三个月"}],
        }
        example = f"样例如下：\n输入：{example_q}。输出json格式：{json.dumps(example_a, ensure_ascii=False)}"
        # 构建提示词系统角色

        system_message = (
            "你是一个审计专家，给你一个问题，请帮我从问题中抽取出专业术语和你认为需要澄清的术语，还需给出你认为的术语定义。专业术语指具备审计行业的独特性定义，好的例子如“销售适当性”是审计已销售的产品是否适合销售对象，不好的例子如”产险“、”寿险“、”农险“是个常识性的术语。澄清术语指回答范围和要求模糊不清，好的例子如“近期”没体现出具体是多久、“优秀案例”没体现优秀的定义，不好的例子如”哪些“、”我“都不是回答范围的模糊。抽取要求：1.只需要抽取原文的内容，不需要额外创造；2.术语只要词汇，不需要句子。%s"
            % example
        )

        return system_message

    async def _process_single_question(self, question: str) -> dict:
        """处理单个问题的异步方法"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_message},
                    {"role": "user", "content": question},
                ],
                temperature=0.5,
                max_tokens=1024,
                stream=False,
                extra_body={"enable_thinking": self.thinking},
            )
            return {
                "question": question,
                "response": response.choices[0].message.content,
                "success": True,
            }
        except Exception as e:
            print(f"处理问题出错: {question}, 错误: {str(e)}")
            return {"question": question, "response": str(e), "success": False}

    async def process_batch(self, questions: list[str]) -> list[dict]:
        """并发处理批量问题"""
        # 使用信号量控制并发数
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def bounded_task(question: str) -> dict:
            async with semaphore:  # 限制并发数量
                return await self._process_single_question(question)

        # # 思路1：创建所有任务并等待完成
        # tasks = [bounded_task(q) for q in questions]
        # results = await asyncio.gather(*tasks)

        # 思路2：逐个打印完成进度
        # 创建任务列表
        tasks = [asyncio.create_task(bounded_task(q)) for q in questions]
        total = len(tasks)
        results: list[dict] = []
        completed = 0
        # 按完成顺序收集，并打印进度
        for coro in asyncio.as_completed(tasks):
            res = await coro
            results.append(res)
            completed += 1
            print(f"【{completed}/{total}】")
            ## 也可以batch的打印
            # if (completed % self.max_concurrent == 0) or (completed == total):
            #     print(f"【{completed}/{total}】")

        return results


async def main():
    base_url = "your_base_url"
    api_key = "your_api_key"

    # 选择供应商和模型
    get_keywords = GetKeywords(
        base_url=base_url,
        api_key=api_key,
        model_name="qwen-plus",
        thinking=False,
        max_concurrent=16,  # 可根据API限制调整并发数
    )

    # 导入指令集，这里换成了我用大模型生成的
    data = pd.read_excel("./data/用户查询指令集.xlsx")
    questions = list(data["问题"])[:]

    # 并发处理所有问题
    print(f"开始处理 {len(questions)} 个问题...")
    results = await get_keywords.process_batch(questions)
    print("所有问题处理完成")

    # 汇总词表，多次出现只保留最后一个。词典只需要术语和定义，词表还要带有来源给业务看原文。
    term_words, illdefined_words = {}, {}
    terms_dict, illdefined_dict = {}, {}
    for result in results:
        if not result["success"]:
            continue

        question = result["question"]
        response = result["response"]
        try:
            response = json.loads(response)
            if "专业术语" in response:
                for item in response["专业术语"]:
                    for term, definition in item.items():
                        terms_dict.update({term: definition})
                        term_words.update(
                            {term: {"question": question, "definition": definition}}
                        )
            if "澄清术语" in response:
                for item in response["澄清术语"]:
                    for term, definition in item.items():
                        illdefined_dict.update({term: definition})
                        illdefined_words.update(
                            {term: {"question": question, "definition": definition}}
                        )
        except json.JSONDecodeError:
            print(f"解析JSON失败: {question}, 响应: {response}")
        except Exception as e:
            print(f"处理结果出错: {question}, 错误: {str(e)}")

    # 创建输出路径
    if not os.path.exists("./output"):
        os.makedirs("./output")

    # 保存字典用于后续做术语定义检索
    with open("./output/terms_dict.json", "w", encoding="utf-8") as f:
        json.dump(terms_dict, f, ensure_ascii=False, indent=4)
    with open("./output/illdefined_dict.json", "w", encoding="utf-8") as f:
        json.dump(illdefined_dict, f, ensure_ascii=False, indent=4)

    # 保存excel给业务去看，自己玩玩的话就用不上了。
    term_words_df = {"术语": [], "来源": [], "定义": []}
    for term, info in term_words.items():
        term_words_df["术语"].append(term)
        term_words_df["来源"].append(info["question"])
        term_words_df["定义"].append(info["definition"])
    term_words_df = pd.DataFrame(term_words_df)
    term_words_df.to_excel("./output/专业术语_v0.xlsx", index=False)

    illdefined_words_df = {"术语": [], "来源": [], "定义": []}
    for term, info in illdefined_words.items():
        illdefined_words_df["术语"].append(term)
        illdefined_words_df["来源"].append(info["question"])
        illdefined_words_df["定义"].append(info["definition"])
    illdefined_words_df = pd.DataFrame(illdefined_words_df)
    illdefined_words_df.to_excel("./output/澄清术语_v0.xlsx", index=False)


if __name__ == "__main__":
    # 别忘了修改大模型的url和api_key
    asyncio.run(main())
