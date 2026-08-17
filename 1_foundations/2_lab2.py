import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from IPython.display import Markdown, display


load_dotenv(override=True)
ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1/"

# Load and verify API keys from environment
api_keys = {
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
}

for key_name, key_value in api_keys.items():
    if key_value:
        print(f"✓ {key_name}: valid and begins with {key_value[:8]}")
    else:
        print(f"✗ {key_name}: invalid. Please check configurations")

# Create prompt text
request = """
Please compe up with a challenging, nuanced question with a succint answer,
that I can ask a number of LLMs to evaluate their intelligence.
Not a mathematical puzzle, but more of a tought-proviking question that requires intelligent insight.
Include in your question that the answer must be short. Answer only with the question, no explanation.
"""

# Structure prompt
messages = [{"role": "user", "content": request}]

# Run API call
openai = OpenAI()
response = openai.chat.completions.create(model="gpt-5.4-mini", messages=messages)
question = response.choices[0].message.content

# Run API call to Anthropic
anthropic = OpenAI(api_key=api_keys["ANTHROPIC_API_KEY"], base_url=ANTHROPIC_BASE_URL)


# Create function for model results
competitors = []
answers = []
messages = [{"role": "user", "content": question}]


def record(model_name, answer):
    competitors.append(model_name)
    answers.append(answer)
    display(Markdown(answer))


# GPT 5.4 answer
model_name = "gpt-5.4"
response = openai.chat.completions.create(model=model_name, messages=messages)
answer = response.choices[0].message.content

record(model_name=model_name, answer=answer)


# Anthropic claude sonnet-5
model_name = "claude-sonnet-5"
response = anthropic.chat.completions.create(model=model_name, messages=messages)
answer = response.choices[0].message.content

record(model_name=model_name, answer=answer)

# Make answers anonymous so that we can use LLM as judge
anonymous_answers = ""
for index, answer in enumerate(answers, start=1):
    anonymous_answers += f"# Response from competitor {index}\n{answer}\n\n"


judge = f"""
You are judging a competition between {len(competitors)} competitors. 
Each model has been given this question:

{question}

Here are the responses from each competitor:

{anonymous_answers}

Now respond with the JSON with the ranked order of the competitors, nothing else. Do not include markdown formatting or code blocks.
"""

display(Markdown(judge))

# Call LLM as a judge
judge_messages = [{"role": "user", "content": judge}]

model_name = "gpt-5.4"
response = openai.chat.completions.create(model=model_name, messages=judge_messages)
results = response.choices[0].message.content
print(results)

results_dict = json.loads(results)
ranks = results_dict["ranked_order"]
for index, result in enumerate(ranks, start=1):
    competitor = competitors[int(result) - 1]
    print(f"Rank {index}: {competitor}")
