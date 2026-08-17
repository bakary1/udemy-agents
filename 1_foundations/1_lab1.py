import os
from dotenv import load_dotenv
from openai import OpenAI
from IPython.display import Markdown, display

# Load and verify OpenAI API key from environment
openai_api_key = os.getenv("OPENAI_API_KEY")

if openai_api_key:
    print(f"OpenAI API key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI key not set - please  head to troubleshooting guide in folder")


# Initialize OpenAI client (uses OPENAI_API_KEY from environment)
openai = OpenAI()

# Example 1: Simple chat completion - get a fun fact
messages = [{"role": "user", "content": "Tell me a fun fact"}]
response = openai.chat.completions.create(model="gpt-5.4-mini", messages=messages)
print(response.choices[0].message.content)

# Example 2: Generate an IQ assessment question
question_prompt = "Please propose a hard, challenging question to assess someone's IQ. Respond only with the question."
messages = [{"role": "user", "content": question_prompt}]

response = openai.chat.completions.create(model="gpt-5.4-mini", messages=messages)
question = response.choices[0].message.content
print(question)

# Display the question in formatted markdown
display(Markdown(question))

# Send prompt to API
messages = [{"role": "user", "content": question}]
response = openai.chat.completions.create(model="gpt-5.4-mini", messages=messages)
answer = response.choices[0].message.content

# Example 3: Have the model evaluate an answer to the generated question
evaluation_prompt = f"""
Here is a question
{question}

And here is a possible answer that might be correct or incorrect:
{answer}

Please evaluate if the answer is correct or incorrect.
"""

messages = [{"role": "user", "content": evaluation_prompt}]
response = openai.chat.completions.create(model="gpt-5.4-mini", messages=messages)
print(response.choices[0].message.content)
