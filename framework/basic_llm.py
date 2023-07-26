import os
import json
import csv
import re
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

# OpenAI API key
openai_api_key_file = open("../openai_api_key.txt", "r")
os.environ["OPENAI_API_KEY"] = openai_api_key_file.read().strip()
openai_api_key_file.close()

# Load the data
file = open("../data/mmlu_test/high_school_government_and_politics_test.csv", "r")
csv_reader = csv.reader(file, delimiter=',')
data = []
for row in csv_reader:
    data.append({"question_text": row[0], "choices": row[1:-1], "correct_answer": row[-1]})

# Create the language model
llm = OpenAI(temperature=0)

num_correct = 0

# Generate a response for each question
for i, item in enumerate(data):
    question = item['question_text']
    print("Q:", question)

    # Convert the list of choices to a string
    choices_text = "\n".join([str(i+1) + ". " + choice for i, choice in enumerate(item['choices'])])
    print("Options:")
    print(choices_text)

    # Create the prompt template which structures the llm input
    prompt = PromptTemplate(
        input_variables=["question", "choices"],
        template="Output the numbered option for the following question: {question}\nOptions:\n{choices}"
    )

    # Generate the response
    response = llm(prompt.format(question=question, choices=choices_text))
    print("Response:", response.strip())

    # Convert the response to the numbered choice
    numbers = [int(num) for num in re.findall(r'\d+', response.strip().split(".")[0])]
    if len(numbers) == 0:
        numbered_output = 1
    else:
        numbered_output = numbers[-1]
    letter_output = chr(ord('A') + int(numbered_output) - 1)

    print("Generated answer:", letter_output)

    # Evaluate the response
    is_correct = letter_output == item['correct_answer']
    print("Correct answer:", item['correct_answer'])
    print("Correct:", is_correct, "\n")

    # Update the number of correct answers
    if letter_output == item['correct_answer']:
        num_correct += 1

    item["llm_answer"] = letter_output
    item["llm_is_correct"] = is_correct

print("EM:", num_correct / len(data))

# Save the QA sets in a JSON file
with open("../output/qa_sets_government.json", "w") as f:
    json.dump(data, f, indent=4)
