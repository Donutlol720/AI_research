from crewai import Agent, Task, Crew, Process
from crewai_tools import ScrapeWebsiteTool
from dotenv import load_dotenv
import os
import json
import requests
import re
import time
# from bs4 import BeautifulSoup
# from urllib.parse import urlparse, parse_qs
from crewai import LLM
from datetime import datetime

# Load environment variables

#google/gemini-2.5-flash

load_dotenv(".env") 
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") 
GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")

# Configure LLM
# llm = LLM(
#     #model = "openrouter/deepseek/deepseek-chat-v3-0324",
#     model = "openrouter/deepseek/deepseek-chat",
#     #model = "google/gemini-2.5-flash",
#     base_url="https://openrouter.ai/api/v1",
#     api_key=OPENROUTER_API_KEY
# )

llm = LLM(
    model="gemini/gemini-2.5-flash",
    temperature=0,
)


json_analyst = Agent(
    role="json file expert include read, write, analysis json file",
    goal="Extract answer from a json file",
    backstory="Specializes in searching for and extracting information from json files",
    #tools=[ScrapeWebsiteTool()],
    llm=llm,
    verbose=False
)

# Problem_year_agent = Agent(
#     role='USA Coding Olympiad expert and literature expert',
#     goal='From the url, retrieve only the the year,month and problem number of the contest.',
#     backstory='Expert in extracting problem year month and problem number from url',
#     tools=[ScrapeWebsiteTool()],
#     llm=llm,
#     verbose=False
#)

def extract_solution(problems):
    analysis_task = Task(
        description=(
            f"read the json file from {problems}, there are 25 problems from Problem 1 to Problem 25. "
            "look at the problems one by one, find the final answer for each problem from five solutions in solutions field. "
            "and then compare the answers with the value in the answer_value field for each problem to see how many of them are correct. "
            "if all 5 answers are correct, output correct_final_answer: 5/5, pay attention to laTex format answer and mixed fraction. "
            "Ensure the output is ONLY the JSON object, nothing else. Only output raw JSON with no extra text"
            "Example: {'Problem 1': {'answers_from_solution':['4','5'], 'correct_final_answer':'5/5'}}"
        ),
        expected_output="output a valid json string with problem key, answers from solutions and the correct final answer like 1/2. \
                            Only output raw JSON with no extra text\
                            Example: {'Problem 1': {'answers_from_solution':['4','5'], 'correct_final_answer':'5/5'}}" ,
        agent=json_analyst,
        output_file="ignore_test_.json"
    )


    # Create crew and assign tasks
    crew = Crew(
        agents=[json_analyst],
        tasks=[analysis_task],
        verbose=False
    )

    result = crew.kickoff()
    return result



def remove_json_block_wrapper(s):
    """
    Remove triple-quoted `''' json` and closing `'''` from a string.
    """
    if s.strip().startswith("```json"):
        s = s.strip()
        s = s[len("''' json"):].strip()
        if s.endswith("```"):
            s = s[:-3].strip()
    return s



def main(input_path):
    with open(input_path, 'r') as f:
        problems = json.load(f)

    test_result = extract_solution(problems)
    cleaned_str = remove_json_block_wrapper(test_result.raw)
    json_output = json.loads(cleaned_str)
    for key, value in json_output.items():
        problems[key]["result"] = value["correct_final_answer"]
        problems[key]["LLM_answer"] = value["answers_from_solution"]


    with open(input_path, 'w', encoding='utf-8') as f:
        json.dump(problems, f, indent=2)



if __name__ == "__main__":

    #filelist = [ '2023_12A','2023_12B']
    filelist = ['2022_12A','2022_12B','2023_12A','2023_12B', '2024_12A', '2024_12B']

    ##change here for different output file name
    #model_ = 'llama-4-maverick'
    #model_='gemma-3'
    model_='phi-4'
    model_round='benchmark'  ## different prompt; keep all prompts as record for future use
    
    #######################

    for file_prefix in filelist:
        
        input_path = f'./Results/AMC_{file_prefix}_{model_}_{model_round}_Results.json'
        print(file_prefix)
        main(input_path)


####################################################
### run single file

# filepath = r".\Results\AMC_2022_12A_phi-4_benchmark_Results.json"   
# with open(filepath, 'r') as f:
#     problems = json.load(f)    

# test_result = extract_solution(problems)
# cleaned_str = remove_json_block_wrapper(test_result.raw)
# json_output = json.loads(cleaned_str)
# print("\n--- Parsed JSON Output ---")
# print(json.dumps(json_output, indent=2))


    
# for key, value in json_output.items():
#     problems[key]["result"] = value["correct_final_answer"]
#     problems[key]["LLM_answer"] = value["answers_from_solution"]

# with open(filepaht, 'w', encoding='utf-8') as f:
#     json.dump(problems, f, indent=2)






