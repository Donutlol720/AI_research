import json
import re
from openai import OpenAI

from dotenv import load_dotenv
import os


# print(os.environ.get("TOGETHER_API_KEY"))  
# print(os.getenv("TOGETHER_API_KEY"))  

load_dotenv(".env") 


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


#model_name = "x-ai/grok-beta"
model_name = "meta-llama/llama-4-maverick"
#model_name = "meta-llama/llama-3.3-8b-instruct:free"
# model="openrouter/qwen/qwq-32b:free",

#model_name = "google/gemini-2.5-pro-preview"
#model_name = 'google/gemini-2.5-flash-preview-05-20'
#model_name = "openrouter/deepseek/deepseek-chat"
#model_name = "deepseek/deepseek-r1-0528:free"



def extract_final_answer(solution):
    
    match = re.search(r'The final answer is: \$\\boxed{(.+?)}\$', solution)
    if match:
        s = match.group(1).replace('\\', '').replace(" ", "") 
        return s
    return None

def solve_problem_with_gemini(problem, client, model_name, prompt_template="{problem} "):
    # Format the prompt with the problem
    prompt = prompt_template.format(problem=problem)
    #print(prompt)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

def check_solutions(problems_data):
    
    for problem_key, problem_data in problems_data.items():
        correct_answer = problem_data['answer_value']
        solutions = problem_data['solutions']
        
        #correct_count = sum(1 for solution in solutions if correct_answer in solution)
        #print(correct_answer)
        
        llm_answer = [extract_final_answer(solution) for solution in solutions]
        correct_count = sum(1 for solution in solutions if extract_final_answer(solution) == correct_answer)
        
        total_solutions = len(solutions)

        problem_data['LLM_answer'] = llm_answer
        problem_data['result'] = f"{correct_count}/{total_solutions}"
              
        

        

def main(input_path, output_path, model_name, prompt_template):
    with open(input_path, 'r', encoding='utf-8') as f:
        problems_data = json.load(f)
    
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY
    )
    results = {}

    for i in range(1,26):  # Adjust range as needed
        problem_key = f"Problem {i}"
        if problem_key in problems_data:
            problem = problems_data[problem_key]['problem_statement']
            result_data = problems_data[problem_key]
            result_data["model"] = model_name
            result_data["prompt"] = prompt_template
            result_data["solutions"] = []

            for _ in range(5):  # Repeat each problem 2 times
                solution = solve_problem_with_gemini(problem, client, model_name, prompt_template)
                result_data["solutions"].append(solution)

            results[problem_key] = result_data

    # Check solutions and update the JSON data with the correct count
    check_solutions(results)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":

    ##change here for different output file name
    model_='llama-4-maverick'
    model_round='benchmark'  ## different prompt; keep all prompts as record for future use
    year = '2022_12B'

    #######################

    input_path = f'./Results/AMC_{year}_AP_Input.json'
    output_path = f'c:/Users/minhe/python/AI_Research/Results/AMC_{year}_{model_}_{model_round}_Results.json'
    model_name = "meta-llama/llama-4-maverick"
    #model_name = "deepseek/deepseek-chat"
    prompt_template = "if the final answer is an improper fraction, the final step convert improper fraction to mixed fractions: {problem}"
    
    main(input_path, output_path, model_name, prompt_template)



# if __name__ == "__main__":
#     filelist = [ '2022_12B', '2023_12A','2023_12B', '2024_12A', '2024_12B']

#     ##change here for different output file name
#     model_='llama-4-maverick'
#     model_round='benchmark'  ## different prompt; keep all prompts as record for future use
    
#     #######################

#     for file_prefix in filelist:
#         input_path = f'./Results/AMC_{file_prefix}_AP_Input.json'
#         output_path = f'c:/Users/minhe/python/AI_Research/Results/AMC_{file_prefix}_{model_}_{model_round}_Results.json'
#         model_name = "meta-llama/llama-4-maverick"
#         prompt_template = "if the final answer is an improper fraction, the final step convert improper fraction to mixed fractions: {problem}"
        
#         main(input_path, output_path, model_name, prompt_template)