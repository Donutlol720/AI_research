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
#model_name = "meta-llama/llama-4-maverick"
#model_name = "meta-llama/llama-3.3-8b-instruct:free"
# model="openrouter/qwen/qwq-32b:free",

#model_name = "google/gemini-2.5-pro-preview"
#model_name = 'google/gemini-2.5-flash-preview-05-20'
#model_name = "openrouter/deepseek/deepseek-chat"
#model_name = "deepseek/deepseek-r1-0528:free"





def extract_final_answer(solution):
    
    
    match = re.search(r'The final answer is: \$\\boxed{(.+?)}\$', solution) or \
            re.search(r'The final answer is: \\boxed{(.+?)} \\', solution)
    
    if match:
        s = match.group(1).replace('\\', '').replace(" ", "") 
        return s
    return None

def solve_problem_with_gemini(problem_statement, client, model_name, prompt_template="{problem_statement} "):
    # Format the prompt with the problem
    prompt = prompt_template.format(problem_statement=problem_statement)
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

    for i in range(15,16):  # Adjust range as needed
        problem_key = f"Problem {i}"
        if problem_key in problems_data:
            problem = problems_data[problem_key]['problem_statement']
            result_data = problems_data[problem_key]
            result_data["model"] = model_name
            result_data["prompt"] = prompt_template
            result_data["solutions"] = []

            for _ in range(7):  # Repeat each problem 2 times
                solution = solve_problem_with_gemini(problem, client, model_name, prompt_template)
                result_data["solutions"].append(solution)

            results[problem_key] = result_data

    # Check solutions and update the JSON data with the correct count
    check_solutions(results)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

# if __name__ == "__main__":

#     ##change here for different output file name
#     model_='llama-4-maverick'
#     model_round='benchmark'  ## different prompt; keep all prompts as record for future use
#     year = '2022_12B'

#     #######################

#     input_path = f'./Results/AMC_{year}_AP_Input.json'
#     output_path = f'./Results/AMC_{year}_{model_}_{model_round}_Results.json'
#     model_name = "meta-llama/llama-4-maverick"
#     #model_name = "deepseek/deepseek-chat"
#     prompt_template = "if the final answer is an improper fraction, the final step convert improper fraction to mixed fractions: {problem}"
    
#     main(input_path, output_path, model_name, prompt_template)



if __name__ == "__main__":

    filelist = [ '2024_12A']
    # filelist = ['2022_12A','2022_12B','2023_12A','2023_12B', '2024_12A', '2024_12B']

    ##change here for different output file name
    # model='phi-4'
    model='gemma-3'
    # model = "llama-4-maverick"
    model_round='COT1'  ## different prompt; keep all prompts as record for future use
    
    #######################

    for file_prefix in filelist:
        print(file_prefix)
        input_path = f'./Results/AMC_{file_prefix}_AP_Input.json'
        output_path = f'./Results/Choice_Removed_CoT_AMC_{file_prefix}_{model}_{model_round}_Results.json'
        # model_name = "meta-llama/llama-4-maverick"
        # model_name = "microsoft/phi-4"
        model_name = "google/gemma-3-27b-it"
        prompt_template = r"You will be provided a math question. Solve the problem carefully and rigorously. Proceed step by step. For each step: explicitly state what you are doing. Make sure to state formulas and casework clearly. Here is an example:\
            Example problem: How many positive integers $n$ satisfy \[\dfrac{{n+1000}}{{70}}\] = floor(\sqrt(n))? (Recall that floor(x) is the greatest integer not exceeding $x$.)\
            Example solution outline:\
                Step 1: We are given that \[\dfrac{{n+1000}}{{70}}] = floor(\sqrt(n)). floor(\sqrt(n)) must be an integer, which means that $n+1000$ is divisible by $70$. As $1000\equiv 20\pmod{{70}}$, this means that $n\equiv 50\pmod{{70}}$, so we can write $n=70k+50$ for some integer value k.\
                    Step 2: Therefore, substituting $n=70k+50$ into the original equation, we get \[\dfrac{{n+1000}}{{70}}\]=\[\dfrac{{70k+1050}}{{70}}\]=k+15=floor(\sqrt(n)).\
                        Step 3: Notice that the right hand side is a floor function, which implies that the value obtained after applying the floor function is at most 1 less than the value obtained before applying the floor function.\
                            Step 4: Then, $k+15$ must be within 1 of $\sqrt{{70k+50}}$. This gives us the inequalities $\sqrt{{70k+50}}-1 < k+15$ and $k+15\leq\sqrt{{70k+50}}$\
                                Step 5: Squaring the second inequality, $k+15\leq\sqrt{{70k+50}}$, we get $k^2+30k+225 \leq 70k+50$. We are allowed to do this since both sides of the inequality are nonnegative. Moving everything to the left side of the inequality, we get $k^2-40k+175 \leq 0$. This is can be factored using the quadratic formula, and the inequality becomes $(k-5)(k-35) \leq 0$. This implies that $5\leq k$ and $k \leq 35$.\
                                    Step 6: For the first inequality, $\sqrt{{70k+50}}-1 < k+15$, we first move $-1$ to the right hand side, to get $\sqrt{{70k+50}}< k+16$. Squaring both sides, we get $70k+50 < k^2+32k+256$. Moving everything to the right side of the inequality, we get $ 0 < k^2-38k+206$.\
                                         Step 7: Using the quadratic formula to solve the inequality, we get $k<19-\sqrt{{155}}$ or $k>19+\sqrt{{155}}$.\
                                            Step 8: Notice that now we have bounds on the value of $k$. The first bound is $5 \leq k \leq 35$, and the second bound is $k<19-\sqrt{{155}}$ or $k>19+\sqrt{{155}}$.\
                                                Step 9: Since $k$ must be an integer, we only need to find integers that satisfy the two inequalities. \
                                                    tep 10: Testing integer values of $k$, we get $k=5,6,32,33,34,35$ as valid values of $k$.\
                                                        Step 11: Recall that $n=70k+50$. Substituting each of the valid values of $k$ into the equation, we test to make sure these values of $k$ produce valid values of $n$ as well.\
                                                            Step 12: Since the original question asked for the number of solutions to the equation, and these six values of $k$ produce valid values of $n$, our answer must be $6$.\
                                                                 Step 13: Therefore, the final answer is $6$.\
                                                                    ------------------\
                                                                        Now solve the following problem. Remember to work carefully and rigorously. Let's think step-by-step: {problem_statement}\
            "
        main(input_path, output_path, model_name, prompt_template)