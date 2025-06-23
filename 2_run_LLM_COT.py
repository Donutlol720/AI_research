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

    #filelist = [ '2024_12A', '2024_12B']
    filelist = ['2022_12A','2022_12B','2023_12A','2023_12B', '2024_12A', '2024_12B']

    ##change here for different output file name
    #model_='phi-4'
    #model_='llama-4-maverick'
    model_='gemma-3'
    model_round='COT2'  ## different prompt; keep all prompts as record for future use
    
    #######################

    for file_prefix in filelist:
        print(file_prefix, model_round, model_ )
        input_path = f'./Results/AMC_{file_prefix}_AP_Input.json'
        output_path = f'./Results/COT2/AMC_{file_prefix}_{model_}_{model_round}_Results.json'
        #model_name = "meta-llama/llama-4-maverick"
        model_name = "google/gemma-3-27b-it"
        #model_name = "microsoft/phi-4"

        prompt_template = """
                            **Role**: You are an expert mathematician specializing in AMC problem-solving. Solve all problems with extreme precision using this framework:

                            ### **Problem-Solving Framework**
                            1. **DECOMPOSE** (3-5 bullet points):
                            - Identify core variables/quantities and their relationships
                            - List all explicit constraints and implicit assumptions
                            - Flag any units/conversion requirements
                            - Define success criteria: "The goal is to find..."

                            2. **STRATEGY SELECTION** (Justify choice):
                            - Consider: Algebra, Combinatorics, Geometry, Number Theory, or Hybrid
                            - Evaluate efficiency: "This method is optimal because..."
                            - For alternator/liar problems: Explicitly model truth tables
                            For inequalities: Define boundary analysis approach
                            For combinatorics: Specify counting principle

                            3. **RIGOROUS EXECUTION** (Never skip steps):
                            - Derive formulas from first principles: 
                                "Starting from [core principle], we derive..."
                            - Show all calculations: 
                                "Calculate: [step1] → [step2] → [result]"
                            - Handle casework: 
                                "Case 1: [condition] → [logic] → [result]"
                            - Track units/dimensions through computations

                            4. **VALIDATION** (Triple-check):
                            - Dimensional analysis: "Units consistency: [unit1] → [unit2] ✓"
                            - Edge case testing: "At boundary x=[value], we verify..."
                            - Answer reversal: "If solution is X, plugging back yields [check]"
                            - Option comparison (if applicable): "Choice A matches because..."
                            - Reality check: "Does this quantity make sense? [Reason]"

                            ### **Example Template** (Adapt to specific problem)
                            **Problem**: "Carlos drove 70 min at 40 mph. Distance traveled?"
                            **Decompose**: 
                            - Variables: time=70 min, speed=40 mph
                            - Constraints: unit mismatch (min vs hours)
                            - Goal: distance in miles
                            **Strategy**: 
                            - Use distance = speed × time with unit conversion
                            - Optimal: Convert minutes to hours first
                            **Work**:
                            - Convert: 70 min = 70/60 hr = 7/6 hr
                            - Calculate: distance = 40 mph × 7/6 hr = 280/6 ≈ 46.67 miles
                            **Validation**:
                            - Units: mph × hr = miles ✓
                            - Reverse: 46.67 miles / 40 mph = 1.167 hr = 70 min ✓
                            - Realistic: 40mph in 70min ≈ 46-47 miles ✓

                            ### **Critical Rules**
                            - For liar/truth-teller problems: 
                            "Build response tables showing all answer combinations"
                            - For geometry: 
                            "Include coordinate system setup and diagram constraints"
                            - For inequalities: 
                            "Test critical points and sign changes"
                            - For combinatorics: 
                            "Justify overcounting adjustments"

                            **Problem to Solve**:
                            {problem}
                            """


        
        main(input_path, output_path, model_name, prompt_template)


        # prompt_template =    r"You will be provided a math question. Solve the problem carefully and rigorously. Proceed step by step. For each step: explicitly state what you are doing. Make sure to state formulas and casework clearly. Here is an example:\
        #                             Example problem: How many positive integers $n$ satisfy \[\dfrac{{n+1000}}{{70}}\] = floor(n)? (Recall that floor(x) is the greatest integer not exceeding $x$.)\
        #                             Example solution outline:\
        #                             Step 1: We are given that \[\dfrac{{n+1000}}{{70}}] = floor(n). floor(n) must be an integer, which means that $n+1000$ is divisible by $70$. As $1000\equiv 20\pmod{{70}}$, this means that $n\equiv 50\pmod{{70}}$, so we can write $n=70k+50$ for some integer value k.\
        #                             Step 2: Therefore, substituting $n=70k+50$ into the original equation, we get \[\dfrac{{n+1000}}{{70}}\]=\[\dfrac{{70k+1050}}{{70}}\]=k+15=floor(n).\
        #                             Step 3: Notice that the right hand side is a floor function, which implies that the value obtained after applying the floor function is at most 1 less than the value obtained before applying the floor function.\
        #                             Step 4: Then, $k+15$ must be within 1 of $\sqrt{{70k+50}}$. This gives us the inequalities $\sqrt{{70k+50}}-1 < k+15$ and $k+15\leq\sqrt{{70k+50}}$\
        #                             Step 5: Squaring the second inequality, $k+15\leq\sqrt{{70k+50}}$, we get $k^2+30k+225 \leq 70k+50$. Moving everything to the left side of the inequality, we get $k^2-40k+175 \leq 0$. This is can be factored using the quadratic formula, and the inequality becomes $(k-5)(k-35) \leq 0$. This implies that $5\leq k$ and $k \leq 35$.\
        #                             Step 6: For the first inequality, $\sqrt{{70k+50}}-1 < k+15$, we first move $-1$ to the right hand side, to get $\sqrt{{70k+50}}< k+16$. Squaring both sides, we get $70k+50 < k^2+32k+256$. Moving everything to the right side of the inequality, we get $ 0 < k^2-38k+206$.\
        #                             Step 7: Using the quadratic formula to solve the inequality, we get $k<19-\sqrt{{155}}$ or $k>19+\sqrt{{155}}$.\
        #                             Step 8: Notice that now we have bounds on the value of $k$. The first bound is $5 \leq k \leq 35$, and the second bound is $k<19-\sqrt{{155}}$ or $k>19+\sqrt{{155}}$.\
        #                             Step 9: Since $k$ must be an integer, we only need to find integers that satisfy the two inequalities. \
        #                             tep 10: Testing integer values of $k$, we get $k=5,6,32,33,34,35$ as valid values of $k$.\
        #                             Step 11: Recall that $n=70k+50$. Substituting each of the valid values of $k$ into the equation, we test to make sure these values of $k$ produce valid values of $n$ as well.\
        #                             Step 12: Since the original question asked for the number of solutions to the equation, and these six values of $k$ produce valid values of $n$, our answer must be $6$.\
        #                             Step 13: Therefore, the final answer is $6$.\
        #                             ------------------\
        #                             Now solve the following problem. it is a multi-choice problem, Compare your answer with multiple-choice options (A)-(E). your answer should be one of the choices. Remember to work carefully and rigorously. Let's think step-by-step: {problem}\
        #                                 "
