## some template
## loop all questions from the json
## pick up one problem and corresponding fields from Json
## pass the problem to the LLM model with customized prompt
import json
import re
from openai import OpenAI

from dotenv import load_dotenv
import os
load_dotenv(".env") 
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


file_path = '/Users/ericmin/AI_research/Results/AMC_2022_12A_llama-4-maverick_COT1_Results.json'
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)     


# print(data["Problem 17"]["Choices"])
# print(data["Problem 17"]["problem_statement"])   
# print(data["Problem 17"]["prompt"])   


client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY
    )

problem = data["Problem 17"]["problem_statement"]
prompt = data["Problem 17"]["prompt"] 

### update prompt
data["Problem 9"]["prompt"] = "You are a mathematician. You will be given a multiple choice problem. Carefully solve the problem step-by-step, explaining your resasoning at each stage. Make sure to pay attention to corner cases and make good use of the problem statement. Make effective use of trigonometric identities." + data["Problem 17"]["problem"]

model_name = data["Problem 17"]["model"]

prompt_final = """You will be provided a math question. Solve the problem carefully and rigorously. Proceed step by step. For each step: explicitly state what you are doing. Make sure to state formulas and casework clearly. Here is an example:

Example problem: How many positive integers $n$ satisfy \[\dfrac{{n+1000}}{{70}}\] = floor(\sqrt(n))? (Recall that floor(x) is the greatest integer not exceeding $x$.)

Example solution outline:

Step 1: We are given that \[\dfrac{{n+1000}}{{70}}] = floor(\sqrt(n)). floor(\sqrt(n)) must be an integer, which means that $n+1000$ is divisible by $70$. As $1000\equiv 20\pmod{{70}}$, this means that $n\equiv 50\pmod{{70}}$, so we can write $n=70k+50$ for some integer value k.

Step 2: Therefore, substituting $n=70k+50$ into the original equation, we get \[\dfrac{{n+1000}}{{70}}\]=\[\dfrac{{70k+1050}}{{70}}\]=k+15=floor(\sqrt(n)).

Step 3: Notice that the right hand side is a floor function, which implies that the value obtained after applying the floor function is at most 1 less than the value obtained before applying the floor function.

Step 4: Then, $k+15$ must be within 1 of $\sqrt(n)$, or $\sqrt{{70k+50}}$. This gives us the inequalities $\sqrt{{70k+50}}-1 < k+15$ and $k+15\leq\sqrt{{70k+50}}$

Step 5: Squaring the second inequality, $k+15\leq\sqrt{{70k+50}}$, we get $k^2+30k+225 \leq 70k+50$. Moving everything to the left side of the inequality, we get $k^2-40k+175 \leq 0$. This is can be factored using the quadratic formula, and the inequality becomes $(k-5)(k-35) \leq 0$. This implies that $5\leq k$ and $k \leq 35$.

Step 6: For the first inequality, $\sqrt{{70k+50}}-1 < k+15$, we first move $-1$ to the right hand side, to get $\sqrt{{70k+50}}< k+16$. Squaring both sides, we get $70k+50 < k^2+32k+256$. Moving everything to the right side of the inequality, we get $ 0 < k^2-38k+206$.

Step 7: Using the quadratic formula to solve the inequality, we get $k<19-\sqrt{{155}}$ or $k>19+\sqrt{{155}}$.

Step 8: Notice that now we have bounds on the value of $k$. The first bound is $5 \leq k \leq 35$, and the second bound is $k<19-\sqrt{{155}}$ or $k>19+\sqrt{{155}}$.

Step 9: Since $k$ must be an integer, we only need to find integers that satisfy the two inequalities.

Step 10: Testing integer values of $k$, we get $k=5,6,32,33,34,35$ as valid values of $k$.

Step 11: Recall that $n=70k+50$. Substituting each of the valid values of $k$ into the equation, we test to make sure these values of $k$ produce valid values of $n$ as well.

Step 12: Since the original question asked for the number of solutions to the equation, and these six values of $k$ produce valid values of $n$, our answer must be $6$.

Step 13: Therefore, the final answer is $6$.

------------------

Now solve the following problem. Remember to work carefully and rigorously. Let's think step-by-step: """ + data["Problem 17"]["problem_statement"]
print(prompt_final)
response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt_final}],
    )
    
result = response.choices[0].message.content   
print(result) 


# def process_problems(json_data, model, prompt_template):
#     for problem_key, problem_data in json_data.items():
#         # Extract problem and fields
#         problem_statement = problem_data['problem_statement']
#         choices = problem_data['Choices']
        
#         # Customize prompt
#         prompt = prompt_template.format(problem=problem_statement)
        
#         # Pass to LLM model (pseudo-code, replace with actual model call)
#         llm_response = model.generate(prompt)
        
#         # Change a specific field (e.g., update 'LLM_answer')
#         problem_data['LLM_answer'] = llm_response

# def save_json(json_data, output_path):
#     with open(output_path, 'w', encoding='utf-8') as f:
#         json.dump(json_data, f, indent=2)

# if __name__ == "__main__":
#     input_path = r'.\Results\AMC_2022_12A_llama-4-maverick_benchmark_Results.json'
#     output_path = r'.\Results\AMC_2022_12A_llama-4-maverick_benchmark_Results_updated.json'
    
#     # Load JSON data
#     json_data = load_json(input_path)
    
#     # Define model and prompt template (pseudo-code)
#     model = None  # Replace with actual model initialization
#     prompt_template = "if the final answer is an improper fraction, the final step convert improper fraction to mixed fractions: {problem}"
    
#     # Process problems
#     process_problems(json_data, model, prompt_template)
    
#     # Save updated JSON data
#     save_json(json_data, output_path)