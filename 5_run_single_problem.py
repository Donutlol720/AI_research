## some template
## loop all questions from the json
## pick up one problem and corresponding fields from Json
## pass the problem to the LLM model with customized prompt
import json

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


file_path = r'c:\Users\minhe\python\AI_research\Results\AMC_2022_12A_llama-4-maverick_benchmark_Results.json'  
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)     


print(data["Problem 1"]["Choices"])
print(data["Problem 1"]["problem_statement"])   
print(data["Problem 1"]["prompt"])   


client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY
    )

problem = data["Problem 1"]["problem_statement"]
prompt = data["Problem 1"]["prompt"] 

### update prompt
#data["Problem 1"]["prompt"] = "if the final answer is an improper fraction, the final step convert improper fraction to mixed fractions: " + data["Problem 1"]["problem_statement"]

    
response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
    )
    
result = response.choices[0].message.content    


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
#     input_path = r'c:\Users\minhe\python\AI_research\Results\AMC_2022_12A_llama-4-maverick_benchmark_Results.json'
#     output_path = r'c:\Users\minhe\python\AI_research\Results\AMC_2022_12A_llama-4-maverick_benchmark_Results_updated.json'
    
#     # Load JSON data
#     json_data = load_json(input_path)
    
#     # Define model and prompt template (pseudo-code)
#     model = None  # Replace with actual model initialization
#     prompt_template = "if the final answer is an improper fraction, the final step convert improper fraction to mixed fractions: {problem}"
    
#     # Process problems
#     process_problems(json_data, model, prompt_template)
    
#     # Save updated JSON data
#     save_json(json_data, output_path)