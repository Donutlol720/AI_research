import re
import json

def extract_problems(problems_text):
    with open(problems_text, 'r', encoding='utf-8') as f:
        content = f.read()
    
    #print(content[:100])
    problem_sections = re.split(r'Problem\s+(\d+)', content)[1:]
    problems = {}
    
    for i in range(0, len(problem_sections), 2):
        problem_number = problem_sections[i]
        content = problem_sections[i+1]
        flag_error = 0

        #choices_match = re.search(r'(\$\s*\\textbf{\(A\)[^$]*\$)', content)
        
        ## match $ until the lastest $ on
        choices_match = re.search(r'(\$\s*\\textbf{\(A\).*\$)', content, re.DOTALL)

        
        if choices_match:
            choices = choices_match.group(1).strip()
            problem_statement = content[:choices_match.start()].strip()
        else:
            choices = ""
            problem_statement = content.strip()
            flag_error = 1
        
        problem_key = f"Problem {problem_number}"
        problems[problem_key] = {
            "problem": content,
            "problem_statement": problem_statement,
            "Choices": choices,
            "flag_error": flag_error
        }
    
    return problems

def parse_choices(choices_str):
    s = choices_str.strip()
    
    if s.startswith('$'):
        s = s[1:]
    if s.endswith('$'):
        s = s[:-1]
    parts = s.split('\\textbf')
    choices_dict = {}
    for part in parts:
        part = part.strip()
        
        if not part:
            continue
        if part.startswith('{') and part[1] == '(' and len(part) >= 3:
            letter = part[2]
            
            if letter in 'ABCDE':
                end_brace = part.find('}', 3)
                
                if end_brace != -1:
                                        
                    value_str = part[end_brace+1:]
                    choices_dict[letter] = value_str
                    
    return choices_dict

def clean_value(s):
    s = s.replace('~', '')
    s = s.replace('\\', '')
    s = s.replace('qquad', '')
    s = s.replace(' ', '')
    return s.strip()

def main(problems_path, answers_path, output_path):
    problems = extract_problems(problems_path)
   
    with open(answers_path, 'r') as f:
        answers = f.read().splitlines()
    
    for i in range(1,26):
        key = f"Problem {i}"
        if key not in problems:
            continue
        problem = problems[key]
        choices_str = problem["Choices"]
        choices_dict = parse_choices(choices_str)
        
        answer_letter = answers[i-1].strip()
        if answer_letter in choices_dict:
            raw_value = choices_dict[answer_letter]
            cleaned_value = clean_value(raw_value)
            problem["answer_choice"] = answer_letter
            problem["answer_value"] = cleaned_value
    
        problem['Raw_file'] =[problems_path,answers_path]

    with open(output_path, 'w') as f:
        json.dump(problems, f, indent=2)

# if __name__ == "__main__":
    
#     answers_path = r'.\Raw_files\AMC_2022_12B_Answer.sty'
#     problems_path = r'.\Raw_files\AMC_2022_12B_Problem.sty'
#     output_path = r'.\Results\AMC_2022_12B_AP_Input.json'
    
#     main(problems_path, answers_path, output_path)

# if we need to run all the files at once
if __name__ == "__main__":
    filelist = ['2022_12A','2022_12B','2023_12A', '2023_12B', '2024_12A', '2024_12B']
    
    for file_prefix in filelist:
        problems_path = rf'.\Raw_files\AMC_{file_prefix}_Problem.sty'
        answers_path = rf'.\Raw_files\AMC_{file_prefix}_Answer.sty'
        output_path = rf'.\Results\AMC_{file_prefix}_AP_Input.json'
        
        main(problems_path, answers_path, output_path)