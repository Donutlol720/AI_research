import json

def analyze_results(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    problem_keys = []
    results = []
    error_flag_count = 0

    for problem_key, problem_data in data.items():
        problem_keys.append(problem_key)
        results.append(problem_data['result'])
        if problem_data['flag_error'] == 1:
            error_flag_count += 1

    return problem_keys, results, error_flag_count

if __name__ == "__main__":
    file_path = r'C:\Users\minhe\python\AI_research\Results\AMC_2022_12A_llama-4-maverick_benchmark_Results.json'
    problem_keys, results, error_flag_count = analyze_results(file_path)

    print("Problem Keys and Results:")
    for key, result in zip(problem_keys, results):
        print(f"{key}: {result}")

    print(f"\nFrequency of Error Flag: {error_flag_count}")