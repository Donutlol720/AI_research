import json
import matplotlib.pyplot as plt

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

def visualize_results(problem_keys, results):
    # Convert results to numerical values for visualization
    correct_counts = [int(result.split('/')[0]) for result in results]
    total_counts = [int(result.split('/')[1]) for result in results]

    # Create a bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(problem_keys, correct_counts, label='Correct Answers')
    plt.bar(problem_keys, total_counts, alpha=0.5, label='Total Attempts')

    plt.xlabel('Problem Keys')
    plt.ylabel('Count')
    plt.title('Results Visualization')
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    file_path = r'C:\Users\minhe\python\AI_research\Results\AMC_2022_12A_llama-4-maverick_benchmark_Results.json'
    problem_keys, results, error_flag_count = analyze_results(file_path)

    print("Problem Keys and Results:")
    for key, result in zip(problem_keys, results):
        print(f"{key}: {result}")

    print(f"\nFrequency of Error Flag: {error_flag_count}")

    # Visualize the results
    visualize_results(problem_keys, results)