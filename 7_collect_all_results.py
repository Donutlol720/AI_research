import os
import json
import re
import pandas as pd

def parse_filename(filename):
    """
    Parse filename of format:
    AMC_{year}_{model}_{cot}_Results.json
    Returns (year, model, cot) or None if not matched.
    """
    pattern = r"AMC_(\d{4}_\d{2}[AB])_([a-zA-Z0-9\-]+)_([a-zA-Z0-9]+)_Results\.json"
    match = re.match(pattern, filename)
    if match:
        year, model, cot = match.groups()
        return year, model.lower(), cot.lower()
    return None

def collect_results(base_dirs, models, cots, years):
    """
    Collect results from JSON files in base_dirs.
    Returns nested dict: results[model][problem_id][year][cot] = result
    """
    results = {model: {} for model in models}

    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            continue
        for root, _, files in os.walk(base_dir):
            for file in files:
                parsed = parse_filename(file)
                if not parsed:
                    continue
                year, model, cot = parsed
                if model not in models or cot not in cots or year not in years:
                    continue
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except Exception as e:
                    print(f"Failed to load {filepath}: {e}")
                    continue

                for problem_id, problem_data in data.items():
                    result = problem_data.get('result', None)
                    if result is None:
                        continue
                    # Initialize nested dicts
                    results[model].setdefault(problem_id, {})
                    results[model][problem_id].setdefault(year, {})
                    results[model][problem_id][year][cot] = result

    return results

def create_tables(results, models, cots, years):
    """
    Create pandas DataFrames for each model.
    Rows: problem_id
    Columns: MultiIndex (year, cot)
    Values: result
    """
    tables = {}
    for model in models:
        model_data = results.get(model, {})
        # Collect all problem_ids
        problem_ids = sorted(model_data.keys())
        # Create MultiIndex columns: all combinations of years and cots
        columns = pd.MultiIndex.from_product([years, cots], names=['Year', 'CoT'])
        # Prepare data matrix
        data = []
        for pid in problem_ids:
            row = []
            for year in years:
                for cot in cots:
                    val = model_data.get(pid, {}).get(year, {}).get(cot, None)
                    row.append(val)
            data.append(row)
        df = pd.DataFrame(data, index=problem_ids, columns=columns)
        tables[model] = df
    return tables

def main():
    base_dirs = [
        r"C:\Users\minhe\python\AI_research\Results",
        r"C:\Users\minhe\python\AI_research\Results\COT1",
        r"C:\Users\minhe\python\AI_research\Results\COT2",
    ]

    models = ['gemma-3', 'phi-4', 'llama-4-maverick']
    cots = ['benchmark', 'cot1', 'cot2', 'cot3', 'cot4']
    years = [
        '2022_12A', '2022_12B',
        '2023_12A', '2023_12B',
        '2024_12A', '2024_12B'
    ]

    results = collect_results(base_dirs, models, cots, years)
    tables = create_tables(results, models, cots, years)

    # Create an Excel writer object
    excel_path = r".\results_summary.xlsx"
    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        for model, df in tables.items():
            print(f"\n=== Results Table for Model: {model} ===")
            print(df)
            # Write each DataFrame to a separate sheet
            df.to_excel(writer, sheet_name=model)

    print(f"Results have been written to {excel_path}")


    for model, df in tables.items():
        print(f"\n=== Results Table for Model: {model} ===")
        print(df)

if __name__ == "__main__":
    main()


