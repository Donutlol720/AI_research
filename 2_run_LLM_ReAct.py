import re

# Dummy tools (replace with real ones)
def call_wolfram(query):
    print(f"[Calling Wolfram with query: {query}]")
    return f"Wolfram result for: {query}"

def run_code(code):
    print(f"[Running code: {code}]")
    try:
        exec_globals = {}
        exec(code, exec_globals)
        return str(exec_globals.get("result", ""))
    except Exception as e:
        return f"Error: {e}"

# Dummy model (replace with real model call)
def chat_with_model(prompt):
    print(f"\n=== MODEL SEES ===\n{prompt[-600:]}\n=== END ===")
    return input("Model reply: ")

# Main loop
def react_loop(question, min_cycles=2):
    context = f"""
You are an AI that solves problems by THINKING, ACTING with tools, OBSERVING results, and repeating this process step-by-step until you can give a FINAL ANSWER.

You can call tools:
    - Wolfram[<query>] for knowledge and computation
    - Code[<python code>] for running Python code

You MUST follow this format:
    THINK: <your reasoning>
    ACT: <tool>[<input>]
    OBSERVE: <tool output>
    THINK: <your reasoning> ...
    (repeat until ready)
    FINAL ANSWER: <your answer>

**Always perform at least 2 THINK → ACT → OBSERVE cycles** before giving FINAL ANSWER.

Example 1:

Question: What is the integral of sin(x)?

THINK: I will use Wolfram to compute the integral.
ACT: Wolfram[integrate sin(x)]
OBSERVE: -cos(x) + C
THINK: The result looks correct.
ACT: Code[print("Check: d/dx of -cos(x) is sin(x)")]
OBSERVE: sin(x)
THINK: Confirmed.
FINAL ANSWER: -cos(x) + C

---

Question: {question}
THINK:
"""

    done = False
    act_observe_cycles = 0

    while not done:
        response = chat_with_model(context)
        context += response

        act_match = re.search(r'ACT:\s*(\w+)\[(.*?)\]', response, re.DOTALL)
        if act_match:
            tool_name = act_match.group(1).strip()
            tool_input = act_match.group(2).strip()

            if tool_name.lower() == "wolfram":
                result = call_wolfram(tool_input)
            elif tool_name.lower() == "code":
                result = run_code(tool_input)
            else:
                result = "Unknown tool"

            observe = f"\nOBSERVE: {result}\nTHINK:"
            context += observe
            act_observe_cycles += 1
            print(f"[OBSERVE: {result}]")

        elif "FINAL ANSWER:" in response:
            if act_observe_cycles < min_cycles:
                print(f"[Model tried to give FINAL ANSWER too early, forcing another THINK!]")
                context = context.rsplit("FINAL ANSWER:", 1)[0] + "\nTHINK:"
            else:
                done = True
                print("\n=== FINAL ANSWER ===")
                print(response.split("FINAL ANSWER:")[-1].strip())

# Example usage:
react_loop("What is the derivative of x^3?")
