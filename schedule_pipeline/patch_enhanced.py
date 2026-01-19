#!/usr/bin/env python3
import sys

# Read the file
with open('run_pipeline_tablebench_pot_enhanced.py', 'r') as f:
    lines = f.readlines()

# Find and replace extract_python_code function
new_extract_func = '''def extract_python_code(response: str) -> str:
    """Extracts code block from response with improved parsing."""
    # Try to find code in ```python blocks
    match = re.search(r'```python\\s*(.*?)\\s*```', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Try to find code in ``` blocks
    match = re.search(r'```\\s*(.*?)\\s*```', response, re.DOTALL)
    if match:
        code = match.group(1).strip()
        # Remove language identifier if present
        if code.startswith(('python\\n', 'py\\n')):
            code = '\\n'.join(code.split('\\n')[1:])
        return code.strip()
    
    # Check if response looks like direct answer (no code)
    if response.strip().startswith(('[Answer Format]', 'Final Answer:')):
        return None  # Signal that this is not code
    
    # Check if response contains Python-like code without code blocks
    if any(pattern in response for pattern in ['import ', 'def ', 'pd.', 'df[', 'print(']):
        lines = response.split('\\n')
        code_lines = []
        in_code = False
        for line in lines:
            if any(p in line for p in ['import ', 'df = ', 'pd.']):
                in_code = True
            if in_code:
                if any(p in line.lower() for p in ['final answer:', 'step ', '[answer']):
                    break
                code_lines.append(line)
        if code_lines:
            return '\\n'.join(code_lines).strip()
    
    # If nothing found, return None
    return None

'''

# Find the function and replace it
new_lines = []
i = 0
while i < len(lines):
    if lines[i].strip().startswith('def extract_python_code('):
        # Skip old function
        new_lines.append(new_extract_func)
        while i < len(lines) and not (lines[i].strip().startswith('def ') and i > 0):
            i += 1
            if i < len(lines) and lines[i].strip() == '':
                break
        continue
    new_lines.append(lines[i])
    i += 1

# Add datetime import if not present
final_lines = []
for line in new_lines:
    final_lines.append(line)
    if 'import time' in line and 'from datetime import datetime' not in ''.join(new_lines):
        final_lines.append('from datetime import datetime\n')

# Add timestamp and parameter logging to main()
final_content = ''.join(final_lines)

# Add timestamp to save path
old_main = '''def main():
    args = parse_args()
    os.makedirs(args.tmp_save_path, exist_ok=True)'''

new_main = '''def main():
    args = parse_args()
    
    # Add timestamp to save path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if '_enhanced' in args.tmp_save_path and not any(c.isdigit() for c in args.tmp_save_path.split('_')[-1]):
        args.tmp_save_path = f"{args.tmp_save_path}_{timestamp}"
    
    os.makedirs(args.tmp_save_path, exist_ok=True)
    
    # Print all key parameters
    print("="*80)
    print("TableBench PoT Pipeline - Enhanced Version")
    print("="*80)
    print(f"Timestamp: {timestamp}")
    print(f"Save Path: {args.tmp_save_path}")
    print(f"Dataset: {args.tablebench_jsonl_path}")
    print(f"First N: {args.first_n}")
    print(f"LLM Name: {args.llm_name}")
    print(f"API Base: {args.api_base}")
    print(f"Code Sample Num: {args.code_sample_num}")
    print(f"Temperature: {args.temperature}")
    print(f"Top P: {args.top_p}")
    print(f"Concurrency: {args.llm_concurrency}")
    print("="*80)
    print()'''

final_content = final_content.replace(old_main, new_main)

# Write back
with open('run_pipeline_tablebench_pot_enhanced.py', 'w') as f:
    f.write(final_content)

print("✅ Enhanced script updated successfully!")
