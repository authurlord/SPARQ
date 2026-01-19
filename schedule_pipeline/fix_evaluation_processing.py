#!/usr/bin/env python3
"""
Unified fix for evaluation string processing in all TableBench pipeline files.
This script adds the same string processing logic as run_full_pipeline_tablebench.py
"""

import os
import re

# Files to fix
files_to_fix = [
    'run_sql_iterative_tablebench.py',
    'run_pipeline_tablebench_pot.py',
    'run_pipeline_tablebench_pot_enhanced.py',
    'run_pipeline_tablebench_pot_enhanced_v2.py'
]

# The correct string processing code (from run_full_pipeline_tablebench.py)
correct_processing = '''    # Extract predictions with proper string processing
    preds = []
    for qa in qa_responses:
        pred_str = qa[0] if isinstance(qa, list) else str(qa)
        # Extract "The answer is:" pattern
        match = re.search(r'(?:the answer is|therefore|answer):\\s*(.+)', pred_str, re.IGNORECASE)
        if match:
            pred_str = match.group(1).strip()
        pred_str = pred_str.strip().strip('"\\'')[:200]  # Limit length to 200 characters
        preds.append(pred_str)'''

def fix_file(filepath):
    """Fix evaluation string processing in a file."""
    if not os.path.exists(filepath):
        print(f"⚠️  File not found: {filepath}")
        return False
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already fixed
    if 're.search(r\'(?:the answer is|therefore|answer):' in content:
        print(f"✅ Already fixed: {filepath}")
        return True
    
    # Pattern to find the evaluation section
    # Look for: preds.append(str(pred))
    pattern = r'(\s+)(preds\.append\(str\(pred\)\))'
    
    if not re.search(pattern, content):
        print(f"⚠️  Pattern not found in: {filepath}")
        return False
    
    # Replace with correct processing
    replacement = r'''\1# Extract predictions with proper string processing
\1pred_str = str(pred)
\1# Extract "The answer is:" pattern
\1match = re.search(r'(?:the answer is|therefore|answer):\\s*(.+)', pred_str, re.IGNORECASE)
\1if match:
\1    pred_str = match.group(1).strip()
\1pred_str = pred_str.strip().strip('"\\'')[:200]  # Limit length to 200 characters
\1preds.append(pred_str)'''
    
    new_content = re.sub(pattern, replacement, content)
    
    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"✅ Fixed: {filepath}")
    return True

def main():
    print("="*80)
    print("Fixing evaluation string processing in TableBench pipeline files")
    print("="*80)
    print()
    
    base_dir = '/home/yanmy/SPARQ/schedule_pipeline'
    os.chdir(base_dir)
    
    fixed_count = 0
    for filename in files_to_fix:
        filepath = os.path.join(base_dir, filename)
        if fix_file(filepath):
            fixed_count += 1
        print()
    
    print("="*80)
    print(f"Fixed {fixed_count}/{len(files_to_fix)} files")
    print("="*80)

if __name__ == "__main__":
    main()

