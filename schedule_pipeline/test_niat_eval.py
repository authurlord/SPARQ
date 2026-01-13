import sys
import os
import pandas as pd
from unittest.mock import MagicMock

# Mock dependencies to avoid import errors
sys.modules["vllm"] = MagicMock()
sys.modules["FlagEmbedding"] = MagicMock()
sys.modules["openai"] = MagicMock()
sys.modules["openai"].__spec__ = MagicMock() # Fix for transformers import check

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.evaluator import Evaluator
# We need to manually replicate the parsing logic or import it.
# Importing form prompt_generate might be heavy, but let's try.
# If it fails, we can verify Evaluator behavior directly and copy parsing logic.
try:
    from utils.prompt_generate import evaluate_predictions
    HAS_EVAL_FUNC = True
except ImportError as e:
    print(f"Import error: {e}")
    HAS_EVAL_FUNC = False

def test_niat_eval_failure_case():
    print("Testing NIAT evaluation with WikiTQ metrics...")
    
    # 1. Test Evaluator.evaluate directly (verifies utils/evaluator.py fix)
    evaluator = Evaluator()
    dataset_name = 'niat'
    
    # Failing case reported by user
    # Pred: "Final Answer: Therefore, the answer is: "IFK Göteborg"" (or similar structure)
    # Actually the string in dataframe 'predict' column might be a list string or raw string
    # run_full_pipeline_niat usually saves list of strings?
    # evaluate_predictions does `eval(output['predict'])[0]` implies it's a string representation of a list.
    
    raw_predict = "['2. Final Answer: Therefore, the answer is: \"IFK Göteborg\"']"
    gold_answer = "IFK Göteborg"
    question = "Which club does Rune 'Killing' Emanuelsson play for?"
    
    # Simulation of parsing logic in evaluate_predictions
    try:
        raw_output = eval(raw_predict)[0]
        if 'Final Answer: ' in raw_output:
            pred_answer_all = raw_output.split('Final Answer: ')[1] # -> 'Therefore, the answer is: "IFK Göteborg"'
        else:
            pred_answer_all = raw_output
            
        # Extract answer inside quotes
        pred_answer = pred_answer_all.split('"')[1:2] # -> ['IFK Göteborg']
        if not pred_answer:
            pred_answer = [pred_answer_all]
            
    except Exception as e:
        print(f"Parsing failed: {e}")
        pred_answer = None

    print(f"Parsed prediction: {pred_answer}")
    
    # Debug: Check normalized strings
    from utils.normalizer import str_normalize
    from utils.wtq.evaluator import to_value_list, check_denotation
    
    # Replicate eval_ex_match steps
    p_lower = [str(p).lower().strip() for p in pred_answer]
    g_lower = [str(gold_answer).lower().strip()]
    
    p_norm = [str_normalize(p) for p in p_lower]
    g_norm = [str_normalize(p) for p in g_lower]
    
    p_val = to_value_list(p_norm)
    g_val = to_value_list(g_norm)
    
    print(f"Processed Pred: {p_val}")
    print(f"Processed Gold: {g_val}")
    
    check_res = check_denotation(p_val, g_val)
    print(f"Check Denotation Result: {check_res}")
    
    # 2. Score using Evaluator
    score = evaluator.evaluate(
        pred_answer,
        gold_answer,
        dataset=dataset_name,
        question=question
    )
    
    print(f"Score: {score}")
    assert score == True or score == 1, "Evaluation failed on valid answer!"
    
    print("Direct Evaluator test passed!")

if __name__ == "__main__":
    test_niat_eval_failure_case()
