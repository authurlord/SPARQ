import sys
import os
import pandas as pd
from unittest.mock import MagicMock

# Mock vllm and FlagEmbedding to avoid import errors
sys.modules["vllm"] = MagicMock()
sys.modules["FlagEmbedding"] = MagicMock()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the functions directly
from schedule_pipeline.run_full_pipeline_niat import build_niat_prompt_from_df, table_to_str_niat

def test_prompt_gen():
    # Mock data
    mock_data = [{
        'table_id': 'Test_Table_1',
        'table_title': 'Test Table with Spaces',
        'question': 'How many rows are in this table?'
    }]
    
    # Create with row_id already present to test valid handling
    mock_df = pd.DataFrame({
        'row_id': [0, 1],
        'col1': ['A', 'B'],
        'col2': [10, 20]
    })
    
    # Generate prompt
    print("\n--- Generating Prompt ---")
    prompt = build_niat_prompt_from_df(mock_data, mock_df, 0)
    
    print("\n[Prompt Tail (Input Section)]:")
    print(prompt[-600:]) 
    
    # Verify key components
    assert "Test_Table_with_Spaces" in prompt, "Table title not underscored in schema"
    assert "row_id" in prompt, "row_id not found"
    assert "col1" in prompt
    assert "How many rows" in prompt
    assert "<input>" in prompt
    assert "<output>" in prompt
    
    # Check formatting of the table string in prompt
    # Should look like:
    # row_id  col1  col2
    # 0       A     10
    # 1       B     20
    assert "row_id" in prompt and "col1" in prompt and "col2" in prompt
    assert "0" in prompt and "A" in prompt
    
    print("\n--- Verification Passed ---")

if __name__ == "__main__":
    test_prompt_gen()
