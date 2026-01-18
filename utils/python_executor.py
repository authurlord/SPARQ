import sys
import io
import contextlib
import pandas as pd
import numpy as np
import scipy
import traceback
import os

def execute_python_code(code: str, df: pd.DataFrame, timeout: int = 5) -> str:
    """
    Executes the provided Python code with the given DataFrame loaded as 'df' (or available via read_csv).
    Captures stdout and returns it.
    
    Args:
        code (str): The Python code to execute.
        df (pd.DataFrame): The DataFrame to process.
        timeout (int): Timeout in seconds (not strictly enforced in this simple exec, 
                       but kept for API compatibility if we upgrade to multiprocessing).
        
    Returns:
        str: Captures stdout output or error message.
    """
    
    # Clean code: remove markdown code blocks if present
    code = code.strip()
    if code.startswith("```python"):
        code = code[9:]
    elif code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
    code = code.strip()
    
    # Identify if code tries to read 'table.csv'
    # We will save the DF to table.csv in the current working directory (careful with concurrency!)
    # Better: create a partial read_csv that returns our df, or actually write the file.
    # Given the constraints of simple 'exec', writing a temp file is risky with concurrency.
    # However, for this task, let's try to mock pd.read_csv or just provide `df` in globals.
    # But the prompt explicitly says "Ensure to load the table with command `df = pd.read_csv('table.csv')`".
    # So we should probably handle that.
    
    # Safe approach: usage of a unique filename or mocking.
    # Let's use a mock function for pd.read_csv that returns the df if 'table.csv' is requested.
    
    output_capture = io.StringIO()
    
    def mocked_read_csv(filepath, *args, **kwargs):
        if filepath == 'table.csv':
            return df.copy()
        return pd.read_csv(filepath, *args, **kwargs)
    
    # Prepare execution environment
    exec_globals = {
        'pd': pd,
        'np': np,
        'scipy': scipy,
        'print': print, # Will be redirected
    }
    
    # Inject our mocked read_csv into the pandas object in the exec scope? 
    # That's hard because `import pandas as pd` inside the code will reload/use the real module.
    # A better way is to actually write the file to a temp path and change CWD, or just expect execution in a thread-safe way.
    # But for "minimal modification" and "local test", actual file writing is easiest if concurrency is low or handled.
    # "concurrency=32" is used in the pipeline. Writing 'table.csv' in current dir is BAD.
    
    # Strategy: The code usually looks like:
    # import pandas as pd
    # df = pd.read_csv('table.csv')
    # ...
    
    # We can prepend setup code that defines `pd` and `read_csv`.
    # Or, we can do a string replacement in the code: `pd.read_csv('table.csv')` -> `df_input.copy()`
    # and provide `df_input` in locals. This is robust enough for generated code.
    
    # Replace simple variations of read_csv('table.csv')
    code_mod = code.replace("pd.read_csv('table.csv')", "df_input.copy()")
    code_mod = code_mod.replace('pd.read_csv("table.csv")', 'df_input.copy()')
    
    exec_locals = {'df_input': df}
    
    # Capture stdout
    try:
        with contextlib.redirect_stdout(output_capture):
            exec(code_mod, exec_globals, exec_locals)
        return output_capture.getvalue()
    except Exception as e:
        # traceback.print_exc(file=output_capture)
        return f"Execution Error: {str(e)}"

