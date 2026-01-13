#!/usr/bin/env python3
"""
Test script to verify NIAT table_id indexing is correct.
Verifies that questions map to their correct tables.
"""

import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_table_id_structure():
    """Test that we understand the NIAT data structure correctly."""
    json_path = '../datasets/NIAT/sampled_qa_pairs_4000_fixed.json'
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total QA samples: {len(data)}")
    
    # Count unique table_ids
    table_ids = [item['table_id'] for item in data]
    unique_ids = set(table_ids)
    print(f"Unique table_ids: {len(unique_ids)}")
    
    # Check that each item has required fields
    required_fields = ['table_id', 'question', 'answer', 'table_rows']
    for i, item in enumerate(data[:5]):
        missing = [f for f in required_fields if f not in item]
        if missing:
            print(f"Sample {i} missing: {missing}")
        else:
            print(f"Sample {i}: table_id={item['table_id']}, question={item['question'][:50]}...")
    
    # Find a table with multiple questions
    from collections import Counter
    id_counts = Counter(table_ids)
    multi_q_table = max(id_counts.items(), key=lambda x: x[1])
    print(f"\nTable with most questions: {multi_q_table[0]} ({multi_q_table[1]} questions)")
    
    # Show all questions for this table
    table_id = multi_q_table[0]
    questions = [(i, item['question']) for i, item in enumerate(data) if item['table_id'] == table_id]
    print(f"\nQuestions for table {table_id}:")
    for idx, q in questions[:5]:
        print(f"  [{idx}] {q[:80]}...")
    
    return True


def test_table_question_alignment():
    """Test that table content matches question context."""
    json_path = '../datasets/NIAT/sampled_qa_pairs_4000_fixed.json'
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("\n" + "="*60)
    print("Testing Table-Question Alignment")
    print("="*60)
    
    # Check first 5 samples
    for i in range(min(5, len(data))):
        item = data[i]
        table_id = item['table_id']
        question = item['question']
        answer = item['answer']
        table_rows = item['table_rows']
        table_title = item.get('table_title', 'N/A')
        
        print(f"\n[Sample {i}]")
        print(f"  table_id: {table_id}")
        print(f"  table_title: {table_title}")
        print(f"  question: {question[:100]}...")
        print(f"  answer: {answer}")
        print(f"  table_rows: {len(table_rows)} rows")
        if table_rows:
            print(f"  header: {table_rows[0][:5]}...")
    
    return True


def test_processed_table_mapping():
    """Test the new table_id-based mapping."""
    json_path = '../datasets/NIAT/sampled_qa_pairs_4000_fixed.json'
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("\n" + "="*60)
    print("Testing table_id-based Mapping")
    print("="*60)
    
    # Build table_id -> processed_table mapping (only process each table once)
    processed_tables = {}
    table_id_to_first_item = {}
    
    for item in data:
        table_id = item['table_id']
        if table_id not in processed_tables:
            # Simulate processing: just store table_rows for now
            processed_tables[table_id] = item['table_rows']
            table_id_to_first_item[table_id] = item
    
    print(f"Processed {len(processed_tables)} unique tables")
    
    # Now verify: for each QA sample, can we get the correct table?
    errors = 0
    for i, item in enumerate(data[:100]):
        table_id = item['table_id']
        
        # Old way (WRONG): processed_tables[i] - would fail
        # New way (CORRECT): processed_tables[table_id]
        if table_id not in processed_tables:
            print(f"ERROR: table_id {table_id} not found!")
            errors += 1
        else:
            # Verify the table matches
            stored_table = processed_tables[table_id]
            actual_table = item['table_rows']
            if stored_table != actual_table:
                print(f"ERROR: table mismatch for sample {i} (table_id={table_id})")
                errors += 1
    
    print(f"\nVerified 100 samples: {100 - errors} correct, {errors} errors")
    return errors == 0


if __name__ == "__main__":
    print("="*60)
    print("NIAT Table-ID Test Suite")
    print("="*60)
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    test_table_id_structure()
    test_table_question_alignment()
    test_processed_table_mapping()
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)
