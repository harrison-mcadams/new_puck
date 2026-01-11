
import re

try:
    with open('scripts/debug_imputation.txt', 'rb') as f:
        content = f.read().decode('utf-8', 'ignore')
    
    # squash whitespace
    content = re.sub(r'\s+', ' ', content)
    
    start_idx = content.find("Question 1")
    if start_idx == -1:
        print("Question 1 not found")
        exit(1)
        
    # Find next question or end
    end_idx = content.find("Question 2", start_idx)
    if end_idx == -1:
        text = content[start_idx:]
    else:
        text = content[start_idx:end_idx]
        
    with open('question_1.txt', 'w') as f:
        f.write(text)
        
    print(f"Extracted: {text[:50]}...")
    
except Exception as e:
    print(f"Error: {e}")
