
import os

target_file = r'c:\Users\harri\Desktop\new_puck\scripts\nested_model_dashboard.py'

str_to_find = "function transform_tensor(inputs, config) {"
injection = """    function transform_tensor(inputs, config) {
        let debug = false;
        if (inputs.distance && Math.abs(inputs.distance - 39.0) < 0.1 && inputs.angle_deg && Math.abs(inputs.angle_deg - 90.0) < 0.1) {
             debug = true;
             console.log("DEBUG: HIT transform_tensor", inputs);
        }
"""

with open(target_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
found = False
for line in lines:
    if str_to_find in line:
        found = True
        # Keep indentation if possible, but JS is forgiving.
        # We replace the line with our injection
        # check indentation
        indent = line[:line.find("function")]
        # Apply indentation to injection lines
        injected_lines = injection.split('\n')
        # fix first line which is already indented in injection string?
        # actually my injection string has indentation.
        # Let's just use the injection string as is, assuming 4 spaces.
        
        # Actually safer to just append the debug logic AFTER the line.
        new_lines.append(line)
        debug_code = """        let debug = false;
        if (inputs.distance && Math.abs(inputs.distance - 39.0) < 0.1 && inputs.angle_deg && Math.abs(inputs.angle_deg - 90.0) < 0.1) {
             debug = true;
             console.log("DEBUG: HIT transform_tensor", inputs);
        }
"""
        new_lines.append(debug_code)
    else:
        new_lines.append(line)
        
    # Also inject logging inside loop
    if "let b = bspline_basis(val, dim.knots, dim.degree);" in line:
        new_lines.append("            if (debug) console.log('Basis raw for ' + dim.col, b);\n")
        
    if "bases.push(b);" in line:
        new_lines.insert(len(new_lines)-1, "            if (debug) console.log('Basis dropped for ' + dim.col, b);\n") 
        # Actually insert BEFORE push logic? 
        # The line is appended, so if I want it before, I should append before.
        # Wait, the logic is: append(line).
        # So "bases.push(b);" is in new_lines.
        
        # Let's do:
        # if "bases.push(b);" in line:
        #    new_lines.append("            if (debug) console.log('Basis dropped for ' + dim.col, b);\n")
        #    new_lines.append(line)
        # BUT I already appended line in the `else` or start of loop?
        # The loop structure above is: `if str_to_find... else: append(line)`.
        # So I need to modify the flow.
        pass

# Easier approach: Read whole content, replace strings.
with open(target_file, 'r', encoding='utf-8') as f:
    content = f.read()

if str_to_find not in content:
    print(f"Error: Could not find '{str_to_find}'")
    # try with different spacing
    str_to_find_alt = "function transform_tensor(inputs, config){"
    if str_to_find_alt in content:
        str_to_find = str_to_find_alt
    else:
        print("Failed to find function signature.")
        exit(1)

# Injection 1: Start of function
replace_1 = """function transform_tensor(inputs, config) {
        let debug = false;
        if (inputs.distance && Math.abs(inputs.distance - 39.0) < 0.1 && inputs.angle_deg && Math.abs(inputs.angle_deg - 90.0) < 0.1) {
             debug = true;
             console.log("DEBUG: HIT transform_tensor", inputs);
        }"""
content = content.replace(str_to_find, replace_1)

# Injection 2: Inside loop (Basis Raw)
# specific line: let b = bspline_basis(val, dim.knots, dim.degree);
# We append log after it.
find_2 = "let b = bspline_basis(val, dim.knots, dim.degree);"
replace_2 = """let b = bspline_basis(val, dim.knots, dim.degree);
            if (debug) console.log('Basis raw ' + dim.col, b);"""
content = content.replace(find_2, replace_2)

# Injection 3: Before push (Basis Dropped)
# find: bases.push(b);
# Insert before.
find_3 = "bases.push(b);"
replace_3 = """if (debug) console.log('Basis dropped ' + dim.col, b);
            bases.push(b);"""
content = content.replace(find_3, replace_3)

# Injection 4: After outer product (vec)
# find: // 3. Slice to match coefs/scaler
find_4 = "// 3. Slice to match coefs/scaler"
replace_4 = """if (debug) console.log('Outer vec', vec);
        // 3. Slice to match coefs/scaler"""
content = content.replace(find_4, replace_4)

# Write back
with open(target_file, 'w', encoding='utf-8') as f:
    f.write(content)

print("Successfully patched file.")
