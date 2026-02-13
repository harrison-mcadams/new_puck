import subprocess
import sys
import re

def main():
    if len(sys.argv) < 2:
        print("Usage: python smart_runner.py <command>")
        sys.exit(1)

    # Parse custom arguments for the runner itself if they exist
    # usage: python smart_runner.py [runner_args] -- command [command_args]
    # But for simplicity, let's just look for a specialized env var or check sys.argv manually
    # Or just use a simple separator?
    # Let's assume the first arg after smart_runner.py *could* be a filter if it starts with --filter=
    
    # Smart Debugging Filter Logic
    # If a filter is provided via CLI, use it.
    # Otherwise, if we are in a debugging mode (implied), use a default set of interesting keywords.
    
    cmd_start_idx = 1
    output_filter = None
    
    if len(sys.argv) > 1 and sys.argv[1].startswith("--filter="):
        output_filter = sys.argv[1].split("=", 1)[1]
        cmd_start_idx = 2
        
    command = sys.argv[cmd_start_idx:]
    if not command:
         print("Usage: python smart_runner.py [--filter=TEXT] <command>")
         sys.exit(1)
         
    print(f"--- Smart Runner Executing: {' '.join(command)} ---")
    
    if output_filter:
        print(f"--- Filtering Output for: '{output_filter}' ---")
    else:
        # Default smart filters for this task
        default_filters = ['Stats', 'Mean', 'Density', 'logloss', 'is_goal', 'Base Model']
        print(f"--- Using Default Smart Filters: {default_filters} ---")

    try:
        # Run the command and capture output
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False 
        )

        stdout = result.stdout
        stderr = result.stderr
        
        full_output = stdout + "\n" + stderr
        
        # Check for success
        if result.returncode == 0:
            print("--- EXECUTION SUCCESS ---")
            
            lines = full_output.splitlines()
            if output_filter:
                matching = [l for l in lines if output_filter in l]
                print(f"--- Found {len(matching)} matching lines ---")
                print("\n".join(matching))
            else:
                 # Use default filters
                 matching = [l for l in lines if any(k in l for k in default_filters)]
                 if matching:
                     print(f"--- Smart Filter Found {len(matching)} interesting lines ---")
                     print("\n".join(matching))
                 else:
                     # Fallback
                     if len(lines) > 20:
                        print("... (output truncated) ...")
                        print("\n".join(lines[-20:]))
                     else:
                        print(full_output)
        else:
            print(f"--- EXECUTION FAILED (Exit Code: {result.returncode}) ---")
            
            # Smart filtering for errors
            # 1. Tracebacks
            traceback_pattern = re.compile(r'Traceback \(most recent call last\):.*', re.DOTALL)
            match = traceback_pattern.search(full_output)
            if match:
                print("\n--- TRACEBACK FOUND ---")
                print(match.group(0))
            else:
                # 2. Look for "Error", "Exception"
                print("\n--- ERROR SEARCH (Last 30 lines) ---")
                lines = full_output.splitlines()
                # Simple heuristic: print last 30 lines regardless, often contains the error at end
                if len(lines) > 30:
                    print("... (preceding output truncated) ...")
                    print("\n".join(lines[-30:]))
                else:
                    print(full_output)

    except Exception as e:
        print(f"Smart Runner Internal Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
