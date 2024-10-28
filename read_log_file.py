def extract_nan_sections(log_file_path, context_lines=5):
    nan_sections = []

    with open(log_file_path, 'r') as file:
        lines = file.readlines()

    # Iterate through lines to find "NaN" mentions
    for i, line in enumerate(lines):
        if 'NaN' in line:
            # Capture surrounding lines for context
            start = max(i - context_lines, 0)
            end = min(i + context_lines + 1, len(lines))
            nan_section = lines[start:end]
            nan_sections.append(''.join(nan_section))
            print(''.join(nan_section))  # Print section around NaN

    if not nan_sections:
        print("No 'NaN' entries found in the log file.")
    else:
        print(f"\nTotal 'NaN' sections found: {len(nan_sections)}")


# Usage
log_file_path = 'experiments/human_nerf/zju_mocap/p387/adventure/logs.txt'
extract_nan_sections(log_file_path)
