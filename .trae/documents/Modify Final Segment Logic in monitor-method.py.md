I will modify `monitor-method.py` to implement the requested logic for the final video segment.

**Changes to** **`monitor-method.py`:**

1. **Locate the final segment handling block** (lines 463-495).
2. **Inside the** **`if should_save:`** **block**, after saving the video and updating `target_process['video_link']`:

   * **Calculate** **`pass_check`**: Iterate through all steps of the `target_process`. If all steps have `passed == 'yes'`, set `target_process['pass_check'] = True`, otherwise `False`.

   * **Invoke Qwen-VL Analysis**: Call `analyze_with_qwen` for the saved video segment using the generated step key (e.g., `step01`, `step02`).

   * **Store Analysis**: Save the result into `target_process['qwen_analysis']`.

   * **Add Logging**: Log the analysis result or any errors.

**Implementation Detail:**

```python
if target_process:
    target_process['video_link'] = out_path
    saved_phases.add(target_process['process_name'])
    
    # 1. Check completion status
    all_passed = True
    if 'steps' in target_process:
        for step in target_process['steps']:
            if step.get('passed') != 'yes':
                all_passed = False
                break
    target_process['pass_check'] = all_passed
    
    # 2. Analyze with Qwen
    current_step_key = f"step{seg_count:02d}"
    print(f"Analyzing final segment for {current_step_key}...")
    try:
        analysis = analyze_with_qwen(out_path, step_key=current_step_key)
        target_process['qwen_analysis'] = analysis
        logging.info(f"Qwen Analysis [{current_step_key}]: {analysis}")
    except Exception as e:
        err_msg = f"Qwen Analysis Failed [{current_step_key}]: {e}"
        print(err_msg)
        logging.error(err_msg)
```

