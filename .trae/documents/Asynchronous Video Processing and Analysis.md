I will modify `monitor-method.py` to offload the blocking video saving and Qwen analysis tasks to a background thread pool. This will prevent the main video processing loop from freezing during file I/O and network requests.

### Implementation Plan

1.  **Import Concurrency Modules**:
    *   Add `import concurrent.futures` and `import threading` to manage background tasks and thread safety.

2.  **Initialize Thread Pool and Locks**:
    *   Create a `ThreadPoolExecutor` (e.g., with 2-4 workers) to handle background tasks.
    *   Create a `threading.Lock` to ensure `save_process_report` (which writes to a JSON file) is thread-safe, as it will be called from both the main loop and background threads.

3.  **Create a Background Task Function**:
    *   Define a new function `process_segment_background` that encapsulates the logic currently performed synchronously:
        *   Calling `save_segment_video`.
        *   Updating the shared `process_report` structure (e.g., adding the `video_link`).
        *   Calling `analyze_with_qwen` (if required for the specific segment).
        *   Updating the `process_report` with analysis results.
        *   Safely saving the report to disk using the lock.

4.  **Refactor Main Loop**:
    *   **Jump Detection Block**: Replace the direct calls to `save_segment_video` and `analyze_with_qwen` with a submission to the executor. Pre-calculate the output path and update the `saved_phases` set immediately in the main thread to prevent duplicate tasks.
    *   **Normal Segment Block**: Similarly, offload the video saving for completed steps.
    *   **Final Segment Block**: Offload the final save and analysis when the video ends or is interrupted.

5.  **Thread Safety**:
    *   Wrap all calls to `save_process_report` with the `report_lock` context manager.
    *   Ensure the main loop continues processing frames immediately after submitting tasks.

### Expected Outcome
*   The video feed will remain smooth and responsive even when saving video segments or waiting for Qwen API responses.
*   Report updates will occur asynchronously and be saved to disk as soon as processing completes.
