The goal is to synchronize the video processing logic in `app.py` with `monitor-method.py` and `LabFSM.py`. This will involve updating the `CLASSES` list, implementing background processing for video segmentation and analysis (to prevent UI freezing), and aligning the Finite State Machine (FSM) integration and segmentation logic.

### 1. Update Imports and Constants
- **Imports**: Add `concurrent.futures` and `threading` to `app.py` to support background tasks.
- **Classes**: Update the `CLASSES` list in `app.py` to match `monitor-method.py` (specifically adding `'试管架'`).

### 2. Implement Background Processing Helpers
- **Resource Management**: Use `st.cache_resource` to initialize and provide a shared `ThreadPoolExecutor` and `threading.Lock`. This ensures thread safety and prevents creating new executors on every script rerun.
- **Background Function**: Copy the `process_segment_background` function from `monitor-method.py` to `app.py`. This function handles:
    1.  Saving the video segment.
    2.  Updating the report with the video link.
    3.  Calling Qwen API for analysis.
    4.  Updating the report with analysis results.
    5.  Saving the report to disk.

### 3. Synchronize Helper Functions
- **`save_segment_video`**: Ensure it matches the implementation in `monitor-method.py`.
- **`update_step_status`**: Update to match the logic in `monitor-method.py` exactly.

### 4. Refactor Main Video Processing Loop
- **Logic Alignment**: Rewrite the `while cap.isOpened()` loop in `app.py` to mirror `monitor-method.py`:
    -   **Inference**: Use the same class mapping logic.
    -   **FSM Update**: Call `fsm.update` similarly.
    -   **Jump Detection**: Implement the logic that detects when steps are skipped (`fsm.current_step_index > prev_step_index + 1`), triggers error segment saving, and launches a background analysis task.
    -   **Normal Segmentation**: Implement the logic that detects step transitions and `start` labels to define segment boundaries (`current_segment_start`), saving valid segments in the background.
    -   **End of Video Handling**: Add the logic to handle the final open segment when the video ends or is stopped.
- **UI Updates**: Ensure the loop continues to update `video_placeholder`, `progress_bar`, and `status_text`.

### 5. Integration
- Connect the background tasks to the Streamlit session state so that updates to the report (from background threads) are eventually reflected in the UI.
