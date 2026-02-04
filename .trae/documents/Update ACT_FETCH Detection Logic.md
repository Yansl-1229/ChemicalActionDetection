I will modify `f:\Code\SOPActionRecognition\化学\20260202\LabFSM.py` to update the `ACT_FETCH` detection logic.

**Implementation Steps:**

1. **Update** **`ExperimentStep.__init__`** **and** **`ExperimentStep.reset`**:

   * Add state variables `self.fetch_start_pos` (to track the object's initial position) and `self.current_fetch_displacement` (to track movement distance).

2. **Modify** **`ExperimentStep.check`**:

   * Inside the `ACT_FETCH` block (around line 164):

     * When `iou > threshold` is met, capture `self.fetch_start_pos` if it's not already set.

     * Calculate the displacement of the object from `self.fetch_start_pos`.

   * Update the return logic (end of method):

     * For `ACT_FETCH`, explicitly require `self.current_frame_count >= 5` AND `self.current_fetch_displacement > 30.0` to consider the step complete.

   * Add logic to reset `self.fetch_start_pos` if the frame count drops to 0 (tracking lost).

**Key Changes:**

* **Frame Count**: Enforce a minimum of 5 frames (overriding the default 3 for `ACT_FETCH`).

* **Displacement**: Enforce a displacement of > 30 pixels.

