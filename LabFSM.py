import json
import os
import logging
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# Constants
ACT_FETCH = "拿取"
ACT_ADD   = "加料"
ACT_POUR  = "倾倒"
ACT_STIR  = "搅拌"
ACT_DROP  = "滴加"
ACT_PUT   = "放置"
ACT_WASH  = "刷洗"

# Mapping from JSON action strings to constants
ACTION_MAP = {
    "拿取": ACT_FETCH,
    "加料": ACT_ADD,
    "倾倒": ACT_POUR,
    "搅拌": ACT_STIR,
    "滴加": ACT_DROP,
    "放置": ACT_PUT,
    "刷洗": ACT_WASH
}

# Required frames default mapping
REQUIRED_FRAMES_MAP = {
    ACT_FETCH: 3,
    ACT_ADD: 1,     # Short action
    ACT_POUR: 5,   # Moderate
    ACT_STIR: 8,   # Long
    ACT_DROP: 6,   # Moderate
    ACT_PUT: 3,
    ACT_WASH: 3
}

# Equivalence group for bottles
BOTTLE_GROUP = {"药剂颗粒瓶", "硫酸铜颗粒广口PE瓶", "氢氧化钠颗粒广口PE瓶", "柠檬酸颗粒广口PE瓶"}

def is_same_obj(obj1, obj2):
    """
    Check if two objects are the same or belong to the same equivalence group.
    """
    if obj1 == obj2:
        return True
    if obj1 in BOTTLE_GROUP and obj2 in BOTTLE_GROUP:
        return True
    return False

class SpatialUtils:
    @staticmethod
    def calculate_iou(box1, box2):
        """
        Calculate Intersection over Union (IoU) of two bounding boxes.
        Boxes are in format [x1, y1, x2, y2].
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union_area = box1_area + box2_area - intersection_area
        
        if union_area == 0:
            return 0
        return intersection_area / union_area

    @staticmethod
    def get_center(box):
        """Calculate center point (cx, cy) of a box."""
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

    @staticmethod
    def calculate_distance(box1, box2):
        """Calculate Euclidean distance between centers of two boxes."""
        c1 = SpatialUtils.get_center(box1)
        c2 = SpatialUtils.get_center(box2)
        return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

    @staticmethod
    def is_close_and_above(upper_box, lower_box, horizontal_thresh=0.8, vertical_dist_thresh=200, class_names=None):
        """
        Check if upper_box is close and above lower_box.
        Used for Pouring/Dropping actions.
        """
        upper_cx = (upper_box[0] + upper_box[2]) / 2
        upper_cy = (upper_box[1] + upper_box[3]) / 2
        upper_h = upper_box[3] - upper_box[1]
        
        lower_cx = (lower_box[0] + lower_box[2]) / 2
        lower_cy = (lower_box[1] + lower_box[3]) / 2
        lower_w = lower_box[2] - lower_box[0]
        
        horizontal_dist = abs(upper_cx - lower_cx)
        is_horizontally_aligned = horizontal_dist < (lower_w * horizontal_thresh)
        
        is_above = upper_cy < lower_cy
        vertical_dist = lower_cy - upper_cy
        is_close_enough = vertical_dist < (vertical_dist_thresh + upper_h)
        
        return is_horizontally_aligned and is_above and is_close_enough

class ExperimentStep:
    def __init__(self, step_id, action_type, subjects, objects, required_frames=5, description="", label="none", raw_subject="", raw_object="", phase=""):
        self.step_id = step_id
        self.phase = phase
        self.action_type = action_type
        self.subjects = subjects  # List of allowed class names for subject (e.g., Hand, Spoon)
        self.objects = objects    # List of allowed class names for object (e.g., Bottle, Beaker)
        self.raw_subject = raw_subject
        self.raw_object = raw_object
        self.required_frames = required_frames
        self.current_frame_count = 0
        self.description = description
        self.label = label
        self.completed = False

        # State tracking variables
        self.prev_distance = None
        self.put_stage = 0  # 0: Wait for Hold, 1: Holding, 2: Released, 3: Leaving
        
        # ACT_FETCH specific state
        self.fetch_start_pos = None
        self.current_fetch_displacement = 0.0

    def reset(self):
        """Reset the step state to allow re-detection."""
        self.current_frame_count = 0
        self.completed = False
        self.prev_distance = None
        self.put_stage = 0
        self.fetch_start_pos = None
        self.current_fetch_displacement = 0.0

    def check(self, detections):
        """
        Check if the step condition is met based on detections.
        detections: list of dicts {'class': name, 'box': [x1,y1,x2,y2], 'conf': float}
        """
        found_subjects = [d for d in detections if d['class'] in self.subjects]
        found_objects = [d for d in detections if d['class'] in self.objects]

        condition_met = False

        # If no subjects/objects found, we might need to handle state carefully
        if not found_subjects or not found_objects:
            if self.action_type not in [ACT_PUT]:
                self.current_frame_count = max(0, self.current_frame_count - 1)
                self.prev_distance = None # Reset distance tracking if lost
            return False

        for subj in found_subjects:
            for obj in found_objects:
                # Skip if comparing same object
                if subj == obj:
                    continue

                # Calculate spatial metrics
                iou = SpatialUtils.calculate_iou(subj['box'], obj['box'])
                distance = SpatialUtils.calculate_distance(subj['box'], obj['box'])
                
                # --- ACT_FETCH Logic ---
                if self.action_type == ACT_FETCH:
                    # Criteria: Distance getting closer (or static hold) AND IoU > threshold
                    threshold = 0.02
                    is_approaching = True
                    if self.prev_distance is not None:
                        if distance > self.prev_distance + 5.0: 
                            is_approaching = False
                    
                    if iou > threshold and is_approaching:
                        condition_met = True
                        
                        # Track start position and displacement
                        obj_center = SpatialUtils.get_center(obj['box'])
                        if self.fetch_start_pos is None:
                            self.fetch_start_pos = obj_center
                        
                        # Calculate displacement
                        dx = obj_center[0] - self.fetch_start_pos[0]
                        dy = obj_center[1] - self.fetch_start_pos[1]
                        self.current_fetch_displacement = np.sqrt(dx*dx + dy*dy)

                    self.prev_distance = distance # Update for next frame
                    if condition_met:
                        break
                

                # --- ACT_PUT Logic ---
                elif self.action_type == ACT_PUT:
                    # Sequence: Start IoU>Thresh -> IoU=0 -> Distance Increasing
                    threshold = 0.02
                    
                    # State Machine
                    if self.put_stage == 0: # Wait for Hold
                        if iou > threshold:
                            self.put_stage = 1
                    
                    elif self.put_stage == 1: # Holding
                        if iou < 0.001: # Released (approx 0)
                            self.put_stage = 2
                        elif iou > threshold:
                            pass
                            
                    elif self.put_stage == 2: # Released
                        if iou > threshold:
                            # Grabbed again? Revert to holding
                            self.put_stage = 1
                        else:
                            # Check if leaving
                            if self.prev_distance is not None:
                                if distance > self.prev_distance + 2.0: # Increasing
                                    self.put_stage = 3
                                    condition_met = True
                                    self.completed = True
                                    return True # Immediate completion for PUT sequence
                    
                    self.prev_distance = distance
                    if self.put_stage == 3:
                         condition_met = True
                         break
                    if self.put_stage > 0:
                        break

                # --- Other Actions (STIR, WASH, POUR, ADD, DROP) ---
                else:
                    if self.action_type in [ACT_STIR, ACT_WASH, ACT_POUR, ACT_ADD]:
                        # Use IoU check
                        # target_thresh = 0.02 if self.action_type in [ACT_WASH, ACT_POUR, ACT_ADD] else 0.3
                        target_thresh = 0.02
                        if iou > target_thresh:
                            condition_met = True
                            break
                    
                    elif self.action_type in [ACT_DROP]:
                        # Use Above check
                        is_above = SpatialUtils.is_close_and_above(subj['box'], obj['box'], class_names=(subj['class'], obj['class']))
                        if is_above:
                            condition_met = True
                            break
            
            if condition_met:
                break

        # Counter logic
        if self.action_type == ACT_PUT:
            if self.completed:
                return True
            return False
            
        if condition_met:
            self.current_frame_count += 1
        else:
            self.current_frame_count = max(0, self.current_frame_count - 1)
            # Reset fetch tracking if continuity is lost
            if self.current_frame_count == 0:
                self.fetch_start_pos = None
                self.current_fetch_displacement = 0.0
        
        if self.action_type == ACT_FETCH:
             # Ensure 5 frames AND >30 pixels displacement
             return self.current_frame_count >= 5 and self.current_fetch_displacement > 30.0
            
        return self.current_frame_count >= self.required_frames

class LabFSM:
    def __init__(self):
        self.steps = []
        self.current_step_index = 0
        self.init_steps()

    def init_steps(self):
        """
        Initialize steps by reading from steps.json
        """
        json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'steps.json')
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                steps_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: {json_path} not found. LabFSM not initialized.")
            return
        except json.JSONDecodeError:
            print(f"Error: Failed to decode {json_path}.")
            return

        for step_data in steps_data:
            step_id = int(step_data.get('id', 0))
            phase = step_data.get('phase', '')
            action_str = step_data.get('action', '')
            subject = step_data.get('subject', '')
            obj = step_data.get('object', '')
            constraint = step_data.get('constraint', '')
            label = step_data.get('label', 'none')
            
            # Map action string to constant
            action_type = ACTION_MAP.get(action_str, action_str)
            
            # Determine required frames
            required_frames = REQUIRED_FRAMES_MAP.get(action_type, 5)
            # Custom adjustments
            if action_type == ACT_DROP and constraint and "多量" in constraint:
                required_frames = 15
            elif action_type == ACT_STIR:
                required_frames = 10
            elif action_type in [ACT_POUR, ACT_DROP]:
                required_frames = 8
            
            # Construct description
            description = f"{phase}: {action_str}{obj}"
            if constraint:
                description += f" ({constraint})"
                
            # Create ExperimentStep
            subjects_list = [subject] if subject else []
            objects_list = [obj] if obj else []
            
            # Alias handling (e.g., droppers)
            if subject == '胶头滴管':
                subjects_list.append('塑料滴管')
            
            step = ExperimentStep(
                step_id=step_id,
                action_type=action_type,
                subjects=subjects_list,
                objects=objects_list,
                required_frames=required_frames,
                description=description,
                label=label,
                raw_subject=subject,
                raw_object=obj,
                phase=phase
            )
            
            self.steps.append(step)

    def find_current_start_index(self):
        """
        Find the index of the 'start' step for the current phase.
        Searches backwards from current_step_index.
        """
        # If current step is start, it is the start
        # But if we are mid-way, we look back.
        # We start looking from current_step_index.
        # If current_step_index is out of bounds (completed), look at last step?
        idx = min(self.current_step_index, len(self.steps) - 1)
        
        while idx >= 0:
            if self.steps[idx].label == "start":
                return idx
            idx -= 1
        return 0 # Default to 0 if not found

    def find_next_start_index(self, current_index):
        """
        Find the index of the next 'start' step after current_index.
        """
        idx = current_index + 1
        while idx < len(self.steps):
            if self.steps[idx].label == "start":
                return idx
            idx += 1
        return None

    def calculate_trigger_index(self, current_start_idx, next_start_idx):
        """
        Calculate the step index that, once completed, triggers listening for the next start step.
        """
        next_start_step = self.steps[next_start_idx]
        target_subj = next_start_step.raw_subject
        target_obj = next_start_step.raw_object

        # Look back from next_start_idx - 1 down to current_start_idx
        # to find the LAST step with same subject and object
        match_idx = -1
        for i in range(next_start_idx - 1, current_start_idx - 1, -1):
            step = self.steps[i]
            # Check subject and object match
            # "Hand" vs "Hand", "Bottle" vs "Bottle" (with equivalence)
            if step.raw_subject == target_subj and is_same_obj(step.raw_object, target_obj):
                match_idx = i
                break
        
        if match_idx != -1:
            # "当这组...的后面一组step执行完之后"
            # So we trigger after match_idx + 1 is completed.
            # Completed means current_step_index > match_idx + 1.
            return match_idx + 1
        else:
            # "没有...匹配...从当前第一个start动作的下一个动作执行完后"
            # Start is current_start_idx. Next action is current_start_idx + 1.
            # Trigger after current_start_idx + 1 is completed.
            return current_start_idx + 1

    def update(self, detections, timestamp_str=""):
        if self.current_step_index >= len(self.steps):
            return True # Completed

        # --- Look-Ahead Logic for Skipped Steps ---
        current_start_idx = self.find_current_start_index()
        next_start_idx = self.find_next_start_index(self.current_step_index)

        if next_start_idx is not None:
            trigger_idx = self.calculate_trigger_index(current_start_idx, next_start_idx)
            
            # If we have passed the trigger point (meaning trigger step is completed)
            if self.current_step_index > trigger_idx:
                next_step = self.steps[next_start_idx]
                # Check the next start step in parallel
                if next_step.check(detections):
                    print(f"[{timestamp_str}] JUMP DETECTED: Skipped to Step {next_step.step_id} ({next_step.description})")
                    logging.info(f"[{timestamp_str}] JUMP DETECTED: Skipped from Step {self.steps[self.current_step_index].step_id} to Step {next_step.step_id}")
                    
                    # Update index to the step AFTER the detected start step
                    # Because check() returning True means that start step is DONE.
                    self.current_step_index = next_start_idx + 1
                    return True

        # --- Normal Sequential Logic ---
        current_step = self.steps[self.current_step_index]
        if current_step.check(detections):
            print(f"[{timestamp_str}] Step {current_step.step_id} Completed: {current_step.description}")
            self.current_step_index += 1
            return True
        
        return False

    def get_status(self):
        if self.current_step_index >= len(self.steps):
            return "实验全部完成", ""
        
        current = self.steps[self.current_step_index]
        return f"Step {current.step_id}/{len(self.steps)}", current.description

    def check_all(self, detections):
        """
        Check all steps against the current detections.
        Returns a list of ExperimentStep objects that are currently satisfied/completed.
        Does NOT enforce order.
        """
        completed_steps = []
        for step in self.steps:
            if step.check(detections):
                completed_steps.append(step)
        return completed_steps
