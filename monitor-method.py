import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import os
import logging
import argparse
import json
from datetime import datetime
import concurrent.futures
import threading
from LabFSM import LabFSM
import generate_step
from qwen_api import analyze_with_qwen

# Constants
ACT_FETCH = "拿取"  # Check IoU
ACT_ADD   = "加料"  # Check IoU
ACT_POUR  = "倾倒"  # Check IoU
ACT_STIR  = "搅拌"  # Check IoU (Inside)
ACT_DROP  = "滴加"  # Check Above
ACT_PUT   = "放置"  # Check IoU
ACT_WASH  = "刷洗"  # Check IoU (Inside)

# YOLO Classes
# Use CLASSES from generate_step to ensure consistency
CLASSES = [
    '酚酞试剂', '石蕊试剂', '药剂颗粒瓶', '药剂颗粒瓶', '药剂颗粒瓶', '试管', '烧杯', '玻璃棒', '量筒', '胶头滴管', 
    '广口玻璃瓶', '塑料滴管', '点滴板', '废液缸', '药匙', '手', '蒸馏水水杯', '试管架'
]

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
        # Center points
        upper_cx = (upper_box[0] + upper_box[2]) / 2
        upper_cy = (upper_box[1] + upper_box[3]) / 2
        upper_h = upper_box[3] - upper_box[1]
        
        lower_cx = (lower_box[0] + lower_box[2]) / 2
        lower_cy = (lower_box[1] + lower_box[3]) / 2
        lower_w = lower_box[2] - lower_box[0]
        
        # Horizontal alignment check
        # The upper object should be roughly within the width of the lower object
        horizontal_dist = abs(upper_cx - lower_cx)
        is_horizontally_aligned = horizontal_dist < (lower_w * horizontal_thresh)
        
        # Vertical check
        # Upper object center should be above lower object center
        # And distance shouldn't be too large
        is_above = upper_cy < lower_cy
        vertical_dist = lower_cy - upper_cy
        is_close_enough = vertical_dist < (vertical_dist_thresh + upper_h) # Adaptive threshold
        
        if class_names:
            msg = (f"Spatial Check [{class_names[0]} vs {class_names[1]}]: "
                   f"Horiz={is_horizontally_aligned} (dist={horizontal_dist:.1f}), "
                   f"Above={is_above}, "
                   f"Close={is_close_enough} (dist={vertical_dist:.1f})")
            logging.info(msg)
            # print(msg) # Optional: print to console too if needed, but user asked for log

        return is_horizontally_aligned and is_above and is_close_enough


def save_segment_video(video_path, output_path, start_frame, end_frame):
    """
    Save a segment of the video to a new file.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Could not open source video for export: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        curr_f = start_frame
        
        while curr_f < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            # Save one frame every two frames
            if (curr_f - start_frame) % 2 == 0:
                out.write(frame)
            curr_f += 1
            
        out.release()
        cap.release()
        logging.info(f"Exported segment: {output_path}")
        print(f"Exported segment: {output_path}")
    except Exception as e:
        logging.error(f"Error exporting segment {output_path}: {e}")

def load_process_report(base_dir):
    path = os.path.join(base_dir, 'process_structure.json')
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_process_report(report, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=4)
    print(f"Experiment report saved to {file_path}")

def update_step_status(report, phase, step_id, status, skipped_ids=None):
    """
    Update step status in the report.
    status: "yes" (passed) or "no" (failed/skipped)
    """
    for process in report:
        if process['process_name'] == phase:
            # Update specific step
            if step_id is not None:
                for step in process['steps']:
                    if int(step['step_id']) == int(step_id):
                        step['passed'] = status
                        break
            
            # Update skipped steps
            if skipped_ids:
                missing_str = []
                for s_id in skipped_ids:
                    for step in process['steps']:
                        if int(step['step_id']) == int(s_id):
                            step['passed'] = "no"
                            missing_str.append(f"step{s_id}")
                
                if missing_str:
                    current_missing = process['missing_incorrect_steps']
                    new_missing = ", ".join(missing_str)
                    if current_missing:
                        process['missing_incorrect_steps'] = current_missing + ", " + new_missing
                    else:
                        process['missing_incorrect_steps'] = new_missing

            # Update Process State
            # Simple logic: if any step passed -> executing. If last step passed -> completed.
            # But we might need more robust logic.
            # Let's assume sequential execution.
            all_steps_count = len(process['steps'])
            passed_count = sum(1 for s in process['steps'] if s.get('passed') == 'yes')
            
            if passed_count == all_steps_count:
                process['state'] = "completed"
            elif passed_count > 0:
                process['state'] = "executing"
            
            break

def process_segment_background(video_path, out_path, start_frame, end_frame, 
                             process_entry, step_key, should_analyze, 
                             report_path, full_report, report_lock):
    """
    Background task to save video and optionally analyze it.
    """
    try:
        # 1. Save Video
        save_segment_video(video_path, out_path, start_frame, end_frame)
        
        # 2. Update Video Link
        if process_entry:
            process_entry['video_link'] = out_path
            # If we are analyzing, we might keep pass_check as False (or whatever was set before)
            # until analysis returns.
        
        # 3. Analyze with Qwen
        if should_analyze:
            print(f"Analyzing {step_key} with Qwen (Background)...")
            try:
                analysis = analyze_with_qwen(out_path, step_key=step_key)
                log_msg = f"Qwen Analysis [{step_key}]: {analysis}"
                print(log_msg)
                logging.info(log_msg)
                
                # Parse JSON
                try:
                    clean_json = analysis.strip()
                    if clean_json.startswith("```json"):
                        clean_json = clean_json[7:]
                    if clean_json.endswith("```"):
                        clean_json = clean_json[:-3]
                    analysis_data = json.loads(clean_json)
                    qwen_pass = analysis_data.get('pass_check', False)
                    qwen_text = analysis_data.get('qwen_analysis', analysis)
                except Exception:
                    qwen_pass = False
                    qwen_text = analysis

                if process_entry:
                    process_entry['pass_check'] = qwen_pass
                    process_entry['qwen_analysis'] = qwen_text
            except Exception as e:
                err_msg = f"Qwen Analysis Failed [{step_key}]: {e}"
                print(err_msg)
                logging.error(err_msg)

        # 4. Save Report
        with report_lock:
            save_process_report(full_report, report_path)
            
    except Exception as e:
        logging.error(f"Background task failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Monitor Chemical Experiment")
    parser.add_argument("--init", action="store_true", help="Initialize steps.json from raw steps before monitoring")
    args = parser.parse_args()

    # Paths
    # Use relative paths for portability
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Initialization
    if args.init:
        print("Initializing steps...")
        current_dir = os.getcwd()
        try:
            os.chdir(base_dir)
            generate_step.main()
        except Exception as e:
            print(f"Error during initialization: {e}")
        finally:
            os.chdir(current_dir)

    model_path = os.path.join(base_dir, 'weights', 'yolo11m-best01222.pt')
    video_path = os.path.join(base_dir, './videos/ChemicalExperimentNOSOP.mp4') 

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    # Setup Logging
    log_path = os.path.join(base_dir, 'experiment_log.txt')
    # Use FileHandler to ensure UTF-8 encoding for Chinese characters
    file_handler = logging.FileHandler(filename=log_path, mode='w', encoding='utf-8')
    logging.basicConfig(
        handlers=[file_handler],
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )

    # Load Model
    print("Loading YOLO model...")
    model = YOLO(model_path)
    
    # Verify Model Classes
    print("Model Classes:", model.names)
    
    # Initialize FSM
    fsm = LabFSM()
    
    # Load Process Report
    process_report = load_process_report(base_dir)
    
    # Generate report path with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f'experiment_result_{timestamp}.json'
    report_path = os.path.join(base_dir, report_filename)
    
    # Segmentation variables
    seg_count = 0
    current_segment_start = None
    last_saved_end_frame = 0
    
    # Process tracking
    total_processes = len(process_report)
    saved_phases = set()
    
    # Ensure output directory exists
    output_dir = os.path.join(base_dir, 'videos', 'SOP_step')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize ThreadPoolExecutor and Lock
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
    report_lock = threading.Lock()

    # Check initial step for 'start' label
    if fsm.steps and fsm.steps[0].label == 'start':
        current_segment_start = 0
        logging.info(f"Segment Started at frame 0 (Step {fsm.steps[0].step_id})")
    
    # Open Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    print("Starting inference...")
    
    frame_count = -1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        # Process one frame every two frames
        if frame_count % 2 != 0:
            continue

        # Inference
        results = model(frame, verbose=False)[0]
        
        # Process Detections
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            # Map model class ID (English) to our Chinese class list
            # The model output is in English, but our logic expects Chinese.
            # We assume the order of CLASSES matches the model's class IDs.
            if 0 <= cls_id < len(CLASSES):
                cls_name = CLASSES[cls_id]
            else:
                # Fallback or skip if ID is unexpected
                continue
                
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            
            detections.append({
                'class': cls_name,
                'id': cls_id,
                'box': [x1, y1, x2, y2],
                'conf': conf
            })

        # Update FSM
        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        total_seconds = int(timestamp_ms / 1000)
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        timestamp_str = f"{minutes:02d}:{seconds:02d}"
        
        prev_step_index = fsm.current_step_index
        fsm.update(detections, timestamp_str)

        # Check for step transition
        if fsm.current_step_index != prev_step_index:
            fps = cap.get(cv2.CAP_PROP_FPS)
            current_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            # Identify completed step
            if fsm.current_step_index > 0:
                completed_step = fsm.steps[fsm.current_step_index - 1]
                update_step_status(process_report, completed_step.phase, completed_step.step_id, "yes")
            
            # --- Jump Detection and Analysis ---
            if fsm.current_step_index > prev_step_index + 1:
                # Identify skipped steps
                # Steps from prev_step_index up to (but not including) the one just completed (fsm.current_step_index - 1)
                skipped_indices = range(prev_step_index, fsm.current_step_index - 1)
                skipped_step_ids = [fsm.steps[i].step_id for i in skipped_indices]
                
                if skipped_indices:
                    print(f"Jump detected! Skipped steps: {skipped_step_ids}")
                    logging.warning(f"Jump detected! Skipped steps: {skipped_step_ids}")
                    
                    # Update Report for skipped steps
                    first_skipped = fsm.steps[skipped_indices[0]]
                    update_step_status(process_report, first_skipped.phase, None, "no", skipped_ids=skipped_step_ids)
                    
                    # Save error segment
                    # From last valid save point to current frame
                    error_start = last_saved_end_frame
                    error_end = current_frame_idx
                    
                    if error_end > error_start:
                        # Check if this process is already saved or if we exceeded total processes
                        if first_skipped.phase not in saved_phases and seg_count < total_processes:
                            seg_count += 1
                            seg_name = f"SOP_step_{seg_count}"
                            out_path = os.path.join(output_dir, f"{seg_name}.mp4")
                            
                            saved_phases.add(first_skipped.phase)
                            
                            # Update Report with Video Link
                            target_process = None
                            for process in process_report:
                                if process['process_name'] == first_skipped.phase:
                                    target_process = process
                                    process['video_link'] = out_path
                                    process['pass_check'] = False
                                    break
                            
                            current_step_key = f"step{seg_count:02d}"
                            print(f"Submitting background task (Jump) for {current_step_key}...")
                            executor.submit(process_segment_background, 
                                          video_path, out_path, error_start, error_end,
                                          target_process, current_step_key, True,
                                          report_path, process_report, report_lock)
                        
                        last_saved_end_frame = error_end

            # --- Normal Segmentation Logic ---
            # Iterate through all steps that we passed or landed on
            start_check_idx = prev_step_index + 1
            end_check_idx = min(fsm.current_step_index + 1, len(fsm.steps))
            
            for i in range(start_check_idx, end_check_idx):
                step = fsm.steps[i]
                
                if step.label == 'start':
                    # Start of a new flow
                    # 1. Close previous segment if any (Overlap/Gap handling)
                    if current_segment_start is not None:
                         # End 1 second before this start
                         end_frame = max(0, int(current_frame_idx))
                         if end_frame > current_segment_start:
                             # Determine previous phase
                             prev_phase = fsm.steps[i-1].phase if i > 0 else None

                             if prev_phase and prev_phase not in saved_phases and seg_count < total_processes:
                                 seg_count += 1
                                 seg_name = f"SOP_step_{seg_count}"
                                 out_path = os.path.join(output_dir, f"{seg_name}.mp4")
                                 
                                 saved_phases.add(prev_phase)
                                 
                                 # Update Report with Video Link (Previous Process)
                                 target_process = None
                                 for process in process_report:
                                     if process['process_name'] == prev_phase:
                                         target_process = process
                                         process['video_link'] = out_path
                                         process['pass_check'] = True
                                         break
                                 
                                 current_step_key = f"step{seg_count:02d}"
                                 executor.submit(process_segment_background,
                                               video_path, out_path, current_segment_start, end_frame,
                                               target_process, current_step_key, False,
                                               report_path, process_report, report_lock)
                             
                             last_saved_end_frame = end_frame
                    
                    # 2. Start new segment
                    # Start 1 second before
                    start_frame = max(0, int(current_frame_idx - fps/2))
                    current_segment_start = start_frame
                    logging.info(f"Segment Started at frame {start_frame} (Step {step.step_id})")
            
            # Save Report
            with report_lock:
                save_process_report(process_report, report_path)
        
        # Visualization
        for d in detections:
            x1, y1, x2, y2 = map(int, d['box'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw numeric label
            label = str(d['id'])
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Show Progress bar
        if fsm.steps:
            progress = fsm.current_step_index / len(fsm.steps)
            bar_width = int(frame.shape[1] * progress)
            cv2.rectangle(frame, (0, frame.shape[0]-20), (bar_width, frame.shape[0]), (0, 255, 0), -1)

        cv2.imshow('Chemistry Experiment Monitor', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Check if there is an open segment when loop ends
    if current_segment_start is not None:
         # Use frame_count + 1 as end_frame because cap.get(cv2.CAP_PROP_POS_FRAMES) might be unreliable at EOF
         end_frame = frame_count + 1
         
         if end_frame > current_segment_start:
             # Identify target process
             target_process = None
             for process in reversed(process_report):
                 # Also check for 'completed' state, as the last process might have finished just before video ended
                 if process['state'] in ['executing', 'completed']:
                     target_process = process
                     break
             
             should_save = False
             if target_process:
                 if target_process['process_name'] not in saved_phases and seg_count < total_processes:
                     should_save = True
             elif seg_count < total_processes:
                 should_save = True
                 
             if should_save:
                 seg_count += 1
                 seg_name = f"SOP_step_{seg_count}"
                 out_path = os.path.join(output_dir, f"{seg_name}.mp4")
                 logging.info(f"Saving final segment (Interrupted/Ended): {seg_name}")
                 
                 should_analyze = False
                 current_step_key = f"step{seg_count:02d}"
                 
                 if target_process:
                    target_process['video_link'] = out_path
                    saved_phases.add(target_process['process_name'])
                    
                    # Check if all steps in the current process are completed
                    all_steps_passed = True
                    for step in target_process.get('steps', []):
                        if step.get('passed') != 'yes':
                            all_steps_passed = False
                            break
                    
                    target_process['pass_check'] = all_steps_passed
                    
                    if all_steps_passed == False:
                        should_analyze = True
                        print(f"Submitting background task (Final) for {current_step_key} with Qwen...")
                 
                 executor.submit(process_segment_background,
                               video_path, out_path, current_segment_start, end_frame,
                               target_process, current_step_key, should_analyze,
                               report_path, process_report, report_lock)
             
             last_saved_end_frame = end_frame

    # Final Save
    with report_lock:
        save_process_report(process_report, report_path)
    
    print("Shutting down executor...")
    executor.shutdown(wait=True)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
