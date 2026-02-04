import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import os
import logging
import json
import time
from datetime import datetime
import threading
import concurrent.futures
from LabFSM import LabFSM
from qwen_api import analyze_with_qwen
from openai import OpenAI

# ==========================================
# Configuration & Setup
# ==========================================
st.set_page_config(page_title="化学实验动作识别", layout="wide")

# CSS for styling
st.markdown("""
<style>
    .step-card {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
        border: 1px solid #ddd;
    }
    h1 { padding-top: 0rem; }
    .status-pending { background-color: #f0f2f6; }
    .status-active { background-color: #fff8c5; border-color: #ffe58f; }
    .status-processing { background-color: #e6f3ff; border-color: #2196f3; }
    .status-pass { background-color: #e6fffa; border-color: #00cc88; }
    .status-fail { background-color: #fff5f5; border-color: #ff4d4f; }
    
    div[data-testid="stVerticalBlock"] > div {
        gap: 1rem;
    }
    /* 聊天记录字体大小调整 */
    [data-testid="stChatMessage"] .stMarkdown * {
        font-size: 14px !important;
    }
</style>
""", unsafe_allow_html=True)

# Constants from monitor-method.py
CLASSES = [
    '酚酞试剂', '石蕊试剂', '药剂颗粒瓶', '药剂颗粒瓶', '药剂颗粒瓶', '试管', '烧杯', '玻璃棒', '量筒', '胶头滴管', 
    '广口玻璃瓶', '塑料滴管', '点滴板', '废液缸', '药匙', '手', '蒸馏水水杯', '试管架'
]

# Initialize Session State
if 'process_report' not in st.session_state:
    st.session_state.process_report = []
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'processing_active' not in st.session_state:
    st.session_state.processing_active = False

# Resource Caching for Executor and Lock
@st.cache_resource
def get_executor():
    return concurrent.futures.ThreadPoolExecutor(max_workers=3)

@st.cache_resource
def get_report_lock():
    return threading.Lock()

# ==========================================
# Helper Functions
# ==========================================

def load_process_report(base_dir):
    path = os.path.join(base_dir, 'process_structure.json')
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_process_report(report, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=4)

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

def chat_with_qwen_api(messages, context_report):
    # Construct system prompt with report context
    report_str = json.dumps(context_report, ensure_ascii=False, indent=2)
    system_prompt = f"""
    你是一个化学实验助手。当前实验的检测报告如下：
    {report_str}
    
    请根据上述报告回答用户的问题。如果报告中有"missing_incorrect_steps"，请重点指出。
    """
    
    # Prepare messages
    api_messages = [{"role": "system", "content": system_prompt}]
    for msg in messages:
        api_messages.append({"role": msg["role"], "content": msg["content"]})
        
    try:
        client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        completion = client.chat.completions.create(
            model="qwen3-max",
            messages=api_messages,
            extra_body={"enable_thinking": False},
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"AI Service Error: {str(e)}"

# ==========================================
# UI Layout
# ==========================================

col_left, col_mid, col_right = st.columns([1, 3, 1.2], gap="medium")

# Load initial report structure
base_dir = os.path.dirname(os.path.abspath(__file__))
if not st.session_state.process_report:
    st.session_state.process_report = load_process_report(base_dir)

step_placeholders = {}
 
# 1. Left Column: Steps
with col_left:
    st.markdown("### 实验步骤")
    # Initialize placeholders
    for process in st.session_state.process_report:
        step_placeholders[process['process_name']] = st.empty()

    def update_sidebar_ui():
        for process in st.session_state.process_report:
            p_name = process['process_name']
            state = process.get('state', 'not executed')
            missing = process.get('missing_incorrect_steps', '')
            
            if state == 'completed':
                status_icon = "✅" if not missing else "❌"
            elif state == 'executing':
                status_icon = "⏳"
            else:
                status_icon = "⚪"
            
            placeholder = step_placeholders[p_name]
            
            # Direct replacement approach
            with placeholder:
                 with st.expander(f"{status_icon} {p_name}", expanded=(state != 'not executed')):
                    st.write(f"状态: {state}")
                    if missing:
                        st.error(f"缺失/错误步骤: {missing}")
                    elif state == 'completed':
                        st.success("步骤通过")
                    
                    for step in process['steps']:
                        s_passed = step.get('passed', 'no')
                        s_icon = "✔️" if s_passed == 'yes' else ".."
                        st.caption(f"{s_icon} {step['description']}")

    # Initial Render
    update_sidebar_ui()

# 2. Middle Column: Video & Control
with col_mid:
    st.markdown("<h1 style='text-align: center;'>化学实验动作识别</h1>", unsafe_allow_html=True)
    
    # Input Source
    input_source = st.radio("选择输入", ["上传视频", "使用示例视频", "摄像头"], label_visibility="collapsed", horizontal=True)
    video_path = None
    
    if input_source == "上传视频":
        uploaded_file = st.file_uploader("选择视频文件", type=["mp4", "avi"], label_visibility="collapsed")
        if uploaded_file:
            os.makedirs("videos", exist_ok=True)
            temp_path = os.path.join("videos", "temp_upload.mp4")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            video_path = temp_path
    elif input_source == "使用示例视频":
        sample_path = "./videos/示例视频01.mp4"
        if os.path.exists(sample_path):
            video_path = sample_path
            st.caption(f"已加载: {sample_path}")
        else:
            st.error("未找到示例视频。")
    elif input_source == "摄像头":
        video_path = 0

    video_placeholder = st.empty()
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    c1, c2 = st.columns(2)
    with c1:
        start_btn = st.button("▶ 开始分析", type="primary", width="stretch", disabled=(video_path is None))
    with c2:
        stop_btn = st.button("⏹ 停止", width="stretch")

    if stop_btn:
        st.session_state.processing_active = False

    if start_btn:
        st.session_state.processing_active = True

# 3. Right Column: Chat
with col_right:
    st.markdown("### AI 助手")
    chat_container = st.container(height=500, border=True)
    
    with chat_container:
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                
    if prompt := st.chat_input("输入问题..."):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.write(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = chat_with_qwen_api(st.session_state.chat_messages, st.session_state.process_report)
                    st.write(response)
                    st.session_state.chat_messages.append({"role": "assistant", "content": response})

# ==========================================
# Main Processing Loop
# ==========================================
if st.session_state.processing_active and video_path is not None:

    # Initialize Logic Components
    model_path = os.path.join(base_dir, 'weights', 'yolo11m-best01222.pt')
    if not os.path.exists(model_path):
        st.error(f"Model not found: {model_path}")
        st.stop()
        
    model = YOLO(model_path)
    fsm = LabFSM()
    
    # Reset Report
    st.session_state.process_report = load_process_report(base_dir)
    
    # Output Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f'experiment_result_{timestamp}.json'
    report_path = os.path.join(base_dir, report_filename)
    output_dir = os.path.join(base_dir, 'videos', 'SOP_step')
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup Logging
    log_path = os.path.join(base_dir, 'experiment_log.txt')
    file_handler = logging.FileHandler(filename=log_path, mode='w', encoding='utf-8')
    logging.basicConfig(
        handlers=[file_handler],
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        force=True
    )

    # Variables
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    seg_count = 0
    current_segment_start = None
    last_saved_end_frame = 0
    saved_phases = set()
    total_processes = len(st.session_state.process_report)
    
    # Executors
    executor = get_executor()
    report_lock = get_report_lock()

    # Initial Segment Start
    if fsm.steps and fsm.steps[0].label == 'start':
        current_segment_start = 0
        logging.info(f"Segment Started at frame 0 (Step {fsm.steps[0].step_id})")
    
    frame_count = -1
    
    while cap.isOpened() and st.session_state.processing_active:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if frame_count % 2 != 0:
            continue
            
        # Inference
        results = model(frame, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if 0 <= cls_id < len(CLASSES):
                cls_name = CLASSES[cls_id]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                detections.append({'class': cls_name, 'id': cls_id, 'box': [x1, y1, x2, y2], 'conf': conf})
                
                # Draw box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, str(cls_id), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # FSM Update
        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        total_seconds = int(timestamp_ms / 1000)
        timestamp_str = f"{total_seconds // 60:02d}:{total_seconds % 60:02d}"
        
        prev_step_index = fsm.current_step_index
        fsm.update(detections, timestamp_str)
        
        # UI Update for Video
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Show Progress bar on frame
        if fsm.steps:
            progress = fsm.current_step_index / len(fsm.steps)
            bar_width = int(frame.shape[1] * progress)
            cv2.rectangle(frame_rgb, (0, frame_rgb.shape[0]-20), (bar_width, frame_rgb.shape[0]), (0, 255, 0), -1)

        video_placeholder.image(frame_rgb, channels="RGB", width="stretch")
        if total_frames > 0:
            progress_bar.progress(min(frame_count / total_frames, 1.0))
        status_text.text(f"Time: {timestamp_str} | Step: {fsm.current_step_index}/{len(fsm.steps)}")

        # Logic for Report & Segmentation (Synced with monitor-method.py)
        if fsm.current_step_index != prev_step_index:
            current_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            # 1. Mark completed step
            if fsm.current_step_index > 0:
                completed_step = fsm.steps[fsm.current_step_index - 1]
                update_step_status(st.session_state.process_report, completed_step.phase, completed_step.step_id, "yes")
            
            # 2. Jump Detection
            if fsm.current_step_index > prev_step_index + 1:
                skipped_indices = range(prev_step_index, fsm.current_step_index - 1)
                skipped_step_ids = [fsm.steps[i].step_id for i in skipped_indices]
                
                if skipped_indices:
                    logging.warning(f"Jump detected! Skipped steps: {skipped_step_ids}")
                    first_skipped = fsm.steps[skipped_indices[0]]
                    update_step_status(st.session_state.process_report, first_skipped.phase, None, "no", skipped_ids=skipped_step_ids)
                    
                    # Save error segment
                    error_start = last_saved_end_frame
                    error_end = current_frame_idx
                    
                    if error_end > error_start:
                        if first_skipped.phase not in saved_phases and seg_count < total_processes:
                            seg_count += 1
                            seg_name = f"SOP_step_{seg_count}"
                            out_path = os.path.join(output_dir, f"{seg_name}.mp4")
                            
                            saved_phases.add(first_skipped.phase)
                            
                            # Update Report Link
                            target_process = None
                            for process in st.session_state.process_report:
                                if process['process_name'] == first_skipped.phase:
                                    target_process = process
                                    process['video_link'] = out_path
                                    process['pass_check'] = False
                                    break
                            
                            current_step_key = f"step{seg_count:02d}"
                            
                            # Submit Background Task
                            executor.submit(process_segment_background, 
                                          video_path, out_path, error_start, error_end,
                                          target_process, current_step_key, True,
                                          report_path, st.session_state.process_report, report_lock)
                        
                        last_saved_end_frame = error_end

            # 3. Normal Segmentation (Start Label)
            start_check_idx = prev_step_index + 1
            end_check_idx = min(fsm.current_step_index + 1, len(fsm.steps))
            
            for i in range(start_check_idx, end_check_idx):
                step = fsm.steps[i]
                if step.label == 'start':
                    # End previous segment
                    if current_segment_start is not None:
                        end_frame = max(0, int(current_frame_idx))
                        if end_frame > current_segment_start:
                            prev_phase = fsm.steps[i-1].phase if i > 0 else None
                            
                            if prev_phase and prev_phase not in saved_phases and seg_count < total_processes:
                                seg_count += 1
                                seg_name = f"SOP_step_{seg_count}"
                                out_path = os.path.join(output_dir, f"{seg_name}.mp4")
                                
                                saved_phases.add(prev_phase)
                                
                                target_process = None
                                for process in st.session_state.process_report:
                                    if process['process_name'] == prev_phase:
                                        target_process = process
                                        process['video_link'] = out_path
                                        process['pass_check'] = True
                                        break
                                
                                current_step_key = f"step{seg_count:02d}"
                                executor.submit(process_segment_background,
                                               video_path, out_path, current_segment_start, end_frame,
                                               target_process, current_step_key, False,
                                               report_path, st.session_state.process_report, report_lock)
                            
                            last_saved_end_frame = end_frame
                    
                    # Start new segment
                    start_frame = max(0, int(current_frame_idx - fps/2))
                    current_segment_start = start_frame
                    logging.info(f"Segment Started at frame {start_frame} (Step {step.step_id})")

            # Force UI Refresh of Steps
            update_sidebar_ui()

    # End of Loop - Check open segment
    if current_segment_start is not None:
         end_frame = frame_count + 1
         
         if end_frame > current_segment_start:
             target_process = None
             # Find last executing or completed process
             for process in reversed(st.session_state.process_report):
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
                 
                 should_analyze = False
                 current_step_key = f"step{seg_count:02d}"
                 
                 if target_process:
                    target_process['video_link'] = out_path
                    saved_phases.add(target_process['process_name'])
                    
                    all_steps_passed = True
                    for step in target_process.get('steps', []):
                        if step.get('passed') != 'yes':
                            all_steps_passed = False
                            break
                    
                    target_process['pass_check'] = all_steps_passed
                    
                    if not all_steps_passed:
                        should_analyze = True
                 
                 executor.submit(process_segment_background,
                               video_path, out_path, current_segment_start, end_frame,
                               target_process, current_step_key, should_analyze,
                               report_path, st.session_state.process_report, report_lock)

    cap.release()
    # Note: We do not shutdown the executor here as it is cached in session
    
    with report_lock:
        save_process_report(st.session_state.process_report, report_path)
    
    st.session_state.processing_active = False
    st.success("Analysis Complete!")
