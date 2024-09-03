
import cv2
import torch
import speech_recognition as sr
from transformers import BertTokenizer, BertModel, GPT2LMHeadModel, GPT2Tokenizer, pipeline
# code belongs to se20uari052

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)

# Initialize of BERT
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Initialize of GPT-2
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')

# Video Preprocessing: Extract key frames from the video more efficiently
def extract_key_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    prev_frame = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to grayscale and resize for faster processing
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.resize(gray_frame, (500, 280))
        
        # Calculate frame difference
        if prev_frame is not None:
            frame_diff = cv2.absdiff(prev_frame, gray_frame)
           
            if cv2.mean(frame_diff)[0] > 10:
                frames.append(frame)
        
        prev_frame = gray_frame
        
    cap.release()
    return frames

# Enhanced Object Detection and Action Recognition
def detect_objects_and_actions(frame, prev_objects):
    results = model(frame)
    detections = results.pandas().xyxy[0] 
    current_objects = {}

    narrative = ""  
    for _, row in detections.iterrows():
     
        obj_name = row['name']
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        center = ((x1+x2)//2, (y1+y2)//2)

        action = 'is static'
        if obj_name in prev_objects:
            prev_center = prev_objects[obj_name]
          
            if center[0] < prev_center[0]:
                action = 'is moving left'
            elif center[0] > prev_center[0]:
                action = 'is moving right'
            if center[1] < prev_center[1]:
                action += ' and up'
            elif center[1] > prev_center[1]:
                action += ' and down'

        current_objects[obj_name] = center 
        narrative += f"The {obj_name} {action}. "

    return current_objects, narrative


def transcribe_audio(audio_file):
    r = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = r.record(source)
        try:
            text = r.recognize_google(audio_data)
        except sr.UnknownValueError:
            text = "[Unintelligible audio]"
        except sr.RequestError as e:
            text = f"[Request failed: {e}]"
    return text

def summarize_audio_transcription(text):
    try:
        if gpt2_tokenizer.pad_token is None:
            gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

        inputs = gpt2_tokenizer.encode_plus(
            text, 
            max_length=512, 
            truncation=True, 
            return_tensors="pt", 
            padding="max_length"
        )

        outputs = gpt2_model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=150,  
            num_return_sequences=1,
            pad_token_id=gpt2_tokenizer.pad_token_id
        )

        summary = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        return f"[Audio summarization failed: {e}]"


def generate_story(key_frame_narrative, audio_summary):
    combined_narrative = key_frame_narrative + " " + audio_summary
    try:
        if gpt2_tokenizer.pad_token is None:
            gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

        inputs = gpt2_tokenizer.encode_plus(
            combined_narrative, 
            max_length=512,  
            truncation=True, 
            return_tensors="pt", 
            padding="max_length"
        )

        outputs = gpt2_model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=150, 
            num_return_sequences=1,
            pad_token_id=gpt2_tokenizer.pad_token_id
        )

        story = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return story
    except Exception as e:
        return f"[Story generation failed: {e}]"


def video_summarization_system(video_path, audio_path):
    try:
        print("Extracting key frames from video...")
        frames = extract_key_frames(video_path)

        key_frame_narrative = ""
        prev_objects = {}  
        print("Detecting objects and actions in frames...")
        for i, frame in enumerate(frames):
            current_objects, frame_narrative = detect_objects_and_actions(frame, prev_objects)
            key_frame_narrative += f"In scene {i+1}, {frame_narrative} "
            prev_objects = current_objects  
            
        print("Transcribing audio...")
        transcribed_text = transcribe_audio(audio_path)

        print("Summarizing audio transcription...")
        audio_summary = summarize_audio_transcription(transcribed_text)
        print("\nAudio Summary:", audio_summary)

        print("Generating story from key frames and audio summary...")
        final_story = generate_story(key_frame_narrative, audio_summary)

        return {
            "key_frames": key_frame_narrative,
            "audio_summary": audio_summary,
            "final_story": final_story
        }
    except Exception as e:
        return f"Failed to summarize video: {e}"


video_path = './video_csm.mp4'  
audio_path = './audio.wav' 


results = video_summarization_system(video_path, audio_path)
print("\nKey Frame Descriptions:", results["key_frames"])
print("\nAudio Summary:", results["audio_summary"])
print("\nCombined Story:", results["final_story"])





