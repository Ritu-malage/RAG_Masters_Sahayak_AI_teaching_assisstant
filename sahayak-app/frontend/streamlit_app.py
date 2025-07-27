import streamlit as st
import requests
import pyrebase
import base64
from streamlit_geolocation import streamlit_geolocation
import uuid
import os
import json
import time
from pathlib import Path
from datetime import datetime, timezone
import io
from PIL import Image
import subprocess
import vertexai
import firebase_admin
from firebase_admin import credentials, storage as firebase_storage
from google.cloud import firestore, texttospeech
from vertexai.generative_models import GenerativeModel as VertexGenerativeModel
from vertexai.preview.vision_models import ImageGenerationModel

# --- CORE CONFIGURATION ---
FIREBASE_CONFIG = {
    "apiKey": "AIzaSyDqX19_B9CK6Uj_9muZwsXQBRlEWJ_aGYw",
    "authDomain": "vertex-ai-sahayak-project.firebaseapp.com",
    "projectId": "vertex-ai-sahayak-project",
    "storageBucket": "vertex-ai-sahayak-project.firebasestorage.app",
    "messagingSenderId": "36367633689",
    "appId": "1:36367633689:web:c40ab6f0fdebcf2e060988",
    "databaseURL": ""
}
PROJECT_ID = "vertex-ai-sahayak-project"
LOCATION = "us-central1"
BACKEND_URL = "http://127.0.0.1:6000"
TEMP_DIR = "temp_media"
HUGGINGFACE_API_KEY = "YOUR KEY"

# --- INITIALIZATION ---
os.makedirs(TEMP_DIR, exist_ok=True)
vertexai.init(project=PROJECT_ID, location=LOCATION)

try:
    if not firebase_admin._apps:
        cred = credentials.ApplicationDefault()
        firebase_admin.initialize_app(cred, {'storageBucket': FIREBASE_CONFIG["storageBucket"]})
    firebase = pyrebase.initialize_app(FIREBASE_CONFIG)
    auth = firebase.auth()
    storage = firebase.storage()
    db = firestore.Client(project=PROJECT_ID)
    st.session_state.firebase_initialized = True
except Exception as e:
    st.session_state.firebase_initialized = False
    st.error(f"Failed to initialize Firebase: {e}")

# Session State Initialization
for key, default_value in [
    ('user', None), ('page', 'Login'), ('messages', []),
    ('location', None), ('eval_result', None), ('canvas_result', {})
]:
    if key not in st.session_state:
        st.session_state[key] = default_value

# --- CONCEPT CANVAS AGENT FUNCTIONS ---
def cc_upload_file(local_path, storage_path, content_type):
    try:
        bucket = firebase_storage.bucket()
        blob = bucket.blob(storage_path)
        blob.upload_from_filename(local_path, content_type=content_type)
        blob.make_public()
        return blob.public_url
    except Exception as e:
        st.error(f"❌ Error uploading {local_path}: {e}")
        return None

def cc_save_metadata(media_type, query, file_info):
    try:
        doc_ref = db.collection('concept_canvas_generations').document()
        metadata = {'id': doc_ref.id, 'media_type': media_type, 'user_query': query, 'files': file_info, 'created_at': datetime.now(timezone.utc)}
        doc_ref.set(metadata)
        return doc_ref.id
    except Exception as e:
        st.error(f"❌ Error saving metadata: {e}")
        return None

def cc_get_history(limit=10):
    try:
        docs = db.collection('concept_canvas_generations').order_by('created_at', direction=firestore.Query.DESCENDING).limit(limit).stream()
        return [doc.to_dict() for doc in docs]
    except Exception as e:
        st.error(f"❌ Error retrieving history: {e}")
        return []

def cc_generate_image_vertex_ai(prompt, output_path):
    try:
        st.info("🎨 Trying Vertex AI for image generation...")
        model = ImageGenerationModel.from_pretrained("imagen-4.0-ultra-generate-preview-06-06")
        response = model.generate_images(prompt=prompt, number_of_images=1)
        if response and response.images:
            response.images[0].save(output_path)
            return True
        st.warning("Vertex AI returned no images.")
        return False
    except Exception as e:
        st.warning(f"⚠️ Vertex AI image generation failed: {e}")
        return False

def cc_generate_image_huggingface(prompt, output_path):
    if not HUGGINGFACE_API_KEY: return False
    try:
        st.info("🎨 Vertex AI failed. Trying Hugging Face fallback...")
        API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
        headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
        response = requests.post(API_URL, headers=headers, json={"inputs": prompt}, timeout=60)
        if response.status_code == 200:
            Image.open(io.BytesIO(response.content)).save(output_path)
            return True
        st.warning(f"⚠️ Hugging Face fallback failed (Status: {response.status_code}).")
        return False
    except Exception as e:
        st.warning(f"⚠️ Hugging Face fallback failed: {e}")
        return False

def cc_generate_single_image_with_fallback(prompt, output_path):
    if cc_generate_image_vertex_ai(prompt, output_path):
        return True
    if cc_generate_image_huggingface(prompt, output_path):
        return True
    st.error("❌ All image generation providers failed.")
    return False

def cc_generate_image(prompt, language):
    try:
        session_id = str(uuid.uuid4())
        full_prompt = f'Create a clear, simple illustration for "{prompt}". ALL text and labels MUST be in {language.upper()}.'
        temp_session_dir = os.path.join(TEMP_DIR, session_id)
        os.makedirs(temp_session_dir, exist_ok=True)
        local_path = os.path.join(temp_session_dir, "image.png")
        if not cc_generate_single_image_with_fallback(full_prompt, local_path):
            return None
        storage_path = f"concept_canvas/images/{session_id}.png"
        public_url = cc_upload_file(local_path, storage_path, "image/png")
        if public_url:
            file_info = [{'type': 'image', 'public_url': public_url}]
            metadata_id = cc_save_metadata('image', prompt, file_info)
            return {"url": public_url, "metadata_id": metadata_id}
        return None
    except Exception as e:
        st.error(f"❌ Image Generation Error: {e}")
        return None

def cc_generate_video(story_text):
    try:
        session_id = str(uuid.uuid4())
        model = VertexGenerativeModel("gemini-2.5-pro")
        scenes_prompt = f'Break this story into 5-7 scenes for a video. For each, give a "scene_description" and "narration_text". Return a JSON list. Story: """{story_text}"""'
        response = model.generate_content(scenes_prompt)
        scenes = json.loads(response.text.strip().replace("```json", "").replace("```", ""))
        if not scenes:
            st.error("Could not generate scenes from the story.")
            return None
        image_files, audio_files = [], []
        image_dir, audio_dir = os.path.join(TEMP_DIR, session_id, "images"), os.path.join(TEMP_DIR, session_id, "audio")
        os.makedirs(image_dir, exist_ok=True); os.makedirs(audio_dir, exist_ok=True)
        tts_client = texttospeech.TextToSpeechClient()
        for i, scene in enumerate(scenes):
            st.info(f"Generating scene {i + 1}/{len(scenes)}...")
            img_prompt = scene['scene_description'] + " | animated storybook illustration, colorful"
            img_path = os.path.join(image_dir, f"scene_{i}.png")
            if cc_generate_single_image_with_fallback(img_prompt, img_path):
                img_url = cc_upload_file(img_path, f"concept_canvas/videos/{session_id}/scene_{i}.png", "image/png")
                image_files.append({'path': img_path, 'url': img_url})
            else:
                st.warning(f"Skipping scene {i + 1} due to image generation failure.")
                continue
            narration = scene['narration_text']
            audio_path = os.path.join(audio_dir, f"scene_{i}.mp3")
            synthesis_input = texttospeech.SynthesisInput(text=narration)
            voice = texttospeech.VoiceSelectionParams(language_code="en-US", name="en-US-Wavenet-D")
            audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
            tts_response = tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
            with open(audio_path, "wb") as out: out.write(tts_response.audio_content)
            audio_url = cc_upload_file(audio_path, f"concept_canvas/videos/{session_id}/scene_{i}.mp3", "audio/mpeg")
            audio_files.append({'path': audio_path, 'url': audio_url})
        if not image_files or not audio_files or len(image_files) != len(audio_files):
            st.error("Mismatch in generated scene assets. Cannot assemble video.")
            return None
        st.info("Assembling video with FFmpeg...")
        final_video_path = os.path.join(TEMP_DIR, session_id, "final.mp4")
        concat_audio_path = os.path.join(audio_dir, "full_audio.mp3")
        audio_list_path = os.path.join(TEMP_DIR, session_id, "audio_list.txt")
        with open(audio_list_path, "w", encoding="utf-8") as f:
            for aud_file in audio_files: f.write(f"file '{os.path.abspath(aud_file['path'])}'\n")
        audio_concat_cmd = ["ffmpeg", "-f", "concat", "-safe", "0", "-i", audio_list_path, "-c:a", "aac", "-b:a", "192k", "-y", concat_audio_path]
        subprocess.run(audio_concat_cmd, check=True, capture_output=True, text=True)
        image_list_path = os.path.join(TEMP_DIR, session_id, "image_list.txt")
        with open(image_list_path, "w") as f:
            for i in range(len(image_files)):
                duration_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", audio_files[i]['path']]
                duration = float(subprocess.check_output(duration_cmd).strip())
                f.write(f"file '{os.path.abspath(image_files[i]['path'])}'\n"); f.write(f"duration {duration}\n")
        intermediate_video_path = os.path.join(TEMP_DIR, session_id, "intermediate.mp4")
        subprocess.run(["ffmpeg", "-f", "concat", "-safe", "0", "-i", image_list_path, "-vsync", "vfr", "-pix_fmt", "yuv420p", "-y", intermediate_video_path], check=True)
        subprocess.run(["ffmpeg", "-i", intermediate_video_path, "-i", concat_audio_path, "-c:v", "copy", "-c:a", "aac", "-shortest", "-y", final_video_path], check=True)
        video_url = cc_upload_file(final_video_path, f"concept_canvas/videos/{session_id}/final.mp4", "video/mp4")
        if video_url:
            metadata_id = cc_save_metadata('video', story_text, [{'type': 'video', 'public_url': video_url}])
            return {"url": video_url, "metadata_id": metadata_id}
        return None
    except Exception as e:
        st.error(f"❌ Video Generation Error: {e}")
        return None

# --- SAHAYAK AGENT AUTH & API CALLS ---
def get_auth_headers():
    if st.session_state.user:
        try:
            user = auth.refresh(st.session_state.user['refreshToken'])
            st.session_state.user['idToken'] = user['idToken']
            return {'Authorization': f'Bearer {st.session_state.user["idToken"]}'}
        except Exception:
            logout(); st.error("Session expired. Please log in again."); st.rerun()
    return None

def upload_file_to_firebase(file_object, user_id):
    if not file_object: return None
    try:
        id_token = st.session_state.user['idToken']
        path = f"uploads/{user_id}/{uuid.uuid4()}"
        storage.child(path).put(file_object, id_token)
        return storage.child(path).get_url(id_token)
    except Exception as e:
        st.error(f"Error uploading {file_object.name}: {e}")
        return None

def call_vidyamitra_agent(goal, file_url=None, coordinates=None, language=None):
    headers, payload = get_auth_headers(), {"goal": goal, "fileUrl": file_url, "coordinates": coordinates, "language": language}
    if not headers: return None
    try:
        response = requests.post(f"{BACKEND_URL}/vidyamitra-agent", json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e: st.error(f"Request Error: {e}"); return None

def call_smarteval_agent(user_goal, answer_key_url, student_sheet_url):
    headers, payload = get_auth_headers(), {"userGoal": user_goal, "answerKeySheetUrl": answer_key_url, "studentAnswerSheetUrl": student_sheet_url}
    if not headers: return None
    try:
        response = requests.post(f"{BACKEND_URL}/smarteval-agent", json=payload, headers=headers)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e: st.error(f"Request Error: {e}"); return None

# --- UI & PAGE DEFINITIONS ---
def show_login_signup_page():
    st.image("https://firebasestorage.googleapis.com/v0/b/vertex-ai-sahayak-project.firebasestorage.app/o/uploads%2FWhatsApp%20Image%202025-07-27%20at%209.16.52%20AM.jpeg?alt=media&token=d6794acb-d54a-46f9-9652-ee3848dbbb8c", width=300)
    st.title("Sahayak: AI Teaching Assistant")
    choice = st.radio("", ["Login", "Sign Up"], horizontal=True, label_visibility="collapsed")
    email = st.text_input("Email", placeholder="Enter your email")
    password = st.text_input("Password", type="password", placeholder="Enter your password")
    if choice == "Sign Up":
        if st.button("Create Account", use_container_width=True):
            if email and password:
                try: auth.create_user_with_email_and_password(email, password); st.success("Account created! Please login.")
                except Exception as e: st.error(f"Signup Failed: {e}")
    else:
        if st.button("Login", use_container_width=True, type="primary"):
            if email and password:
                try: st.session_state.user = auth.sign_in_with_email_and_password(email, password); st.session_state.page = "Teacher Homepage"; st.rerun()
                except Exception: st.error("Login Failed: Invalid credentials.")

def show_teacher_homepage():
    st.title("Sahayak Agent Hub")
    st.sidebar.button("Logout", on_click=lambda: logout(True), use_container_width=True)
    st.header("Available Agents")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📝 VidyaMitra", use_container_width=True): st.session_state.page = "VidyaMitra"; st.rerun()
        st.caption("AI-powered multimodal teaching assistant for lesson plans, worksheets, and explanations.")
    with col2:
        if st.button("🧠 SmartEval", use_container_width=True): st.session_state.page = "SmartEval"; st.rerun()
        st.caption("AI-powered assessment correction and feedback assistant for MCQ and descriptive answers.")
    with col3:
        if st.button("🎨 ConceptCanvas", use_container_width=True): st.session_state.page = "ConceptCanvas"; st.rerun()
        st.caption("AI-powered engine to generate visual learning aids like diagrams, flowcharts, and animated videos.")

def show_vidyamitra_page():
    st.title("VidyaMitra ✨")
    st.caption("Your multimodal teaching assistant")
    with st.sidebar:
        st.header("Contextual Tools 🛠️")
        st.info("Optionally add a file, language, or location to tailor the AI's response.")
        uploaded_file = st.file_uploader("Upload a File", type=['png', 'jpg', 'jpeg', 'pdf', 'mp4'])
        language_options = {"Default (English)": "en-IN", "Hindi": "hi-IN", "Tamil": "ta-IN"}
        language_code = language_options[st.selectbox("Select Language", options=list(language_options.keys()))]
        location_data = streamlit_geolocation()
        if location_data and location_data.get('latitude'):
            st.session_state.location = {'latitude': location_data['latitude'], 'longitude': location_data['longitude']}
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    if prompt := st.chat_input("Ask VidyaMitra..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.spinner("Processing..."):
            file_url = upload_file_to_firebase(uploaded_file, st.session_state.user['localId']) if uploaded_file else None
            response = call_vidyamitra_agent(prompt, file_url, st.session_state.get('location'), language_code)
            if response:
                text = response.get("result", response.get("passageText", "Sorry, an error occurred."))
                st.session_state.messages.append({"role": "assistant", "content": text})
        st.rerun()

def show_smarteval_page():
    st.title("SmartEval 🧠")
    st.caption("AI-powered Assessment Correction")
    with st.form("smarteval_form"):
        st.header("1. Evaluation Goal")
        goal = st.text_input("Describe the evaluation task", placeholder="e.g., Evaluate this MCQ test.")
        st.header("2. Upload Sheets")
        col1, col2 = st.columns(2)
        key_sheet = col1.file_uploader("Upload Answer Key", type=["png", "jpg"])
        student_sheet = col2.file_uploader("Upload Student Sheet", type=["png", "jpg"])
        if st.form_submit_button("✨ Evaluate Now", use_container_width=True, type="primary"):
            if all([goal, key_sheet, student_sheet]):
                with st.spinner("Uploading and evaluating..."):
                    user_id = st.session_state.user['localId']
                    key_url = upload_file_to_firebase(key_sheet, user_id)
                    student_url = upload_file_to_firebase(student_sheet, user_id)
                    if key_url and student_url:
                        st.session_state.eval_result = call_smarteval_agent(goal, key_url, student_url)
            else:
                st.warning("Please provide the goal and upload both sheets.")
    if st.session_state.eval_result:
        st.divider()
        st.header("Evaluation Report")
        st.markdown(st.session_state.eval_result)

def show_conceptcanvas_page():
    st.title("ConceptCanvas 🎨")
    st.caption("Generate visual learning aids like diagrams and animated videos.")
    img_tab, vid_tab, hist_tab = st.tabs(["🖼️ Image Generation", "🎬 Video Generation", "📜 History"])
    with img_tab:
        st.subheader("Generate a Diagram or Illustration")
        with st.form("image_form"):
            img_prompt = st.text_area("Enter a concept to visualize:", height=150, placeholder="e.g., The water cycle with labels for evaporation, condensation, and precipitation.")
            lang = "English"
            if st.form_submit_button("Generate Image", type="primary"):
                if img_prompt:
                    with st.spinner("🎨 Creating your image... this may take a moment."):
                        st.session_state.canvas_result['image'] = cc_generate_image(img_prompt, lang)
        if st.session_state.canvas_result.get('image'):
            st.success(f"Image generated! Metadata ID: {st.session_state.canvas_result['image']['metadata_id']}")
            st.image(st.session_state.canvas_result['image']['url'], caption="Generated Image")
    with vid_tab:
        st.subheader("Generate an Animated Video from a Story")
        with st.form("video_form"):
            video_prompt = st.text_area("Enter a short story to animate:", height=200, placeholder="e.g., A thirsty crow sees a pitcher with water at the bottom...")
            if st.form_submit_button("Generate Video", type="primary"):
                if video_prompt:
                    with st.spinner("🎬 Generating your video... this can take several minutes."):
                        st.session_state.canvas_result['video'] = cc_generate_video(video_prompt)
        if st.session_state.canvas_result.get('video'):
            st.success(f"Video generated! Metadata ID: {st.session_state.canvas_result['video']['metadata_id']}")
            st.video(st.session_state.canvas_result['video']['url'])
    with hist_tab:
        st.subheader("Recent Generations")
        if st.button("Refresh History"):
            st.session_state.canvas_result['history'] = cc_get_history()
        if st.session_state.canvas_result.get('history'):
            for item in st.session_state.canvas_result['history']:
                with st.expander(f"{item['media_type'].upper()} - {item['created_at'].strftime('%Y-%m-%d %H:%M')}"):
                    st.write(f"**Query:** {item['user_query']}")
                    if item.get('files'): st.link_button("View Asset", item['files'][-1]['public_url'])

# --- MAIN APP ROUTER ---
def logout(manual=False):
    for key in list(st.session_state.keys()):
        if key not in ['firebase_initialized']: del st.session_state[key]
    st.session_state.page = 'Login'
    if manual: st.rerun()

if st.session_state.get('firebase_initialized', False):
    if not st.session_state.get('user'): st.session_state.page = 'Login'
    page = st.session_state.get('page', 'Login')
    if page != 'Login':
        if st.sidebar.button("← Agent Hub", use_container_width=True):
            st.session_state.page = "Teacher Homepage"; st.rerun()
    if page == 'Login': show_login_signup_page()
    elif page == 'Teacher Homepage': show_teacher_homepage()
    elif page == 'VidyaMitra': show_vidyamitra_page()
    elif page == 'SmartEval': show_smarteval_page()
    elif page == 'ConceptCanvas': show_conceptcanvas_page()
else:
    st.error("Firebase initialization failed. The application cannot start.")