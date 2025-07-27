import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, auth, firestore
import vertexai
from vertexai.generative_models import (
    GenerativeModel,
    Tool,
    Part,
    FunctionDeclaration,
    Image,
)
from google.cloud import texttospeech, speech
import requests
import base64
from datetime import datetime
import cv2
import string
import numpy as np
import json

# --- Initialization ---
app = Flask(__name__)
CORS(app)

# --- Firebase & Vertex AI Initialization ---
try:
    cred = credentials.ApplicationDefault()
    # IMPORTANT: Replace with your actual GCP Project ID
    PROJECT_ID = "vertex-ai-sahayak-project"
    firebase_admin.initialize_app(cred, {"projectId": PROJECT_ID})
    LOCATION = "us-central1"
    vertexai.init(project=PROJECT_ID, location=LOCATION)

    # Initialize Firestore DB and API clients
    db = firestore.client()
    tts_client = texttospeech.TextToSpeechClient()
    speech_client = speech.SpeechClient()

    print("Firebase and Vertex AI initialized successfully.")
except Exception as e:
    print(f"Initialization failed: {e}")


# --- Tool Definitions (Functions the AI can call) ---


def create_lesson_plan(topic: str, grade_level: str, duration_minutes: int) -> str:
    """Generates a structured lesson plan for a given topic, grade, and duration."""
    model = GenerativeModel("gemini-2.5-pro")
    prompt = f"Create a detailed, structured lesson plan for a {duration_minutes}-minute class for {grade_level} students on the topic of '{topic}'. Include sections for: Learning Objectives, Materials, Introduction, Main Activity, and Assessment."
    response = model.generate_content(prompt)
    return response.text


def create_worksheets(topic: str, grade_level: str, num_questions: int) -> str:
    """Creates a worksheet with questions for a specific topic and grade level."""
    model = GenerativeModel("gemini-2.5-pro")
    prompt = f"Generate a worksheet with {num_questions} questions for {grade_level} students on the topic '{topic}'. Include a mix of multiple-choice and short-answer questions. Provide an answer key at the end."
    response = model.generate_content(prompt)
    return response.text


def simplify_concept(concept: str, grade_level: str) -> str:
    """Explains a complex concept in simple terms for a specific grade level, using analogies relevant to a rural Indian context."""
    model = GenerativeModel("gemini-2.5-pro")
    prompt = f"Explain the concept of '{concept}' in simple terms suitable for {grade_level} students. Use helpful analogies from a rural Indian context (e.g., farming, festivals, local life) to make it relatable."
    response = model.generate_content(prompt)
    return response.text


def generate_hyperlocal_explanation(
    concept: str, grade_level: str, coordinates: dict, language: str
) -> str:
    """
    Explains a complex concept in simple terms, using analogies relevant to the user's geographic coordinates.
    Args:
        concept (str): The concept to explain.
        grade_level (str): The target grade level for the explanation.
        coordinates (dict): A dictionary containing the user's 'latitude' and 'longitude'.
        language (str): The language for the response.
    """
    model = GenerativeModel("gemini-2.5-pro")

    lat = coordinates.get("latitude")
    lon = coordinates.get("longitude")

    prompt = f"""
    Task: Explain the concept of '{concept}' in simple terms suitable for a {grade_level} student.
    Language: The entire response must be in the {language} language. Or if the value of language is not mentioned then infer the primary local language spoken in this region. If that also is not possible then let the default language be English(India)
    Hyperlocal Context: The user is located at latitude {lat} and longitude {lon}. First, infer the likely region, state, and cultural context (e.g., rural, urban, coastal, farming region) from these coordinates. Then, to make the explanation relatable, you MUST use helpful analogies, stories, or examples from that specific inferred context.
    Structure: Start with the explanation and then clearly label a section for the hyperlocal analogy.
    """
    response = model.generate_content(prompt)
    return response.text


def analyze_image_content(image_url: str, prompt: str) -> str:
    """Analyzes an image from a URL and answers a question or follows an instruction about it."""
    model = GenerativeModel("gemini-2.5-pro")
    try:
        image_response = requests.get(image_url)
        image_response.raise_for_status()
        image_part = Part.from_data(image_response.content, mime_type="image/jpeg")
        instructional_prompt = f"""
                You must answer the user's query based ONLY on the content of the provided image. 
                Do not use any external knowledge. If the answer cannot be found in the image, 
                you must state that the information is not present in the image.

                User's query: "{prompt}"
                """
        full_prompt = [instructional_prompt, image_part]
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"Error processing image: {e}"


def analyze_pdf_content(pdf_url: str, prompt: str) -> str:
    """Downloads a PDF from a URL and performs a task based on its content and the user's prompt (e.g., summarize, answer questions)."""
    # Use the powerful multimodal model for direct PDF analysis
    model = GenerativeModel("gemini-2.5-pro")
    try:
        # Download the PDF content
        pdf_response = requests.get(pdf_url)
        pdf_response.raise_for_status()

        # Create a Part object directly from the PDF bytes
        pdf_part = Part.from_data(
            data=pdf_response.content, mime_type="application/pdf"
        )

        instructional_prompt = f"""
                You must answer the user's query based ONLY on the content of the provided PDF document. 
                Do not use any external knowledge. If the answer cannot be found in the PDF, 
                you must state that the information is not present in the document.

                User's query: "{prompt}"
                """

        # Send the prompt and the PDF Part to the model
        full_prompt = [instructional_prompt, pdf_part]
        response = model.generate_content(full_prompt)
        return response.text

    except Exception as e:
        return f"Error processing PDF: {e}"


def analyze_video_content(video_url: str, prompt: str) -> str:
    """Downloads a video from a URL and performs a task based on its content and the user's prompt (e.g., summarize, answer questions)."""
    model = GenerativeModel("gemini-2.5-pro")
    try:
        # Download the video content
        video_response = requests.get(video_url)
        video_response.raise_for_status()

        # Create a Part object directly from the video bytes.
        # The model can handle common video formats like MP4.
        video_part = Part.from_data(data=video_response.content, mime_type="video/mp4")

        instructional_prompt = f"""
                You must answer the user's query based ONLY on the content of the provided video. 
                Do not use any external knowledge. If the answer cannot be found in the video, 
                you must state that the information is not present in the video.

                User's query: "{prompt}"
                """

        # Send the prompt and the video Part to the model
        full_prompt = [instructional_prompt, video_part]
        response = model.generate_content(full_prompt)
        return response.text

    except Exception as e:
        return f"Error processing video: {e}"


def generate_audio_reading_assessment(
    topic: str, grade_level: str, language_code: str = "en-IN"
) -> str:
    """Generates a text passage for a reading assessment and converts it to speech. Returns a base64 encoded audio string."""
    model = GenerativeModel("gemini-2.5-pro")
    prompt = f"Create a short, simple paragraph (about 3-4 sentences) on the topic of '{topic}' suitable for a {grade_level} student's reading assessment."
    passage_response = model.generate_content(prompt)
    passage_text = passage_response.text

    synthesis_input = texttospeech.SynthesisInput(text=passage_text)
    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code, ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = tts_client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    # Combine text and audio in the response. Use a clear separator.
    audio_base64 = base64.b64encode(response.audio_content).decode("utf-8")
    return f"PASSAGE_TEXT_START\n{passage_text}\nPASSAGE_TEXT_END\nAUDIO_BASE64_START\n{audio_base64}\nAUDIO_BASE64_END"


# ----------SMART_EVAL AGENT FUNCTIONS-------------
def get_usn_name_from_image(image):
    """Given an image it reads the Name, and USN from the image"""

    # 1. Define the schema for the model's output
    student_info_extractor = Tool(
        function_declarations=[
            FunctionDeclaration(
                name="extract_student_info",
                description="Extracts the student's name and USN from an image.",
                # The 'parameters' define the exact JSON output structure
                parameters={
                    "type": "object",
                    "properties": {
                        "USN": {
                            "type": "string",
                            "description": "The Unique Serial Number (USN) of the student.",
                        },
                        "student_name": {
                            "type": "string",
                            "description": "The full name of the student.",
                        },
                    },
                    "required": ["USN", "student_name"],
                },
            )
        ]
    )

    multimodal_model = GenerativeModel("gemini-2.5-pro", tools=[student_info_extractor])

    prompt = (
        "Please extract the student's name and USN from the provided ID card image."
    )

    response = multimodal_model.generate_content([image, prompt])

    try:
        function_call = response.candidates[0].content.parts[0].function_call
        if function_call.name == "extract_student_info":
            extracted_data = {
                "USN": function_call.args.get("USN"),
                "student_name": function_call.args.get("student_name"),
            }

            return extracted_data

    except Exception as e:
        raise Exception(e)


def find_rectangle_contours(contours, image):
    rectangle_contours = []
    corner_points = []
    for i in contours:
        contour_area = cv2.contourArea(i)

        if contour_area > 500:
            # Total length of contour
            perimeter = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * perimeter, True)
            # When the length is 4 its implies its a rectangle
            if len(approx) == 4:
                corner_points.append(approx)
                rectangle_contours.append(i)

    # Sorts the rectangle countours such that the biggest rectangle contour is 1
    return sorted(rectangle_contours, key=cv2.contourArea, reverse=True), corner_points


def reorder_points(pts):
    """
    To ensure that the boundary box points are in the order of top-left, top-right, bottom right, bottom-left
    """
    pts = pts.reshape((4, 2))
    pts_new = np.zeros((4, 1, 2), np.int32)

    add = pts.sum(1)

    pts_new[0] = pts[np.argmin(add)]  # top-left
    pts_new[3] = pts[np.argmax(add)]  # bottom-left

    diff = np.diff(pts, axis=1)

    pts_new[1] = pts[np.argmin(diff)]  # top-right
    pts_new[2] = pts[np.argmax(diff)]  # bottom-right

    return pts_new


def get_grids(image, no_of_rows, no_of_columns):
    """
    To split the entire OMR into square grids
    """
    grids = []

    # Find vertical rows
    rows = np.vsplit(image, no_of_rows)  # Cuts the image into the no of rows mentioned

    # Iterating through each column
    for r in rows:
        row_wise_grids = []

        # Column wise split
        cols = np.hsplit(r, no_of_columns)
        # Iterating through each cell of the OMR grid
        for choice_grid in cols:
            row_wise_grids.append(choice_grid)

        grids.append(row_wise_grids)

    return grids


def get_region_of_interest_image(bounding_box, image, height, width):
    """
    Generates an image with only the region of interest
    """
    pt1 = np.float32(bounding_box)
    # Bounding box for a new resized image
    pt2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    # To change the perspective of the MCQ OMR section and removing the unnecessary stuff in the image
    matrix = cv2.getPerspectiveTransform(pt1, pt2)
    region_of_interest = cv2.warpPerspective(image, matrix, (width, height))

    return region_of_interest


def get_mcq_answers(
    image_response,
    width: int = 700,
    height: int = 700,
    no_of_questions: int = 5,
    no_of_options: int = 5,
) -> dict:
    """Given an OMR sheet it returns the answers marked"""
    # image = cv2.imread(image_path)

    # 2. Convert the raw bytes into a NumPy array
    image_array = np.frombuffer(image_response.content, np.uint8)

    # 3. Decode the NumPy array into an OpenCV image
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    resized_image = cv2.resize(image, (width, height))
    # contour_image = resized_image.copy()

    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # Glaussian blur to reduce the noise in the image
    blur_image = cv2.GaussianBlur(gray_image, (5, 5), 1)

    # Edge detection
    canny_image = cv2.Canny(blur_image, 130, 130)

    # Countour detections
    # External : Outer edge detction,
    contours, hierarchy = cv2.findContours(
        canny_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Draws all the contours seen
    # cv2.drawContours(contour_image, contours, -1, (0,255,0), 2)

    rectangle_contours, bounding_box = find_rectangle_contours(contours, resized_image)

    # Biggest rectangle
    mcq_section = rectangle_contours[0]
    # The bounding points are not reordered
    mcq_section_bounding_box = bounding_box[0]

    # # Draws the MCQ section contour box
    # cv2.drawContours(contour_image, mcq_section_bounding_box, -1, (0,255,0), 5)

    # # Draws the USN section contour box
    # cv2.drawContours(contour_image, usn_section_bounding_box, -1, (0,255,0), 5)

    # The returned value will intially not be in order, it must be in the order of top left,
    mcq_section_bounding_box = reorder_points(mcq_section_bounding_box)
    # usn_section_bounding_box = reorder_points(usn_section_bounding_box)

    # Birds eye view of the image section where there is OMR
    mcq_omr = get_region_of_interest_image(
        bounding_box=mcq_section_bounding_box,
        image=resized_image,
        height=height,
        width=width,
    )

    # Thresholding - The background will be black and edges will be white, the region where white pixels are more will be the answer marked
    # If you want to make the partially shaded also visible increase the 2nd parameters value
    mcq_thresholded_image = cv2.threshold(
        cv2.cvtColor(mcq_omr, cv2.COLOR_BGR2GRAY), 150, 255, cv2.THRESH_BINARY_INV
    )[1]

    # Get each bubbles we need to break it into grids
    grids = get_grids(
        image=mcq_thresholded_image,
        no_of_rows=no_of_questions,
        no_of_columns=no_of_options,
    )

    # Initializing the answers will None
    answers = {i + 1: "" for i in range(0, no_of_questions)}
    options = [alphabet for alphabet in string.ascii_uppercase[:no_of_options]]

    # Iterating through each cell of the OMR, and then finding the options marked
    for question_no in range(0, len(grids)):
        total_white_pixels = []
        for option in range(0, len(grids[question_no])):
            total_white_pixels.append(cv2.countNonZero(grids[question_no][option]))
        # Validating if the shade is acceptable
        if max(total_white_pixels) >= 4000:  # its considered as answered
            answers[question_no + 1] = options[
                total_white_pixels.index(max(total_white_pixels))
            ]
    return answers


def get_information_from_answer_paper(image):
    """
    Given an image of an answer paper, extracts student details, questions,
    answers, and marks in a structured format.

    Args:
        image: The image file.

    Returns:
        A dictionary containing the structured data from the answer paper.
    """

    # 1. Define the tool with a nested schema for the answer paper
    answer_paper_extractor = Tool(
        function_declarations=[
            FunctionDeclaration(
                name="get_information_from_answer_paper",
                description="Extracts student info and a list of questions, answers, and marks from an answer paper.",
                parameters={
                    "type": "object",
                    "properties": {
                        "questions": {
                            "type": "array",
                            "description": "A list of all questions, their corresponding answers, and marks.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "question_number": {
                                        "type": "string",
                                        "description": "The number of the question (e.g., '1a', '2').",
                                    },
                                    "answer": {
                                        "type": "string",
                                        "description": "The complete answer written by the student for the question.",
                                    },
                                    "marks_awarded": {
                                        "type": "number",
                                        "description": "The marks awarded for the answer.",
                                    },
                                },
                                "required": [
                                    "question_number",
                                    "answer",
                                    "marks_awarded",
                                ],
                            },
                        }
                    },
                    "required": ["questions"],
                },
            )
        ]
    )

    # 2. Initialize the model and specify the tool
    model = GenerativeModel("gemini-2.5-pro", tools=[answer_paper_extractor])

    # 3. Create a prompt focused on the task
    prompt = "Analyze the attached answer paper. Extract the student's name, USN, and for each question, extract the question number, the student's answer, and the marks awarded. Populate all the details in the provided tool."

    # 4. Generate content
    response = model.generate_content([image, prompt])

    # 5. Extract the structured JSON data from the function call
    try:
        function_call = response.candidates[0].content.parts[0].function_call
        if function_call.name == "get_information_from_answer_paper":
            # The .args attribute directly holds the dictionary
            return dict(function_call.args)
    except (IndexError, AttributeError) as e:
        return None


# --- Agent Configurations ---
vidyamitra_tools = Tool(
    function_declarations=[
        FunctionDeclaration.from_func(create_lesson_plan),
        FunctionDeclaration.from_func(create_worksheets),
        FunctionDeclaration.from_func(simplify_concept),
        FunctionDeclaration.from_func(generate_hyperlocal_explanation),
        FunctionDeclaration.from_func(analyze_image_content),
        FunctionDeclaration.from_func(analyze_pdf_content),
        FunctionDeclaration.from_func(analyze_video_content),
        FunctionDeclaration.from_func(generate_audio_reading_assessment),
    ]
)
vidyamitra_model = GenerativeModel("gemini-2.5-pro", tools=[vidyamitra_tools])
# VIDYAMITRA_SYSTEM_PROMPT = """You are VidyaMitra, an expert AI teaching assistant for India. Your goal is to help teachers by creating educational materials. You must use the provided tools to fulfill the user's request. If an image is provided, use the analyze_image_content tool. When asked for a reading assessment, use the audio generation tool. Always respond in a helpful, encouraging tone and format your text responses in clear markdown."""
# VIDYAMITRA_SYSTEM_PROMPT = """You are VidyaMitra, an expert AI teaching assistant for India. Your goal is to help teachers by creating educational materials. You must use the provided tools to fulfill the user's request. If an image is provided, use 'analyze_image_content'. If a PDF is mentioned, use 'analyze_pdf_content'. If a video is mentioned, use 'analyze_video_content'. When asked for a reading assessment, use the audio generation tool. Always respond in a helpful, encouraging tone and format your text responses in clear markdown."""
VIDYAMITRA_SYSTEM_PROMPT = """You are VidyaMitra, an expert AI teaching assistant for India. Your goal is to help teachers by creating educational materials. You must use the provided tools to fulfill the user's request. When the user asks to 'explain', 'simplify', or wants an analogy for a concept, you MUST use the 'generate_hyperlocal_explanation' tool, using the provided coordinates. For other tasks like creating lesson plans or worksheets, use the appropriate tools. If an image is provided, use 'analyze_image_content'. If a PDF is mentioned, use 'analyze_pdf_content'. If a video is mentioned, use 'analyze_video_content'. When asked for a reading assessment, use the audio generation tool. Always respond in a helpful, encouraging tone and format your text responses in clear markdown."""
PLANNER_SYSTEM_PROMPT = """
You are a task routing agent. Your job is to analyze the user's request and select the appropriate workflow.
Based on the user's goal, you must return ONLY ONE of the following workflow names:

- "MCQ_EVALUATION": Choose this only if an MCQ has to be evaluated.
- "DESCRIPTIVE_EVALUATION": Choose this only if the user wants to evaluate long-form text, essays, or descriptive answers only.
- "UNKNOWN": If the user's request doesn't match any known workflow.

Do not explain your reasoning. Only return the single workflow name.
"""

# A dictionary to hold your defined workflows
WORKFLOWS = {
    "MCQ_EVALUATION": ["get_usn_name_from_image", "get_mcq_answers", "get_mcq_report"],
    "DESCRIPTIVE_EVALUATION": [
        "get_usn_name_from_image",
        "get_information_from_answer_paper",
        "evaluate_and_generate_feedback_with_gemini",
        "create_mark_down_report",
    ],
    "UNKNOWN": ["get_generic_answer"],
}


def evaluate_question(student_answer: str, correct_answer: str) -> dict:
    """
    Evaluates a single student answer against a correct answer using Gemini's
    native function calling for a structured response.

    Args:
        student_answer: The answer provided by the student.
        correct_answer: The correct answer from the answer key.

    Returns:
        A dictionary containing the evaluation status and reasoning.
    """
    # 1. Define the tool schema for the evaluation output
    evaluation_tool = Tool(
        function_declarations=[
            FunctionDeclaration(
                name="grade_answer",
                description="Grades the student's answer and provides a status and reasoning for the grade.",
                parameters={
                    "type": "object",
                    "properties": {
                        "status": {
                            "type": "string",
                            "description": "The evaluation status of the answer.",
                            # Using enum makes the model's output more reliable
                            "enum": [
                                "✅ Correct",
                                "⚠️ Partially Correct",
                                "❌ Incorrect",
                            ],
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "A brief, one-sentence explanation for the evaluation.",
                        },
                    },
                    "required": ["status", "reasoning"],
                },
            )
        ]
    )

    # 2. Initialize the model with the specific tool
    # Using gemini-1.5-pro for better reasoning capabilities
    model = GenerativeModel("gemini-2.5-pro", tools=[evaluation_tool])

    # 3. Create the prompt instructing the model to use the tool
    prompt = f"""
    You are an expert AI teaching assistant. Your task is to evaluate the student's answer against the provided answer key.

    **INPUT:**
    - **Answer Key:** "{correct_answer}"
    - **Student's Answer:** "{student_answer}"

    **TASK:**
    Based on the input, call the `grade_answer` function to record your evaluation.
    """

    # 4. Generate content and extract the function call arguments
    try:
        response = model.generate_content(prompt)
        function_call = response.candidates[0].content.parts[0].function_call

        if function_call.name == "grade_answer":
            # The .args attribute directly holds the structured dictionary
            return dict(function_call.args)
        else:
            # Fallback if the model calls an unexpected function
            return {
                "status": "❓ Error",
                "reasoning": "Model returned an unexpected tool call.",
            }

    except (IndexError, AttributeError, ValueError) as e:
        # Fallback for any other API or parsing errors
        return {
            "status": "❓ Error",
            "reasoning": f"Could not evaluate answer due to an API error: {e}",
        }


def evaluate_and_generate_feedback_with_gemini(agent_state):
    """
    Orchestrates the AI-powered evaluation for an entire answer sheet
    and generates a complete feedback report.
    """
    student_info = agent_state.get("student_answer_sheet_info", {})
    key_info = agent_state.get("answer_key_sheet_info", {})

    answer_key_map = {q["question_number"]: q for q in key_info.get("questions", [])}

    evaluation_results = []
    student_total_score = 0.0
    max_total_score = 0.0

    for student_q in student_info.get("questions", []):
        q_num = student_q["question_number"]
        key_q = answer_key_map.get(q_num)

        achieved_marks = 0.0
        max_marks = float(key_q.get("marks_awarded", 0)) if key_q else 0.0

        if key_q:
            # Call the modular function to get the evaluation component
            score_comp = evaluate_question(student_q["answer"], key_q["answer"])

            # Calculate achieved marks based on the structured AI response
            status = score_comp.get("status")
            if status == "✅ Correct":
                achieved_marks = max_marks
            elif status == "⚠️ Partially Correct":
                achieved_marks = max_marks * 0.5  # Award 50% for partial credit
            # For "Incorrect" or "Error", achieved_marks remains 0.0
        else:
            score_comp = {
                "status": "Not Evaluated",
                "reasoning": "This question was not found in the answer key.",
            }

        student_total_score += achieved_marks
        max_total_score += max_marks

        evaluation_results.append(
            {
                "question_number": q_num,
                "student_answer": student_q["answer"],
                "correct_answer": key_q.get("answer", "N/A") if key_q else "N/A",
                "achieved_marks": achieved_marks,
                "max_marks": max_marks,
                "score_component": score_comp,
            }
        )

    percentage = (
        (student_total_score / max_total_score * 100) if max_total_score > 0 else 0
    )
    student_name = agent_state.get("student_name", "Student")

    # Overall feedback logic (can be customized further)
    if percentage == 100:
        student_feedback = f"Excellent work, {student_name}! 🌟 You've demonstrated a perfect understanding of all the concepts."
        teacher_suggestion = "The student has a complete grasp of the material. No immediate intervention is needed."
    elif percentage >= 80:
        student_feedback = f"Great job, {student_name}! 👍 You have a strong understanding. Review the partially correct answers to aim for perfection."
        teacher_suggestion = "The student understands the material well. A quick review of partially correct answers would be beneficial."
    else:
        student_feedback = f"Good effort, {student_name}. You have a foundational understanding. Let's focus on the areas marked incorrect to build your confidence."
        teacher_suggestion = "The student is struggling with some key concepts. Personal attention is recommended."

    a = {
        "student_info": {"name": student_name, "usn": agent_state.get("USN", "N/A")},
        "summary_score": {
            "student_total": student_total_score,
            "max_total": max_total_score,
            "percentage": f"{percentage:.2f}%",
        },
        "evaluation_results": evaluation_results,
        "overall_feedback": {
            "for_student": student_feedback,
            "for_teacher": teacher_suggestion,
        },
    }

    # Final structured output
    return {
        "summary_score": {
            "student_total": student_total_score,
            "max_total": max_total_score,
            "percentage": f"{percentage:.2f}%",
        },
        "evaluation_results": evaluation_results,
        "overall_feedback": {
            "for_student": student_feedback,
            "for_teacher": teacher_suggestion,
        },
    }


def create_mark_down_report(report_data):
    """
    Takes the final report data and uses an LLM to generate a structured
    Markdown file for frontend display.

    Args:
        report_data: The final dictionary containing all evaluation details.

    Returns:
        A string containing the complete Markdown report.
    """

    keys_to_exclude = [
        "answer_key_sheet_image_response",
        "student_answer_sheet_image_response",
        "answer_key_sheet",
        "students_answer_sheet",
    ]

    serializable_report_data = {
        key: value for key, value in report_data.items() if key not in keys_to_exclude
    }

    # Convert the report data dictionary to a JSON string for the prompt
    report_json_string = json.dumps(serializable_report_data, indent=2)

    # Initialize the model
    # vertexai.init(project="your-gcp-project-id", location="your-gcp-location")
    model = GenerativeModel("gemini-2.5-pro")

    # Create a detailed prompt instructing the model to generate Markdown
    prompt = f"""
    You are an expert report generator. Your task is to convert the following JSON data into a clear, well-structured, and encouraging Markdown report for a student.

    **JSON Data:**
    ```json
    {report_json_string}
    ```

    **Markdown Output Instructions:**
    1.  **Main Title:** Start with a main title: `# Performance Report`.
    2.  **Student Summary:** Display the student's name and USN clearly.
    3.  **Overall Performance:**
        - Show the final score with a title like `## 🏆 Overall Performance`.
        - Display the total score, max marks, and percentage in bold.
        - Include the personalized `for_student` feedback directly under the score.
    4.  **Detailed Breakdown:**
        - Create a section titled `## Detailed Breakdown`.
        - Generate a Markdown table with the following columns: `Question`, `Your Answer`, `Correct Answer`, `Evaluation`, and `Marks`.
        - For the "Evaluation" column, combine the `status` emoji with the `reasoning` text.
        - For the "Marks" column, display it as `achieved_marks / max_marks`.
    5.  **Teacher's Corner:**
        - Create a final section `## For the Teacher`.
        - Display the `for_teacher` feedback as a blockquote (`>`).

    Produce only the Markdown content as your final output.
    """

    # Generate the Markdown content
    response = model.generate_content(prompt)
    return response.text


def get_mcq_report(evaluation_data):
    """
    Generates a structured student report in Markdown using Gemini.

    Args:
        evaluation_data: A dictionary with the student's evaluation results.

    Returns:
        A string containing the report in Markdown format.
    """
    # Construct a detailed, structured prompt for the Gemini model
    prompt = f"""
    You are an AI Teaching Assistant. Your task is to generate a personalized student performance report in Markdown format based on the data provided below.

    **Instructions:**
    1.  **Main Title:** Start with "# Student Performance Report 📝".
    2.  **Student Info:** Create a section for the student's name and USN.
    3.  **Performance Summary:** Create a section summarizing the final score. Add a congratulatory remark for a perfect score.
    4.  **Answer Table:** Create an "Answer Analysis" section with a Markdown table. The table must have columns: "Question #", "Your Answer", "Correct Answer", and "Result". Use '✅' for correct and '❌' for incorrect answers.
    5.  **Feedback:** Conclude with a "Feedback & Remarks" section. Provide encouraging and constructive feedback.

    **Evaluation Data:**
    - **Name:** {evaluation_data.get('student_name', 'N/A')}
    - **USN:** {evaluation_data.get('USN', 'N/A')}
    - **Score Obtained:** {evaluation_data.get('scores_obtained', 0)} / {evaluation_data.get('maximum_score', 0)}
    - **Student's Answers:** {evaluation_data.get('students_answers', {})}
    - **Correct Answers:** {evaluation_data.get('answer_key_answers', {})}

    Generate the complete report now.
    """

    # Select the model and generate the content
    model = GenerativeModel("gemini-2.5-pro")
    response = model.generate_content(prompt)

    return response.text


def get_generic_answer(query):
    prompt = f"""
            You are a specialized AI assistant. Your primary functions are "MCQ Evaluation" and "Descriptive Answer Evaluation".
            A user has submitted a request that falls outside of these specialized capabilities.

            Your task is to generate a response that does the following in a helpful and friendly tone:
            1.  Start by clearly stating that your main purpose is evaluating MCQs and descriptive answers.
            2.  Add a brief disclaimer that because their request is outside this scope, your answer might not be as relevant or detailed as your core functions would provide.
            3.  Finally, proceed to give a direct and helpful answer to the user's original request.

            User's Original Request: "{query}"
        """
    model = GenerativeModel("gemini-2.5-pro")
    output = model.generate_content(prompt)

    return output.candidates[0].content.text


AVAILABLE_TOOLS_SMARTEVAL = {
    "get_usn_name_from_image": get_usn_name_from_image,
    "get_mcq_answers": get_mcq_answers,
    "get_information_from_answer_paper": get_information_from_answer_paper,
    "evaluate_and_generate_feedback_with_gemini": evaluate_and_generate_feedback_with_gemini,
    "create_mark_down_report": create_mark_down_report,
    "get_mcq_report": get_mcq_report,
    "get_generic_answer": get_generic_answer,
}
# VIDYAMITRA_SYSTEM_PROMPT = """You are VidyaMitra, an expert AI teaching assistant for India. Your goal is to help teachers by creating educational materials. You must use the provided tools to fulfill the user's request. If an image is provided, use 'analyze_image_content'. If a PDF is mentioned, use 'analyze_pdf_content'. When asked for a reading assessment, use the audio generation tool. Always respond in a helpful, encouraging tone and format your text responses in clear markdown."""


AVAILABLE_TOOLS = {
    "create_lesson_plan": create_lesson_plan,
    "create_worksheets": create_worksheets,
    "simplify_concept": simplify_concept,
    "analyze_image_content": analyze_image_content,
    "analyze_pdf_content": analyze_pdf_content,
    "analyze_video_content": analyze_video_content,
    "generate_hyperlocal_explanation": generate_hyperlocal_explanation,
    "generate_audio_reading_assessment": generate_audio_reading_assessment,
}


# --- Authentication Decorator ---
def check_auth(f):
    def wrapper(*args, **kwargs):
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return jsonify({"error": "Unauthorized"}), 401
        id_token = auth_header.split("Bearer ")[1]
        try:
            decoded_token = auth.verify_id_token(id_token)
            request.user_id = decoded_token["uid"]  # Attach user_id to request
        except Exception as e:
            return jsonify({"error": "Invalid token", "message": str(e)}), 401
        return f(*args, **kwargs)

    wrapper.__name__ = f.__name__
    return wrapper


# --- Firestore Chat History Functions ---
def get_chat_history(user_id):
    history_ref = (
        db.collection("chats")
        .document(user_id)
        .collection("vidyamitra")
        .order_by("timestamp")
        .limit(20)
    )
    docs = history_ref.stream()
    history = []
    for doc in docs:
        history.append(doc.to_dict())
    return history


def save_chat_turn(user_id, user_message, agent_response):
    chat_doc_ref = (
        db.collection("chats").document(user_id).collection("vidyamitra").document()
    )
    chat_doc_ref.set(
        {
            "user_message": user_message,
            "agent_response": agent_response,
            "timestamp": datetime.now(),
        }
    )


# --- Main Agent Logic ---
@app.route("/vidyamitra-agent", methods=["POST"])
@check_auth
def vidyamitra_agent_endpoint():
    data = request.json
    goal = data.get("goal")
    file_url = data.get("fileUrl")
    coordinates = data.get("coordinates")
    language = data.get("language")
    if not goal:
        return jsonify({"error": "Missing goal"}), 400

    # Here you could load past history to continue a conversation, but for simplicity we start fresh each time
    chat = vidyamitra_model.start_chat()
    prompt_parts = [VIDYAMITRA_SYSTEM_PROMPT, f'\nUser\'s Goal: "{goal}"']
    if file_url:
        prompt_parts.append(
            f"\nA file has been provided for context at this URL: {file_url}"
        )
    if coordinates and language:
        prompt_parts.append(
            f"\nThe user's context is: Coordinates={coordinates}, Language='{language}'. Use this for hyperlocal explanations.")
    elif coordinates:
        prompt_parts.append(
            f"\nThe user's context is: Coordinates={coordinates}. Use this for hyperlocal explanations.")
    elif language:
        prompt_parts.append(f"\nThe user's context is: Language='{language}'. Use this for hyperlocal explanations.")

    print(prompt_parts)

    response = chat.send_message(prompt_parts)

    print("Response ", response)

    while response.candidates and response.candidates[0].function_calls:
        function_calls = response.candidates[0].function_calls
        api_responses = []
        for function_call in function_calls:
            tool_function = AVAILABLE_TOOLS.get(function_call.name)
            if tool_function:
                function_args = {
                    key: value for key, value in function_call.args.items()
                }
                # --- START: NEW LOGIC FOR AUDIO ASSESSMENT ---
                if function_call.name == "generate_audio_reading_assessment":
                    print("Handling special audio assessment case...")
                    # Execute the tool function
                    raw_tool_response = tool_function(**function_args)

                    try:
                        # Parse the custom string format from the tool
                        text_part = raw_tool_response.split("PASSAGE_TEXT_START\n")[
                            1
                        ].split("\nPASSAGE_TEXT_END")[0]
                        audio_base64_part = raw_tool_response.split(
                            "AUDIO_BASE64_START\n"
                        )[1].split("\nAUDIO_BASE64_END")[0]

                        # Create the final, structured JSON response
                        final_result = {
                            "responseType": "audio_assessment",
                            "passageText": text_part.strip(),
                            "audioBase64": audio_base64_part.strip(),
                        }

                        save_chat_turn(request.user_id, goal, final_result)
                        return jsonify(
                            final_result
                        )  # Return directly and exit the function

                    except IndexError:
                        # Fallback in case parsing fails
                        return (
                            jsonify({"error": "Failed to parse audio tool response"}),
                            500,
                        )
                # --- END: NEW LOGIC FOR AUDIO ASSESSMENT ---
                if function_call.name == "generate_hyperlocal_explanation":
                    function_args["coordinates"] = coordinates
                    function_args["language"] = language
                if function_call.name == "analyze_pdf_content":
                    function_args["pdf_url"] = file_url
                elif function_call.name == "analyze_image_content":
                    function_args["image_url"] = file_url
                elif function_call.name == "analyze_video_content":
                    function_args["video_url"] = file_url

                if "fileUrl" in function_args:
                    del function_args["fileUrl"]
                function_response = tool_function(**function_args)
                api_responses.append(
                    Part.from_function_response(
                        name=function_call.name, response={"content": function_response}
                    )
                )
            else:
                api_responses.append(
                    Part.from_function_response(
                        name=function_call.name,
                        response={
                            "content": f"Error: Tool '{function_call.name}' not found."
                        },
                    )
                )
        response = chat.send_message(api_responses)

    result = response.text
    save_chat_turn(request.user_id, goal, result)  # Save conversation turn
    return jsonify({"result": result})


@app.route("/smarteval-agent", methods=["POST"])
# @check_auth
def smarteval_agent_endpoint():
    data = request.json
    user_goal = data.get("userGoal")
    answer_key_sheet_url = data.get("answerKeySheetUrl")
    student_answer_sheet_url = data.get("studentAnswerSheetUrl")
    if not all([user_goal, answer_key_sheet_url, student_answer_sheet_url]):
        return (
            jsonify(
                {
                    "error": "Missing required fields: goal, answerKeyUrl, studentSheetUrl"
                }
            ),
            400,
        )

    planner_model = GenerativeModel("gemini-2.5-pro")

    answer_key_sheet_image_response = requests.get(answer_key_sheet_url)
    answer_key_sheet_image_response.raise_for_status()
    answer_key_sheet = Image.from_bytes(answer_key_sheet_image_response.content)

    student_answer_sheet_image_response = requests.get(student_answer_sheet_url)
    student_answer_sheet_image_response.raise_for_status()
    students_answer_sheet = Image.from_bytes(
        student_answer_sheet_image_response.content
    )

    prompt = [PLANNER_SYSTEM_PROMPT, "User's Goal:", user_goal]

    response = planner_model.generate_content(prompt)
    chosen_workflow_name = response.text.strip()
    print(f"Planner decided to run: {chosen_workflow_name}")

    chosen_workflow = WORKFLOWS.get(chosen_workflow_name)

    if not chosen_workflow:
        print("Could not determine a valid workflow. Aborting.")
        return

    # --- 3. Execute the chosen_workflow step-by-step ---
    # This dictionary will store the results from each step
    execution_context = {
        "answer_key_sheet_image_response": answer_key_sheet_image_response,
        "student_answer_sheet_image_response": student_answer_sheet_image_response,
        "answer_key_sheet": answer_key_sheet,  # Initial data
        "students_answer_sheet": students_answer_sheet,
    }

    for tool_name in chosen_workflow:
        print(f"Executing step: {tool_name}")
        # Identifying the function associated with the tool chosen
        tool_function = AVAILABLE_TOOLS_SMARTEVAL.get(tool_name)

        if not tool_function:
            print(f"Error: Tool '{tool_name}' not found.")
            continue

        try:
            # Dynamically prepare the arguments for the current tool
            args_to_pass = {}
            if tool_name == "get_usn_name_from_image":
                args_to_pass["image"] = execution_context["students_answer_sheet"]
                result = tool_function(**args_to_pass)
                result = json.dumps(result)
                execution_context.update(json.loads(result))

            elif tool_name == "get_mcq_answers":
                # This tool needs both the student sheet and the answer key
                # Finding the choices that the student has chosen
                args_to_pass["image_response"] = execution_context[
                    "student_answer_sheet_image_response"
                ]
                result = tool_function(**args_to_pass)
                execution_context["students_answers"] = result

                # Finding the choice that the teacher answer key
                args_to_pass["image_response"] = execution_context[
                    "answer_key_sheet_image_response"
                ]
                result = tool_function(**args_to_pass)
                execution_context["answer_key_answers"] = result

                # Finding the scores
                scores = [
                    1
                    for students_ans, teachers_ans in zip(
                        execution_context["students_answers"].values(),
                        execution_context["answer_key_answers"].values(),
                    )
                    if students_ans == teachers_ans
                ]
                execution_context["scores_obtained_per_question"] = scores
                execution_context["scores_obtained"] = sum(scores)
                execution_context["maximum_score"] = len(
                    execution_context["answer_key_answers"]
                )

                # You might also want to call it on the answer key separately
                # For simplicity, we assume the tool handles this logic.

            elif tool_name == "get_text_from_image":
                args_to_pass["image_path"] = execution_context["students_answer_sheet"]

            elif tool_name == "get_information_from_answer_paper":
                # Appending students answer sheet information
                args_to_pass["image"] = execution_context["students_answer_sheet"]
                temp = {}
                result = tool_function(**args_to_pass)

                temp["student_answer_sheet_info"] = result
                execution_context.update(temp)

                # Appending teachers answer sheet information
                args_to_pass["image"] = execution_context["answer_key_sheet"]
                temp = {}
                result = tool_function(**args_to_pass)

                temp["answer_key_sheet_info"] = result
                execution_context.update(temp)

            elif tool_name == "evaluate_descriptive_answers":
                # This tool needs the text extracted from the PREVIOUS step
                if "get_text_from_image_result" in execution_context:
                    args_to_pass["extracted_text"] = execution_context[
                        "get_text_from_image_result"
                    ]
                else:
                    raise ValueError(
                        "Cannot evaluate descriptive answers without extracted text from the previous step."
                    )
            elif tool_name == "evaluate_and_generate_feedback_with_gemini":
                args_to_pass["agent_state"] = execution_context
                result = tool_function(**args_to_pass)
                execution_context.update(result)

            elif tool_name == "create_mark_down_report":

                # args_to_pass["report_data"]=execution_context
                result = tool_function(report_data=execution_context)

                return result

            elif tool_name == "get_mcq_report":
                result = tool_function(evaluation_data=execution_context)

                return result

            elif tool_name == "get_generic_answer":
                args_to_pass["query"] = user_goal
                result = tool_function(**args_to_pass)

                return result

            else:
                raise Exception(f"Tool {tool_name} not implemented.")
        except Exception as e:
            raise Exception(f"Error executing tool {tool_name}: {e}")


# --- Other API Endpoints ---
@app.route("/vidyamitra-agent/chathistory", methods=["GET"])
@check_auth
def get_history_endpoint():
    history = get_chat_history(request.user_id)
    return jsonify(history)


@app.route("/transcribe-audio", methods=["POST"])
@check_auth
def transcribe_audio_endpoint():
    if "audio_file" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio_file"]
    content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,  # Common for browser MediaRecorder
        sample_rate_hertz=48000,  # Common for browser MediaRecorder
        language_code="en-US",  # Or make this dynamic
    )

    try:
        response = speech_client.recognize(config=config, audio=audio)
        if response.results:
            transcript = response.results[0].alternatives[0].transcript
            return jsonify({"transcript": transcript})
        else:
            return jsonify({"transcript": ""})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=6000)