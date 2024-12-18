import google.generativeai as genai
from pathlib import Path
import gradio as gr
from dotenv import load_dotenv
import os
import pandas as pd

# Load environment variables from a .env file
load_dotenv()

# Configure the GenerativeAI API key using the loaded environment variable
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Set up the model configuration for text generation
generation_config = {
    "temperature": 0.4,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
}

# Define safety settings for content generation
safety_settings = [
    {"category": f"HARM_CATEGORY_{category}", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
    for category in ["HARASSMENT", "HATE_SPEECH", "SEXUALLY_EXPLICIT", "DANGEROUS_CONTENT"]
]

# Initialize the GenerativeModel with the new model name
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash", 
    generation_config=generation_config,
    safety_settings=safety_settings,
)

# Load the dataset and clean column names
dataset = pd.read_csv("dataset.csv")
dataset.columns = dataset.columns.str.strip()  # Clean column names by stripping spaces

# Plant Pathologist Prompt
plant_pathologist_prompt = """
As a highly skilled plant pathologist, analyze the provided image of a plant to identify potential diseases. Your response should include:
1. **Disease Identification**: Name and describe the disease or issue.
2. **Affected Areas**: Specify which parts of the plant are affected.
3. **Causes**: List potential causes (e.g., pests, environment, infection).
4. **Next Steps**: Suggest treatments, preventive measures, and future steps.

**Important**: Be concise, detailed, and helpful. Ensure actionable recommendations.
"""

# Function to read image data from a file path
def read_image_data(file_path):
    image_path = Path(file_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Could not find image: {image_path}")
    return {"mime_type": "image/jpeg", "data": image_path.read_bytes()}

# Function to generate a plant disease analysis response
def generate_gemini_response(prompt, image_path):
    image_data = read_image_data(image_path)
    response = model.generate_content([plant_pathologist_prompt, image_data])
    return response.text

# Function to process uploaded files and analyze plant diseases
def process_uploaded_files(files):
    file_path = files[0].name if files else None
    response = generate_gemini_response(plant_pathologist_prompt, file_path) if file_path else None
    return file_path, response

# Function to provide fertilizer recommendations
def get_fertilizer_recommendation(crop_type, soil_type, temperature):
    # Filter dataset based on user inputs
    recommendations = dataset[(dataset['Crop_Type'].str.strip() == crop_type) & 
                            (dataset['Soil_Type'].str.strip() == soil_type) & 
                            (dataset['Temparature'] == temperature)]
    
    if recommendations.empty:
        return "**No recommendations found for the given input.**"

    # Get the first recommendation (assuming one is sufficient)
    recommendation = recommendations.iloc[0]
    return f"""
<div style="background-color: white; padding: 10px; border-radius: 8px; box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.2);">
    <b>Recommended Fertilizer</b>: {recommendation['Fertilizer']}<br>
    <b>Brand</b>: {recommendation['Brand']}<br>
    <b>Usage</b>: {recommendation['Usage']}<br>
    <b>Instructions</b>: {recommendation['Instructions']}<br>
    <b>Fertilizer Quantity per Acre</b>: {recommendation['Fertilizer_Quantity_per_Acre']}<br>
    <b>Irrigation Requirement</b>: {recommendation['Irrigation_Requirement']}
</div>
    """

# Gradio interface setup
with gr.Blocks() as demo:
    # Add custom CSS for styling the page
    gr.HTML("""
    <style>
        .gradio-container {
            background: url('https://picsum.photos/id/89/4608/2592'); /* Background image */
            background-size: cover;
            background-position: center;
            height: 100vh;
        }
        .header {
            color: white;
            font-size: 2rem;
            font-weight: bold;
            margin: 0;
            text-align: center;
            padding: 10px 0;
            background-color: #228B22;
        }
        .button-custom {
            background-color: #228B22 !important;
            color: white !important;
            border: none !important;
        }
        h2, h3 {
            color: white !important;
        }
    </style>
    """)

    # Header Section
    gr.HTML("<div class='header'>🌾 Agribot Plant Health and Fertilizer Advisor 🌱</div>")

    # Plant disease analysis section
    gr.Markdown("## 🌱 Plant Disease Analysis")
    image_output = gr.Textbox(label="Analysis Results")
    file_output = gr.Image(label="Uploaded Image")
    combined_output = [file_output, image_output]

    upload_button = gr.UploadButton(
        "Click to Upload an Image",
        file_types=["image"],
        file_count="multiple",
        elem_classes=["button-custom"]  # Custom button styling
    )
    upload_button.upload(process_uploaded_files, inputs=[upload_button], outputs=combined_output)

    # Fertilizer recommendation section
    gr.Markdown("## 🌾 Fertilizer Recommendation System")
    with gr.Row():
        crop_type_input = gr.Textbox(label="Crop Type")
        soil_type_input = gr.Textbox(label="Soil Type")
        temperature_input = gr.Number(label="Temperature (°C)")

    recommendation_button = gr.Button("Get Recommendation", elem_classes=["button-custom"])
    recommendation_output = gr.HTML()  # Using HTML to support the styled white div output

    recommendation_button.click(
        get_fertilizer_recommendation,
        inputs=[crop_type_input, soil_type_input, temperature_input],
        outputs=recommendation_output,
    )

# Launch the Gradio interface
demo.launch(debug=True)
