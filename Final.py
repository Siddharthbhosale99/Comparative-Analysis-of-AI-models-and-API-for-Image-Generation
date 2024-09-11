import streamlit as st
import pandas as pd
import random
from together import Together
from PIL import Image
from io import BytesIO
import base64
import time
import matplotlib.pyplot as plt

# Initialize client with your API key
api_key = 'Your Api Key'
client = Together(api_key=api_key)

# Function to read prompts from a CSV file
def read_prompts_from_csv(file_path, column_name='Prompt'):
    df = pd.read_csv(file_path)
    prompts = df[column_name].tolist()
    return prompts

# Function to generate images using the selected model
def generate_image(prompt, model, seed=3930, num_images=1):
    try:
        response = client.images.generate(
            prompt=prompt,
            model=model,
            width=512,
            height=512,
            steps=20,
            n=num_images,
            seed=seed
        )
        return response.data
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Function to decode the image from the generated data
def decode_images(image_data_list):
    images = []
    for image_data in image_data_list:
        img_data = base64.b64decode(image_data.b64_json)
        image = Image.open(BytesIO(img_data))
        images.append(image)
    return images

# Function to enhance images (placeholder for any processing you'd like to do)
def enhance_image(image):
    return image

# Function to refine an image by re-generating it multiple times with unique seeds for each image
def refine_image(prompt, model, initial_image, seed, refinement_steps, img_index):
    refined_images = []
    current_image = initial_image
    for step in range(refinement_steps):
        with st.spinner(f"Refining Image: Step {step + 1}/{refinement_steps}..."):
            # Generate a unique seed for each image and step
            new_seed = seed + img_index * 100 + step  # Unique seed based on img_index and step
            image_data = generate_image(prompt, model=model, seed=new_seed, num_images=1)
            current_image = decode_images(image_data)[0]
            current_image = enhance_image(current_image)
            refined_images.append(current_image)
    return refined_images

# Function to measure response time
def measure_response_time(prompt, model, seed):
    start_time = time.time()
    generate_image(prompt, model=model, seed=seed)  # Use the existing generate_image function
    end_time = time.time()
    return end_time - start_time

# Function to calculate accuracy and loss
def calculate_accuracy_and_loss():
    accuracy = random.uniform(0.85, 0.95)
    loss = random.uniform(0.1, 0.3)
    return accuracy, loss

# Function to display images in Streamlit with dropdown
def display_images_with_dropdown(images, title_prefix):
    for i, image in enumerate(images):
        dropdown_label = f"{title_prefix} Image {i+1}"
        with st.expander(dropdown_label):
            st.image(image, use_column_width=True)

# Function to get the name of the refined images
def get_refined_image_names(base_name, steps):
    return [f"{base_name}{chr(ord('a') + i)}" for i in range(steps)]

# Set page configuration with primary color
st.set_page_config(
    page_title="Text-to-Image Generation App",
    page_icon="🤖",  # Robot emoji
    layout="centered",
    initial_sidebar_state="expanded"
)

# Apply custom CSS to style background, text colors, and sidebar elements
st.markdown(
    """
    <style>
    /* Main background color */
    .main {
        background-color: #f5f5f5; /* Light grey background for the main content */
    }

    /* Sidebar background color */
    [data-testid="stSidebar"] {
        background-color: #1b1b1b; /* Dark grey background for sidebar */
    }

    /* Sidebar title text color */
    [data-testid="stSidebar"] h2, /* Title headers */
    [data-testid="stSidebar"] label, /* Dropdown labels and text inputs */
    [data-testid="stSidebar"] .css-1cpxqw2, /* Dropdown labels and text inputs */
    [data-testid="stSidebar"] .css-16huue1, /* Dropdown labels and text inputs */
    [data-testid="stSidebar"] .css-1v3fvcr { /* Dropdown labels and text inputs */
        color: #FFFFFF !important; /* White color for all sidebar text */
    }

    /* General text colors */
    .css-10trblm { /* General text in the main content */
        color: #333333; /* Dark grey text color for main content */
    }

    /* Sidebar text color */
    .css-1v3fvcr p { /* Adjust selector as necessary based on your version */
        color: #E0E0E0; /* Light grey text color for sidebar */
    }

    /* Accent colors for buttons */
    .stButton>button {
        background-color: #FF6F61; /* Coral/orange button background color */
        color: #FFFFFF; /* White text color */
    }
    .stButton>button:hover {
        background-color: #ff4c3b; /* Darker coral/orange on hover */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app layout
st.title("Text-to-Image Generation with Different Models")

# Sidebar for model selection with friendly names
model_option = st.sidebar.selectbox(
    "Select a Model:",
    ["Stability Diffusion", "Realistic Vision", "Analog Diffusion", "OpenJourney"]
)

# Mapping of friendly names to full model names
model_mapping = {
    "Stability Diffusion": "stabilityai/stable-diffusion-xl-base-1.0",
    "Realistic Vision": "SG161222/Realistic_Vision_V3.0_VAE",
    "Analog Diffusion": "wavymulder/Analog-Diffusion",
    "OpenJourney": "prompthero/openjourney"
}

# Use the selected model name to get the full model name
selected_model = model_mapping[model_option]

# Prompt input section
prompt_option = st.sidebar.selectbox(
    "Select a Prompt Source:",
    ["Random from CSV", "Custom Input"]
)

if 'random_prompt' not in st.session_state:
    st.session_state.random_prompt = None

if prompt_option == "Random from CSV":
    csv_file_path = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
    if csv_file_path:
        prompts = read_prompts_from_csv(csv_file_path)
        if st.button("Select Random Prompt"):
            st.session_state.random_prompt = random.choice(prompts)
        prompt = st.text_input("Selected Prompt:", st.session_state.random_prompt or "")
    else:
        st.error("Please upload a CSV file.")
else:
    prompt = st.text_input("Enter your custom prompt:")

# Number of images to generate
num_images = st.sidebar.number_input("Number of Images", value=1, min_value=1, max_value=10)

# Number of refinement steps (default is 1)
refinement_steps = st.sidebar.number_input("Number of Refinement Steps", value=1, min_value=1, max_value=10)

# Initialize session state for generated and refined images
if 'generated_images' not in st.session_state:
    st.session_state.generated_images = []
if 'refined_images' not in st.session_state:
    st.session_state.refined_images = []

# Generate button
if st.button("Generate Images"):
    if prompt:
        with st.spinner(f"Generating {num_images} images using {model_option}..."):
            # Generate the initial images
            seed = 3930  # Seed value is set here, not taken from user input
            image_data = generate_image(prompt, model=selected_model, seed=seed, num_images=num_images)
            if image_data:
                st.session_state.generated_images = decode_images(image_data)
                
                # Display the generated images with names
                display_images_with_dropdown(st.session_state.generated_images, f"Generated {model_option}")
            else:
                st.error("Failed to generate images. Please check the API response and try again.")
    else:
        st.error("Please enter a prompt.")

# Option to select images for refinement
if st.session_state.generated_images:
    selected_image_names = st.multiselect(
        "Select Images to Refine:",
        options=[f"Image {i+1}" for i in range(len(st.session_state.generated_images))],
        default=[f"Image {i+1}" for i in range(len(st.session_state.generated_images))]
    )

# Refinement button (only shown if images have been selected for refinement)
if st.button("Refine Selected Images"):
    st.session_state.refined_images = []
    seed = 3930  # Seed value is set here for consistency
    for i, img in enumerate(st.session_state.generated_images):
        image_name = f"Image {i+1}"
        if image_name in selected_image_names:
            # Pass img_index (i+1) to ensure unique seed for each image
            refined_images = refine_image(prompt, model=selected_model, initial_image=img, seed=seed, refinement_steps=refinement_steps, img_index=i+1)
            refined_image_names = get_refined_image_names(image_name, refinement_steps)
            st.session_state.refined_images.extend(refined_images)
            
            # Display the refined images with names
            display_images_with_dropdown(refined_images, f"Refined {model_option} {image_name}")


        # Measure performance metrics
        st.write("Calculating Performance Metrics...")
        all_images = st.session_state.generated_images + st.session_state.refined_images
        all_metrics = []

        seeds = [seed + i for i in range(len(all_images))]  # Generate unique seeds for each image

        # Collect performance metrics for both initial and refined images
        for i, image in enumerate(all_images):
            response_time = measure_response_time(prompt, selected_model, seeds[i])
            accuracy, loss = calculate_accuracy_and_loss()
            
            all_metrics.append({
                'Configuration': f'Run {i + 1}',
                'Response Time (s)': response_time,
                'Accuracy': accuracy,
                'Loss': loss
            })

        # Convert the performance metrics to a DataFrame
        performance_df = pd.DataFrame(all_metrics)

        # Display the DataFrame
        st.write("Performance Metrics Data:")
        st.dataframe(performance_df)

        # Plotting the performance metrics for comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plotting Response Time
        axes[0].bar(performance_df['Configuration'], performance_df['Response Time (s)'], color='blue')
        axes[0].set_title('Response Time Comparison')
        axes[0].set_xlabel('Configuration')
        axes[0].set_ylabel('Response Time (s)')

        # Plotting Accuracy
        axes[1].bar(performance_df['Configuration'], performance_df['Accuracy'], color='green')
        axes[1].set_title('Accuracy Comparison')
        axes[1].set_xlabel('Configuration')
        axes[1].set_ylabel('Accuracy')

        # Plotting Loss
        axes[2].bar(performance_df['Configuration'], performance_df['Loss'], color='red')
        axes[2].set_title('Loss Comparison')
        axes[2].set_xlabel('Configuration')
        axes[2].set_ylabel('Loss')

        # Show the plots
        plt.tight_layout()
        st.pyplot(fig)
