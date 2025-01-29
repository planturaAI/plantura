# app.py
import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from utils import load_model
from torchvision.models import resnet18
import json
import google.generativeai as genai

GEMINI_CONFIG = {
    "temperature": 0.5,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 3096
}

st.set_page_config(
    page_title="Plantura",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stButton>button { width: 100%; }
    .css-1v0mbdj.etr89bj1 { text-align: center; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_plant_model():
    try:
        model = resnet18(num_classes=1081)
        loaded_model = load_model(
            model=model,
            filename='resnet18_weights_best_acc.tar',
            use_gpu=False
        )
        loaded_model.eval()
        return loaded_model
    except Exception as e:
        st.error(f"Failed to initialize model: {str(e)}")
        raise e

@st.cache_resource
def initialize_gemini():
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        return genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        st.error(f"Failed to initialize Gemini: {str(e)}")
        raise e

def classify_plant(image_file, model):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_file).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(img_tensor)
    
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    _, class_id = torch.max(probabilities, 0)
    return class_id.item(), get_plant_name(class_id.item())  

def get_plant_name(class_id):
    try:
        with open('classFile.json') as f:
            class_mapping = json.load(f)
        intermediate_id = class_mapping.get(str(class_id))
        
        with open('nameFile.json') as f:
            name_mapping = json.load(f)
            
        plant_name = name_mapping.get(str(intermediate_id)) or \
                    name_mapping.get(str(int(intermediate_id)))
        
        return plant_name
        
    except Exception as e:
        st.error(f"Name lookup error: {str(e)}")
        return "Unknown Plant"

def analyze_plant_health(image, plant_name, gemini_model):
    try:
        prompt = f"""
        Analyze the health of this {plant_name} plant. Consider:
        - Leaf color and texture
        - Stem condition
        - Signs of pests/disease
        - Overall vitality
        Provide care recommendations in markdown format with bullet points.
        Include intersting helpful facts for the {plant_name}.
        Note: dont prompt anything like i will analysis he image or smth like that, just answer immediatily.
        Pleace be consice and straightforaward get to the point directly.
        this is an tested prompt from u:That's not Mercurialis annua L. That's a Epipremnum aureum, commonly known as a Golden Pothos.

The image shows a healthy Epipremnum aureum. The leaves are a vibrant green with yellow variegation, indicating good light conditions. The stems appear strong and flexible. There are no visible signs of pests or diseases. Overall vitality is excellent.

Care Recommendations:
Light: Bright, indirect light is ideal. Avoid direct sunlight, which can scorch the leaves.
Watering: Allow the top inch of soil to dry out between waterings. Avoid overwatering, which can lead to root rot.
Soil: Well-draining potting mix is essential.
Humidity: Prefers moderate to high humidity. Consider grouping with other plants or using a pebble tray to increase humidity.
Fertilizing: Feed with a balanced liquid fertilizer every 2-4 weeks during the growing season (spring and summer).
Pruning: Prune as needed to maintain shape and size. Propagate cuttings easily in water.
Interesting Facts about Epipremnum aureum (Golden Pothos):
Air purification: NASA research suggests Pothos can help filter indoor air.
Low maintenance: Extremely tolerant of neglect, making it a great plant for beginners.
Toxicity: Toxic to cats and dogs if ingested. Keep out of reach of pets.
Versatile: Can be grown in hanging baskets, as a trailing plant, or trained to climb a support.
There is no information provided about Mercurialis annua L. If you have a different image or would like information about that species please provide it.

remove this:That's not Mercurialis annua L. That's a Epipremnum aureum, commonly known as a Golden Pothos.

The image shows a healthy Epipremnum aureum. The leaves are a vibrant green with yellow variegation, indicating good light conditions. The stems appear strong and flexible. There are no visible signs of pests or diseases. Overall vitality is excellent.
and this: There is no information provided about Mercurialis annua L. If you have a different image or would like information about that species please provide it.
dont say the plant name! just starting talking about the health and other things just start with this name "{get_common_name(plant_name, gemini_model)}" pick one if there is multiple names.
        """
        response = gemini_model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        st.error(f"Health analysis failed: {str(e)}")
        return None
    
def get_common_name(plant_name, gemini_model):
    try:
        prompt = f"""
        you will get a scientific plant name, your job is to give him back the only the common name nothing else!!!,

        Example:
        Scientific name:  Alocasia macrorrhizos (L.) G.Don
        Common name: Giant Taro, or Elephant Ear

        """
        gemini_model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-thinking-exp-01-21",
            system_instruction=prompt
        )
        response = gemini_model.generate_content(contents=plant_name)
        return response.text
    except Exception as e:
        st.error(f"Health analysis failed: {str(e)}")
        return None

def main():
    try:
        model = initialize_plant_model()
        gemini_model = initialize_gemini()
    except Exception as e:
        st.error("Initialization failed")
        st.stop()

    with st.sidebar:
        st.image("https://via.placeholder.com/150", caption="Plantura")
        st.markdown("## About")
        st.info("Plantura helps you identify plants and assess their health using AI.")
        
        st.markdown("### Features")
        st.markdown("- 🔍 Plant Identification\n- 🌿 Health Analysis\n- 🏥 Care Recommendations\n- ✨ intersting helpful facts")
        
        st.markdown("### Tips")
        st.markdown("- Use clear, well-lit photos\n- Capture the whole plant")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🌱 Plantura")
        st.markdown("### Your Smart Plant Care Assistant")
    
    st.markdown("---")
    upload_col, preview_col = st.columns([1, 1])
    
    with upload_col:
        st.markdown("### 📸 Upload Plant Image")
        uploaded_file = st.file_uploader(
            "Choose a plant photo...", 
            type=['jpg', 'jpeg', 'png']
        )

    if uploaded_file:
        with preview_col:
            st.markdown("### 🖼️ Preview")
            img = Image.open(uploaded_file)
            st.image(img, use_column_width=True)
        
        st.markdown("---")
        analysis_col1, analysis_col2 = st.columns([1, 1])
        
        with analysis_col1:
            with st.spinner('🔍 Identifying plant species...'):
                class_id, plant_name = classify_plant(uploaded_file, model)  
            
            if class_id is not None and plant_name:
                st.success(f"### Identified Plant")
                plant_common_name = get_common_name(plant_name,gemini_model)
                st.markdown(f"**Species**: {plant_name}, which is known as {plant_common_name}")  
                
                if st.button("🌿 Analyze Plant Health", type="primary"):
                    with st.spinner('🔬 Analyzing plant health...'):
                        analysis = analyze_plant_health(img, plant_name, gemini_model)
                    
                    if analysis:
                        with analysis_col2:
                            st.markdown("### 📊 Health Analysis")
                            st.markdown(analysis)
                    else:
                        st.error("Analysis failed. Please try again.")
            else:
                st.error("Identification failed. Try another image.")
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
    <p>Made with ❤️ by Plantura</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
