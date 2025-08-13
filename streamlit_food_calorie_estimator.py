import streamlit as st
from PIL import Image
import base64
import requests
import json
from io import BytesIO
import time

# ---- Constants ----
# The API key is automatically provided by the Canvas environment.
API_KEY = ""
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key="
MODEL_URL = f"{API_URL}{API_KEY}"

# ---- Helper Functions ----

def image_to_base64(image):
    """Converts a PIL Image object to a base64 encoded string."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def call_gemini_api(prompt, base64_image_data, retries=5):
    """
    Calls the Gemini API with a retry mechanism and exponential backoff.
    The response is expected to be a JSON string.
    """
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": prompt
                    },
                    {
                        "inlineData": {
                            "mimeType": "image/jpeg",
                            "data": base64_image_data
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "food_item": {"type": "STRING"},
                        "calories": {"type": "NUMBER"}
                    }
                }
            }
        }
    }

    delay = 1
    for i in range(retries):
        try:
            response = requests.post(MODEL_URL, json=payload)
            response.raise_for_status()  # Raise an exception for bad status codes
            result = response.json()
            json_string = result["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(json_string)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                st.error("Error: 403 Forbidden. There may be an issue with API key permissions.")
                return None
            st.warning(f"API call failed, retrying in {delay} seconds... (Attempt {i+1}/{retries})")
            time.sleep(delay)
            delay *= 2
        except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError) as e:
            st.error(f"Error calling Gemini API: {e}")
            return None

    st.error("Maximum retries exceeded. Could not get a response from the API.")
    return None

# ---- Streamlit UI ----

st.set_page_config(
    page_title="Food Calorie Estimator",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("Food Calorie Estimator")
st.markdown("Upload a picture of a meal to get a calorie estimate.")

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button("Estimate Calories"):
        if uploaded_file:
            with st.spinner("Analyzing image..."):
                base64_image = image_to_base64(image)
                
                # Craft a detailed prompt for the LLM
                prompt = (
                    "What food is in this picture? Provide the name of the food and a reasonable "
                    "estimated calorie count for a typical single serving size in an array of JSON objects. "
                    "For example, if you see an apple and a banana, return: "
                    "[{\"food_item\": \"Apple\", \"calories\": 95}, {\"food_item\": \"Banana\", \"calories\": 105}]. "
                    "Only respond with the JSON array, nothing else."
                )
                
                results = call_gemini_api(prompt, base64_image)

                if results:
                    st.success("Analysis complete!")
                    st.subheader("Calorie Breakdown")
                    
                    # Display results in a table
                    st.table(results)

                    # Calculate and display total calories
                    total_calories = sum(item.get("calories", 0) for item in results)
                    st.header(f"Total Estimated Calories: {total_calories} kcal")

        else:
            st.warning("Please upload an image first.")
