# AI Wildlife Identifier


## Overview
This Streamlit-based web application classifies wildlife species using the ResNet50 deep learning model. Additionally, it integrates Generative AI to provide detailed species information, fun facts, and an interactive Q&A feature.

## Features
- **Wildlife Classification**: Uses the pre-trained ResNet50 model to classify uploaded wildlife images.
- **Generative AI Insights**: Provides detailed descriptions, fun facts, and interactive Q&A on the identified species.
- **Image Upload & URL Support**: Users can upload an image or enter an image URL for classification.

## Installation
### Prerequisites
Ensure you have Python installed on your system.

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/wildlife-classifier.git
   cd wildlife-classifier
   ```
2. Install dependencies:
   ```bash
   pip install -r Requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Dependencies
The required Python libraries are:
- `streamlit`
- `tensorflow`
- `numpy`
- `pillow`
- `requests`
- `transformers`

Install them using:
```bash
pip install streamlit tensorflow numpy pillow requests transformers
```

## Usage
1. Open the application in your browser.
2. Upload an image or enter an image URL.
3. The app will classify the image and provide relevant insights.
4. View fun facts and ask questions about the species.

## Note : 
if you want to train your own model use the given dataset - https://www.kaggle.com/datasets/utkarshsaxenadn/animal-image-classification-dataset
## Acknowledgments
- ResNet50 Model: Pre-trained on ImageNet
- Hugging Face Transformers for Generative AI
  

