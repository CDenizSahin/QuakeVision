🌍 QuakeVision

AI-Powered Disaster Damage Assessment & Emergency Response Platform

<p align="center">
  <strong>Turning Satellite & Drone Imagery into Actionable Disaster Intelligence.</strong>
</p>

QuakeVision is an AI-powered disaster assessment platform designed to analyze satellite and drone imagery after earthquakes, automatically detect structural damage, visualize affected areas geographically, and generate intelligent emergency response reports.

By combining Computer Vision, Deep Learning, Geospatial Analysis, and Generative AI, QuakeVision transforms raw aerial imagery into actionable intelligence that can support disaster management teams and emergency response operations.

⸻

🚀 Overview

In the aftermath of a major earthquake, identifying damaged buildings and determining priority intervention areas is critical.

QuakeVision addresses this challenge through an end-to-end AI pipeline that:

* Detects buildings and classifies structural damage using a custom-trained YOLOv8 model.
* Processes satellite and drone imagery.
* Extracts GPS coordinates from image EXIF metadata.
* Visualizes detected damage on interactive geographical maps.
* Analyzes regional destruction levels.
* Generates AI-assisted Search and Rescue (SAR) and logistical response recommendations.
* Exports structured disaster assessment reports as PDF documents.

The goal is to reduce manual assessment time and provide decision-makers with faster, data-driven insights.

⸻

✨ Key Features

1. AI-Powered Structural Damage Detection

Powered by a YOLOv8 model trained using Transfer Learning on the xBD disaster damage assessment dataset.

The system classifies detected structures into four damage categories:

Class	Description
🟢 No Damage	Structurally undamaged buildings
🟡 Minor Damage	Buildings with minor structural damage
🟠 Major Damage	Buildings with severe structural damage
🔴 Destroyed	Completely destroyed buildings

Detection results include bounding boxes and class predictions, enabling automated visual assessment.

2. Geospatial Intelligence & GIS

QuakeVision integrates image metadata processing with geospatial visualization.

* Extracts GPS coordinates from EXIF metadata.
* Maps detected buildings to their geographical locations.
* Generates interactive damage heatmaps using Folium.
* Helps identify high-priority disaster zones.

3. AI-Assisted Disaster Response

The platform integrates Large Language Models (LLMs) to transform detection results into actionable recommendations.

* Generates disaster assessment summaries.
* Evaluates regional damage distribution.
* Produces logistics and Search and Rescue (SAR) planning recommendations.
* Uses Google Gemini 3.5 Flash and locally hosted Llama 3 models.

4. Intelligent Cloud Fallback Architecture

QuakeVision incorporates a fallback mechanism to handle limited local hardware resources.

When local inference or processing resources become insufficient, the system can switch to cloud-based AI services.

This architecture improves resource flexibility and supports more resilient AI workflows.

5. Automated PDF Reporting

The platform converts analysis results into structured PDF reports using FPDF.

Reports include:

* Damage assessment summaries.
* Structural damage distribution.
* Geographical analysis results.
* AI-generated emergency response recommendations.
* Logistics and SAR planning insights.

6. Modern Interactive Dashboard

Built with Streamlit, the user interface provides an accessible environment for uploading imagery, reviewing AI predictions, exploring geographical results, and generating reports.

⸻

🏗️ System Architecture

QuakeVision follows a modular architecture that separates the user interface, backend services, computer vision pipeline, geospatial processing, and AI-powered reporting.

flowchart TD
    A[Satellite / Drone Images] --> B[Streamlit Dashboard]
    B --> C[FastAPI Backend]
    C --> D[Image Processing Pipeline]
    D --> E[YOLOv8 Damage Detection]
    E --> F[Damage Classification]
    F --> G[EXIF GPS Extraction]
    G --> H[Geospatial Analysis]
    H --> I[Folium Heatmap]
    F --> J[LLM Intelligence Layer]
    J --> K[Gemini 3.5 Flash]
    J --> L[Local Llama 3]
    K --> M[Response Planning]
    L --> M
    M --> N[PDF Report Generation]
    I --> O[Interactive Dashboard]
    N --> O

⸻

🛠️ Technology Stack

Category	Technologies
Programming Language	Python
Computer Vision	YOLOv8, OpenCV
Deep Learning	PyTorch
Dataset	xBD
Backend	FastAPI
Frontend	Streamlit
Geospatial Analysis	Folium, EXIF
Generative AI	Google Gemini 3.5 Flash, Llama 3
Reporting	FPDF
API Architecture	RESTful API, Asynchronous Processing

⸻

🔬 Machine Learning Pipeline

The computer vision component is based on Transfer Learning using the xBD dataset.

The pipeline consists of:

1. Input image acquisition.
2. Image preprocessing.
3. YOLOv8 inference.
4. Bounding box generation.
5. Structural damage classification.
6. Prediction result extraction.
7. Geospatial association.
8. Damage distribution analysis.

The model identifies four structural damage categories, allowing the platform to transform aerial imagery into structured damage assessment data.

⸻

🗺️ Geospatial Visualization

One of the core capabilities of QuakeVision is connecting AI predictions with real-world geographical locations.

By extracting GPS metadata from uploaded images, the platform can associate detected damage with geographical coordinates.

Folium is used to generate interactive maps and heatmaps, helping visualize damage concentration and identify potentially critical areas.

⸻

🤖 Generative AI Integration

QuakeVision combines local and cloud-based language models to generate contextual disaster assessment insights.

Supported models:

* Google Gemini 3.5 Flash
* Llama 3

The AI layer processes structured damage assessment results and generates recommendations for emergency response, logistics, and Search and Rescue planning.

⸻

📄 Automated Reporting

The reporting module transforms analysis outputs into structured PDF documents.

Its purpose is to make technical AI predictions more accessible to disaster management teams by presenting findings in a readable and organized format.

⸻

🎯 Project Objectives

QuakeVision was developed with the following objectives:

* Accelerate post-earthquake structural damage assessment.
* Reduce dependency on time-consuming manual inspections.
* Connect AI-based detection with geographical intelligence.
* Support emergency logistics and rescue prioritization.
* Provide an integrated and modular disaster intelligence architecture.

⸻

👨‍💻 Development & Engineering

QuakeVision was designed and developed end-to-end as part of the Microsoft AI Innovators Summer Internship Program 2026.

The project involved approximately 20 days of research, architecture design, model development, backend implementation, AI integration, and interface development.

Key engineering areas explored:

* End-to-end AI system architecture.
* Transfer Learning and custom YOLOv8 training.
* Asynchronous backend development with FastAPI.
* RESTful service communication.
* GPU/VRAM and memory optimization.
* Local and cloud-based LLM integration.
* Exception handling and fault-tolerant design.
* Modular software architecture and Clean Code principles.

⸻

⚠️ Disclaimer

QuakeVision is an experimental AI-powered disaster assessment and decision-support platform.

Its outputs are intended to support preliminary damage assessment and emergency response planning. AI-generated predictions should not replace professional structural inspections, official damage assessments, or decisions made by authorized disaster management organizations.

⸻

👤 Author

Deniz Şahin

Software Engineering | Artificial Intelligence | Computer Vision | Backend Development

Developed as part of the Microsoft AI Innovators Summer Internship Program 2026.

⸻

<p align="center">
  <strong>QuakeVision — From Aerial Imagery to Disaster Intelligence.</strong>
  <br/>
  Built with Python, AI, and Engineering.
</p>
