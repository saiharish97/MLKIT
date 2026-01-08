# AI Video Vision

AI Video Vision is a powerful Android application built with Kotlin that leverages Google's ML Kit and cutting-edge GenAI to provide real-time object detection and detailed image descriptions.

![App Icon](https://raw.githubusercontent.com/saiharish97/MLKIT/main/app/src/main/res/drawable/ml_kit_video_app_icon.png)

## Features

- **Live Scanner (Hybrid Mode):**
  - **Real-time Object Detection:** Draws polished, responsive bounding boxes around detected objects.
  - **Image Labeling:** Provides detailed textual labels and categories for identified objects in the camera feed.
- **GenAI Scene Descriptions:**
  - **On-Device Analysis:** Uses ML Kit's Generative AI (powered by Gemini Nano) to generate rich, natural language narratives for images selected from the gallery.
  - **Privacy First:** All GenAI processing happens entirely on-device.
- **Modern Material 3 UI:**
  - Dynamic navigation with a Bottom Navigation Bar.
  - Sleek cards and typography following Material You principles.
  - Responsive layout constrained for all screen sizes.

## Technology Stack

- **Languge:** Kotlin 2.1.0
- **UI Framework:** Material 3 & Jetpack Components (ViewBinding, ViewModel, ConstraintLayout)
- **Camera:** CameraX API
- **Machine Learning:** 
  - [ML Kit Object Detection](https://developers.google.com/ml-kit/vision/object-detection)
  - [ML Kit Image Labeling](https://developers.google.com/ml-kit/vision/image-labeling)
  - [ML Kit GenAI Image Description](https://developers.google.com/ml-kit/vision/gen-ai/image-description) (Beta)
- **Async Processing:** Guava ListenableFuture & Kotlin Coroutines

## Requirements

- **Minimum SDK:** 26 (Android 8.0 Oreo)
- **Target SDK:** 34 (Android 14)
- **Compatible Device:** GenAI Image Description requires a device supporting Gemini Nano (e.g., Pixel 8 Pro, Galaxy S24) or a modern emulator with GenAI capabilities enabled.

## Setup & Run

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/saiharish97/MLKIT.git
    ```
2.  **Open in Android Studio:** Open the project folder in the latest version of Android Studio.
3.  **Sync Gradle:** Allow Android Studio to download dependencies (includes ML Kit, Guava, and CameraX).
4.  **Run:** Deployment on a compatible device or emulator.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
