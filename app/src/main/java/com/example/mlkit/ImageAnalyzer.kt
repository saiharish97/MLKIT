package com.example.mlkit

import android.annotation.SuppressLint
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.label.ImageLabel
import com.google.mlkit.vision.label.ImageLabeling
import com.google.mlkit.vision.label.defaults.ImageLabelerOptions
import com.google.mlkit.vision.objects.DetectedObject
import com.google.mlkit.vision.objects.ObjectDetection
import com.google.mlkit.vision.objects.defaults.ObjectDetectorOptions

class ImageAnalyzer(
    private val listener: (List<DetectedObject>, List<ImageLabel>, Int, Int) -> Unit
) : ImageAnalysis.Analyzer {

    // Object Detector for bounding boxes (fast)
    private val objOptions = ObjectDetectorOptions.Builder()
        .setDetectorMode(ObjectDetectorOptions.STREAM_MODE)
        .enableClassification() 
        .build()
    private val objectDetector = ObjectDetection.getClient(objOptions)

    // Image Labeler for detailed descriptions (accurate/comprehensive)
    private val labeler = ImageLabeling.getClient(ImageLabelerOptions.DEFAULT_OPTIONS)

    @SuppressLint("UnsafeOptInUsageError")
    override fun analyze(imageProxy: ImageProxy) {
        val mediaImage = imageProxy.image
        if (mediaImage != null) {
            val rotation = imageProxy.imageInfo.rotationDegrees
            val image = InputImage.fromMediaImage(mediaImage, rotation)
            
            val isRotated = rotation == 90 || rotation == 270
            val width = if (isRotated) imageProxy.height else imageProxy.width
            val height = if (isRotated) imageProxy.width else imageProxy.height

            // Run both detectors in parallel using Google Tasks API
            val taskObj = objectDetector.process(image)
            val taskLabel = labeler.process(image)

            Tasks.whenAllComplete(taskObj, taskLabel)
                .addOnSuccessListener {
                    val objects = taskObj.result ?: emptyList()
                    val labels = taskLabel.result ?: emptyList()
                    listener(objects, labels, width, height)
                }
                .addOnCompleteListener {
                    imageProxy.close()
                }
        } else {
            imageProxy.close()
        }
    }
}
