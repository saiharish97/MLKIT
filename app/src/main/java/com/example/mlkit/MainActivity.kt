package com.example.mlkit

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.mlkit.databinding.ActivityMainBinding
import com.google.mlkit.genai.imagedescription.ImageDescription
import com.google.mlkit.genai.imagedescription.ImageDescriptionRequest
import com.google.mlkit.genai.imagedescription.ImageDescriber
import com.google.mlkit.genai.imagedescription.ImageDescriberOptions
import com.google.mlkit.genai.imagedescription.ImageDescriptionResult
import com.google.common.util.concurrent.FutureCallback
import com.google.common.util.concurrent.Futures
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var viewBinding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService
    
    private var imageDescriber: ImageDescriber? = null

    private val selectImageLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        uri?.let {
            val bitmap = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                val source = ImageDecoder.createSource(contentResolver, it)
                ImageDecoder.decodeBitmap(source)
            } else {
                MediaStore.Images.Media.getBitmap(contentResolver, it)
            }
            viewBinding.selectedImageView.setImageBitmap(bitmap)
            generateDescription(bitmap)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)

        setupNavigation()
        setupGenAi()

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }

        cameraExecutor = Executors.newSingleThreadExecutor()
    }

    private fun setupGenAi() {
        val options = ImageDescriberOptions.builder(this).build()
        imageDescriber = ImageDescription.getClient(options)
    }

    private fun generateDescription(bitmap: Bitmap) {
        val describer = imageDescriber ?: return
        viewBinding.textGenAiResult.text = "Generating description..."
        
        val request = ImageDescriptionRequest.builder(bitmap).build()
        val future = describer.runInference(request)
        
        Futures.addCallback(
            future,
            object : FutureCallback<ImageDescriptionResult> {
                override fun onSuccess(result: ImageDescriptionResult?) {
                    runOnUiThread {
                        viewBinding.textGenAiResult.text = result?.description
                    }
                }

                override fun onFailure(t: Throwable) {
                    runOnUiThread {
                        Log.e(TAG, "GenAI Inference failed", t)
                        viewBinding.textGenAiResult.text = "Error: ${t.message}"
                    }
                }
            },
            ContextCompat.getMainExecutor(this@MainActivity)
        )
    }

    private fun setupNavigation() {
        viewBinding.bottomNavigation.setOnItemSelectedListener { item ->
            when(item.itemId) {
                R.id.nav_scanner -> {
                    viewBinding.paneScanner.visibility = View.VISIBLE
                    viewBinding.paneDescriptions.visibility = View.GONE
                    viewBinding.toolbar.title = "AI Video Vision"
                    true
                }
                R.id.nav_descriptions -> {
                    viewBinding.paneScanner.visibility = View.GONE
                    viewBinding.paneDescriptions.visibility = View.VISIBLE
                    viewBinding.toolbar.title = "Scene Descriptions"
                    true
                }
                R.id.nav_settings -> {
                    Toast.makeText(this, "Settings feature coming soon!", Toast.LENGTH_SHORT).show()
                    true
                }
                else -> false
            }
        }
        
        viewBinding.fabSelectImage.setOnClickListener {
            selectImageLauncher.launch("image/*")
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(viewBinding.viewFinder.surfaceProvider)
            }
            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor, ImageAnalyzer { objects, labels, width, height ->
                        runOnUiThread {
                            viewBinding.graphicOverlay.setSourceInfo(width, height)
                            viewBinding.graphicOverlay.updateObjects(objects)
                            val detailedDescription = if (labels.isNotEmpty()) {
                                labels.joinToString(", ") { it.text }
                            } else if (objects.isNotEmpty()) {
                                "Detected ${objects.size} objects"
                            } else {
                                "Analyzing scene..."
                            }
                            viewBinding.textPrediction.text = detailedDescription
                        }
                    })
                }
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalyzer)
            } catch (exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) startCamera()
            else {
                Toast.makeText(this, "Permissions not granted.", Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        imageDescriber?.close()
    }

    companion object {
        private const val TAG = "CameraXApp"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
}
