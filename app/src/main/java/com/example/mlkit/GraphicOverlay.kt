package com.example.mlkit

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import com.google.mlkit.vision.objects.DetectedObject

class GraphicOverlay(context: Context, attrs: AttributeSet?) : View(context, attrs) {

    private val lock = Any()
    private var imageWidth: Int = 0
    private var imageHeight: Int = 0
    private var scaleX: Float = 1.0f
    private var scaleY: Float = 1.0f
    
    // Material 3 style Cyan/Primary hybrid
    private val boxPaint = Paint().apply {
        color = Color.parseColor("#6750A4") // Primary Material 3 color
        style = Paint.Style.STROKE
        strokeWidth = 6.0f
        isAntiAlias = true
    }
    
    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 34.0f
        style = Paint.Style.FILL
        isAntiAlias = true
        setShadowLayer(4f, 2f, 2f, Color.BLACK)
    }
    
    private val textBackgroundPaint = Paint().apply {
        color = Color.parseColor("#996750A4") // Semi-transparent Primary
        style = Paint.Style.FILL
        isAntiAlias = true
    }
    
    private var detectedObjects = mutableListOf<DetectedObject>()

    fun setSourceInfo(width: Int, height: Int) {
        synchronized(lock) {
            imageWidth = width
            imageHeight = height
            postInvalidate()
        }
    }

    fun updateObjects(objects: List<DetectedObject>) {
        synchronized(lock) {
            detectedObjects.clear()
            detectedObjects.addAll(objects)
            postInvalidate()
        }
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        synchronized(lock) {
            if (imageWidth == 0 || imageHeight == 0) return

            scaleX = width.toFloat() / imageWidth
            scaleY = height.toFloat() / imageHeight

            for (obj in detectedObjects) {
                val boundingBox = obj.boundingBox
                val rect = RectF(
                    boundingBox.left * scaleX,
                    boundingBox.top * scaleY,
                    boundingBox.right * scaleX,
                    boundingBox.bottom * scaleY
                )
                
                // Draw rounded bounding box
                canvas.drawRoundRect(rect, 16f, 16f, boxPaint)

                // Draw label
                val labelText = if (obj.labels.isNotEmpty()) {
                    obj.labels[0].text
                } else {
                    "Object"
                }

                val padding = 12f
                val textWidth = textPaint.measureText(labelText)
                val textHeight = 40f
                
                val labelRect = RectF(
                    rect.left,
                    rect.top - textHeight - padding,
                    rect.left + textWidth + (padding * 2),
                    rect.top
                )
                
                // Draw rounded label background
                canvas.drawRoundRect(labelRect, 8f, 8f, textBackgroundPaint)
                canvas.drawText(labelText, rect.left + padding, rect.top - (padding), textPaint)
            }
        }
    }
}
