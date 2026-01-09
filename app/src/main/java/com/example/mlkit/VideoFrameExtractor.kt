package com.example.mlkit

import android.content.Context
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.util.Log

class VideoFrameExtractor(private val context: Context) {

    /**
     * Extracts a fixed number of frames from a video at regular intervals.
     */
    fun extractFrames(videoUri: Uri, numFrames: Int = 8): List<Bitmap> {
        val frames = mutableListOf<Bitmap>()
        val retriever = MediaMetadataRetriever()
        
        try {
            retriever.setDataSource(context, videoUri)
            val durationStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
            val durationMs = durationStr?.toLong() ?: 0L
            
            if (durationMs <= 0) return emptyList()

            val intervalMs = (durationMs / (numFrames + 1)).toInt()

            for (i in 1..numFrames) {
                val timeUs = (i * intervalMs * 1000).toLong()
                val bitmap = retriever.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
                bitmap?.let {
                    // Resize to the resolution expected by SmolVLM2 (512x512)
                    val resized = Bitmap.createScaledBitmap(it, 512, 512, true)
                    frames.add(resized)
                }
            }
        } catch (e: Exception) {
            Log.e("VideoFrameExtractor", "Error extracting frames: ${e.message}")
        } finally {
            try {
                retriever.release()
            } catch (e: Exception) {}
        }
        
        return frames
    }
}
