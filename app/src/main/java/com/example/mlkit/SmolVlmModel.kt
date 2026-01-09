package com.example.mlkit

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.util.Collections
import org.json.JSONObject
import org.json.JSONArray

class SmolVlmModel(private val context: Context) {

    private val ortEnv = OrtEnvironment.getEnvironment()
    private var visionSession: OrtSession? = null
    private var embedSession: OrtSession? = null
    private var decoderSession: OrtSession? = null

    // Model configuration (verified from SmolVLM2-256M config)
    private val numHiddenLayers = 30 
    private val numKeyValueHeads = 3 
    private val headDim = 64 
    private val tokensPerImage = 64 
    private val hiddenSize = 576 
    private val imageTokenId = 49190L // <image>

    fun loadModels() {
        try {
            visionSession = createSession("vision_encoder.onnx")
            embedSession = createSession("embed_tokens.onnx")
            decoderSession = createSession("decoder_model_merged.onnx")
            
            inspectSession(visionSession, "Vision")
            inspectSession(embedSession, "Embed")
            inspectSession(decoderSession, "Decoder")
            
            Log.d("SmolVlmModel", "Models loaded successfully")
        } catch (e: Exception) {
            Log.e("SmolVlmModel", "Error loading models: ${e.message}")
        }
    }

    private fun createSession(modelName: String): OrtSession {
        val modelFile = java.io.File(context.cacheDir, modelName)
        if (!modelFile.exists()) {
            Log.d("SmolVlmModel", "Copying $modelName to cache...")
            context.assets.open("onnx/$modelName").use { input ->
                modelFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
        }
        return ortEnv.createSession(modelFile.absolutePath)
    }

    suspend fun analyzeVideo(frames: List<Bitmap>, prompt: String, onProgress: (String) -> Unit): String {
        if (visionSession == null || embedSession == null || decoderSession == null) {
            return "Models not loaded. Please ensure ONNX files are in assets/onnx/"
        }

        try {
            // 1. Tokenization
            val tokenizer = JsonTokenizer(context) 
            val tokens = tokenizer.encodePrompt(prompt, frames.size)
            var currentInputIds = tokens.toLongArray()
            var currentAttentionMask = LongArray(currentInputIds.size) { 1L }
            var currentPositionIds = LongArray(currentInputIds.size) { (it + 1).toLong() }
            
            // 2. Vision Encoding
            onProgress("Encoding vision features...")
            val pixelValues = preparePixelValues(frames)
            val pixelAttentionMask = preparePixelAttentionMask(frames.size)
            val visionInputs = mapOf(
                "pixel_values" to pixelValues,
                "pixel_attention_mask" to pixelAttentionMask
            )
            val visionOutput = visionSession?.run(visionInputs)
            val imageFeatures = visionOutput?.get(0) as OnnxTensor
            Log.d("SmolVlmModel", "Image features shape: ${imageFeatures.info.shape.contentToString()}")

            // 3. Generation Loop
            onProgress("Starting generation...")
            val generatedTokens = mutableListOf<Long>()
            var pastKeyValues = prepareInitialPastKeyValues()
            var imageFeaturesConsumed = false

            val maxNewTokens = 100
            for (step in 0 until maxNewTokens) {
                // a. Embedding
                val inputIdsTensor = OnnxTensor.createTensor(ortEnv, LongBuffer.wrap(currentInputIds), longArrayOf(1L, currentInputIds.size.toLong()))
                val embedOutput = embedSession?.run(mapOf("input_ids" to inputIdsTensor))
                val embedTensor = embedOutput?.get(0) as OnnxTensor
                
                // CRITICAL: Copy embedding to a fresh buffer. 
                // ONNX Runtime's direct buffer might be larger than the tensor shape, 
                // causing a "buffer size mismatch" error in the next stage.
                val embedsSize = currentInputIds.size * hiddenSize
                val inputsEmbedsBuffer = FloatBuffer.allocate(embedsSize)
                val rawEmbeds = embedTensor.floatBuffer
                rawEmbeds.rewind()
                rawEmbeds.limit(embedsSize)
                inputsEmbedsBuffer.put(rawEmbeds)
                inputsEmbedsBuffer.rewind()
                
                // b. Vision feature merging (Only on first step)
                if (!imageFeaturesConsumed) {
                    mergeVisionFeatures(currentInputIds, inputsEmbedsBuffer, imageFeatures)
                    imageFeaturesConsumed = true
                }

                // c. Decoder step
                val inputsEmbedsTensor = OnnxTensor.createTensor(ortEnv, inputsEmbedsBuffer, longArrayOf(1L, currentInputIds.size.toLong(), hiddenSize.toLong())) 
                val attentionMaskTensor = OnnxTensor.createTensor(ortEnv, LongBuffer.wrap(currentAttentionMask), longArrayOf(1L, currentAttentionMask.size.toLong()))
                val positionIdsTensor = OnnxTensor.createTensor(ortEnv, LongBuffer.wrap(currentPositionIds), longArrayOf(1L, currentPositionIds.size.toLong()))

                val decoderInputs = mutableMapOf<String, OnnxTensor>(
                    "inputs_embeds" to inputsEmbedsTensor,
                    "attention_mask" to attentionMaskTensor,
                    "position_ids" to positionIdsTensor
                )
                decoderInputs.putAll(pastKeyValues)

                val decoderOutput = decoderSession?.run(decoderInputs)
                val logits = decoderOutput?.get(0) as OnnxTensor
                
                // d. Token sampling (Greedy)
                val nextToken = sampleNextToken(logits)
                generatedTokens.add(nextToken)
                
                if (nextToken == 49279L || nextToken == 2L) break // <end_of_utterance> or <|im_end|>

                // e. Update state for next step
                pastKeyValues = updatePastKeyValues(decoderOutput)
                currentInputIds = longArrayOf(nextToken)
                currentAttentionMask = longArrayOf(1L)
                currentPositionIds = longArrayOf(currentPositionIds.last() + 1)

                val partialText = tokenizer.decode(generatedTokens)
                onProgress("Generating: $partialText")
            }

            return tokenizer.decode(generatedTokens)

        } catch (e: Exception) {
            Log.e("SmolVlmModel", "Inference error: ${e.message}")
            return "Error: ${e.message}"
        }
    }

    private fun prepareInitialPastKeyValues(): Map<String, OnnxTensor> {
        val kvMap = mutableMapOf<String, OnnxTensor>()
        for (layer in 0 until numHiddenLayers) {
            for (kv in listOf("key", "value")) {
                val buffer = FloatBuffer.allocate(0)
                val tensor = OnnxTensor.createTensor(ortEnv, buffer, longArrayOf(1L, numKeyValueHeads.toLong(), 0L, headDim.toLong()))
                kvMap["past_key_values.$layer.$kv"] = tensor
            }
        }
        return kvMap
    }

    private fun mergeVisionFeatures(inputIds: LongArray, embeds: FloatBuffer, visionFeatures: OnnxTensor) {
        val visionBuffer = visionFeatures.floatBuffer
        val visionSeqLen = visionFeatures.info.shape[1].toInt()
        
        var visionIdx = 0
        for (i in inputIds.indices) {
            // Check for image token to inject vision features
            if (inputIds[i] == imageTokenId && visionIdx < visionSeqLen) {
                for (h in 0 until hiddenSize) {
                    embeds.put(i * hiddenSize + h, visionBuffer.get(visionIdx * hiddenSize + h))
                }
                visionIdx++
            }
        }
        embeds.rewind()
    }

    private fun sampleNextToken(logits: OnnxTensor): Long {
        val buffer = logits.floatBuffer
        val seqLen = logits.info.shape[1].toInt()
        val vocabSize = logits.info.shape[2].toInt()
        
        var maxIdx = 0L
        var maxVal = Float.NEGATIVE_INFINITY
        
        // Only look at the last token in the sequence
        val offset = (seqLen - 1) * vocabSize
        for (i in 0 until vocabSize) {
            val v = buffer.get(offset + i)
            if (v > maxVal) {
                maxVal = v
                maxIdx = i.toLong()
            }
        }
        return maxIdx
    }

    private fun updatePastKeyValues(result: OrtSession.Result?): Map<String, OnnxTensor> {
        val nextKv = mutableMapOf<String, OnnxTensor>()
        // result[0] is logits, result[1..N] are the new KVs
        for (layer in 0 until numHiddenLayers) {
            nextKv["past_key_values.$layer.key"] = result?.get(1 + layer * 2) as OnnxTensor
            nextKv["past_key_values.$layer.value"] = result?.get(2 + layer * 2) as OnnxTensor
        }
        return nextKv
    }

    class JsonTokenizer(context: Context) {
        private val idToToken = mutableMapOf<Long, String>()
        
        init {
            try {
                val jsonString = context.assets.open("onnx/tokenizer.json").bufferedReader().use { it.readText() }
                val root = JSONObject(jsonString)
                
                // 1. Added tokens (Special tokens like <image>, <|im_start|>)
                val addedTokens = root.getJSONArray("added_tokens")
                for (i in 0 until addedTokens.length()) {
                    val obj = addedTokens.getJSONObject(i)
                    idToToken[obj.getLong("id")] = obj.getString("content")
                }
                
                // 2. Main vocabulary
                val model = root.getJSONObject("model")
                val vocab = model.getJSONObject("vocab")
                val keys = vocab.keys()
                while (keys.hasNext()) {
                    val token = keys.next()
                    idToToken[vocab.getLong(token)] = token
                }
                Log.d("SmolVlmModel", "Loaded ${idToToken.size} tokens into vocabulary")
            } catch (e: Exception) {
                Log.e("SmolVlmModel", "Failed to load tokenizer: ${e.message}")
            }
        }

        fun encodePrompt(prompt: String, numImages: Int): List<Long> {
            val tokens = mutableListOf<Long>()
            // Template: <|im_start|>User:<image>...<image>{prompt}<end_of_utterance>\nAssistant:

            // 1. <|im_start|>User:
            tokens.add(1L) // <|im_start|>
            tokens.addAll(encodeText("User:"))

            // 2. <image> tokens
            tokens.add(49189L) // <fake_token_around_image>
            repeat(numImages * 64) { tokens.add(49190L) }
            tokens.add(49189L) // <fake_token_around_image>
            
            // 3. User prompt
            tokens.addAll(encodeText(prompt))
            tokens.add(49279L) // <end_of_utterance>
            tokens.addAll(encodeText("\n"))
            
            // 4. Assistant:
            tokens.addAll(encodeText(" Assistant:"))
            
            Log.d("SmolVlmModel", "Encoded prompt into ${tokens.size} tokens")
            return tokens
        }

        private fun encodeText(text: String): List<Long> {
            val result = mutableListOf<Long>()
            // Map common words/chars to tokens. 
            // Real BPE would use a prefix tree. Here we do a simple greedy match.
            // For SmolVLM2/Llama, words often start with ' ' (mapped to 'Ġ')
            var i = 0
            val tokenToId = idToToken.entries.associate { it.value to it.key }
            
            // Pre-process text to match BPE representation (space -> Ġ, newline -> Ċ)
            val processed = text.replace(" ", "Ġ").replace("\n", "Ċ")
            
            while (i < processed.length) {
                var match: String? = null
                var matchId: Long = -1
                
                // Try longest match from i
                for (len in minOf(processed.length - i, 20) downTo 1) {
                    val sub = processed.substring(i, i + len)
                    val id = tokenToId[sub]
                    if (id != null) {
                        match = sub
                        matchId = id
                        break
                    }
                }
                
                if (match != null) {
                    result.add(matchId)
                    i += match.length
                } else {
                    // Fallback to byte tokens if no match (e.g. <0xXX>)
                    val char = processed[i]
                    result.add(char.toLong()) // Very naive fallback
                    i++
                }
            }
            return result
        }

        fun decode(tokens: List<Long>): String {
            val sb = StringBuilder()
            for (id in tokens) {
                val token = idToToken[id] ?: ""
                // Byte-level BPE cleanup 
                val cleanToken = token.replace("Ġ", " ").replace("Ċ", "\n")
                if (cleanToken.startsWith("<0x") && cleanToken.endsWith(">")) {
                    try {
                        val hex = cleanToken.substring(3, 5)
                        sb.append(hex.toInt(16).toChar())
                    } catch (e: Exception) {}
                } else {
                    sb.append(cleanToken)
                }
            }
            return sb.toString().trim()
        }
    }

    private fun preparePixelValues(frames: List<Bitmap>): OnnxTensor {
        // Flatten bitmaps into float buffer (N, C, H, W)
        val numFrames = frames.size
        val channels = 3
        val height = 512
        val width = 512
        val floatBuffer = FloatBuffer.allocate(numFrames * channels * height * width)
        
        for (frame in frames) {
            // Planar layout (CHW): All R, then all G, then all B
            val pixels = IntArray(width * height)
            frame.getPixels(pixels, 0, width, 0, 0, width, height)
            
            // Channel R
            for (p in pixels) {
                floatBuffer.put(((p shr 16 and 0xFF) / 255.0f - 0.5f) / 0.5f)
            }
            // Channel G
            for (p in pixels) {
                floatBuffer.put(((p shr 8 and 0xFF) / 255.0f - 0.5f) / 0.5f)
            }
            // Channel B
            for (p in pixels) {
                floatBuffer.put(((p and 0xFF) / 255.0f - 0.5f) / 0.5f)
            }
        }
        floatBuffer.rewind()
        return OnnxTensor.createTensor(ortEnv, floatBuffer, longArrayOf(1L, numFrames.toLong(), channels.toLong(), height.toLong(), width.toLong()))
    }

    private fun preparePixelAttentionMask(numFrames: Int): OnnxTensor {
        val height = 512
        val width = 512
        val size = numFrames * height * width
        val maskBuffer = ByteBuffer.allocateDirect(size)
        maskBuffer.order(ByteOrder.nativeOrder())
        for (i in 0 until size) {
            maskBuffer.put(1.toByte()) // 1 for true
        }
        maskBuffer.rewind()
        return OnnxTensor.createTensor(ortEnv, maskBuffer, longArrayOf(1L, numFrames.toLong(), height.toLong(), width.toLong()), OnnxJavaType.BOOL)
    }

    private fun inspectSession(session: OrtSession?, name: String) {
        session?.inputNames?.forEach { inputName ->
            val info = session.inputInfo[inputName]
            Log.d("SmolVlmModel", "$name Input: $inputName, Info: $info")
        }
    }

    fun close() {
        visionSession?.close()
        embedSession?.close()
        decoderSession?.close()
    }
}
