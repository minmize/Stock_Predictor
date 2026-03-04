package com.neuralnet.builder.results

import android.os.Bundle
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.neuralnet.builder.data.DataHolder
import com.neuralnet.builder.databinding.ActivityResultsBinding
import com.neuralnet.builder.engine.LossType
import com.neuralnet.builder.training.ResultsHolder
import kotlin.math.abs

class ResultsActivity : AppCompatActivity() {

    private lateinit var binding: ActivityResultsBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityResultsBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setupToolbar()
        showResults()
        setupPrediction()
    }

    private fun setupToolbar() {
        setSupportActionBar(binding.toolbar)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        supportActionBar?.title = "Results"
        binding.toolbar.setNavigationOnClickListener { onBackPressedDispatcher.onBackPressed() }
    }

    private fun showResults() {
        val net    = ResultsHolder.network ?: return
        val valSet = DataHolder.valSet

        binding.summaryText.text = net.summary()

        if (valSet.isNotEmpty()) {
            val (loss, acc) = net.evaluate(valSet)
            binding.evalLossText.text = "Validation Loss:     ${"%.6f".format(loss)}"
            if (acc != null) {
                binding.evalAccText.text    = "Validation Accuracy: ${"%.2f".format(acc * 100)}%"
                binding.evalAccText.visibility = View.VISIBLE
            }
        }

        // Show best & final train loss
        if (ResultsHolder.trainLoss.isNotEmpty()) {
            binding.bestLossText.text = "Best Train Loss: ${"%.6f".format(ResultsHolder.trainLoss.min())}"
            binding.finalLossText.text = "Final Train Loss: ${"%.6f".format(ResultsHolder.trainLoss.last())}"
        }

        // Show a few sample predictions
        val samplePreds = buildPredictionSamples(net, valSet.take(10))
        binding.samplePredsText.text = samplePreds
    }

    private fun buildPredictionSamples(
        net: com.neuralnet.builder.engine.NeuralNetwork,
        samples: List<Pair<DoubleArray, DoubleArray>>
    ): String {
        if (samples.isEmpty()) return "No validation samples available."
        val sb = StringBuilder()
        sb.appendLine("Sample Predictions (first ${samples.size} from validation set):")
        sb.appendLine("─".repeat(50))
        samples.forEachIndexed { i, (x, y) ->
            val pred = net.predict(x)
            val predStr   = pred.joinToString(", ") { "%.4f".format(it) }
            val targetStr = y.joinToString(", ")   { "%.4f".format(it) }
            val isRegress = net.lossType == LossType.MSE
            val match = if (isRegress) {
                val err = abs(pred[0] - y[0])
                "err=${"%.4f".format(err)}"
            } else {
                val predClass   = pred.indexOfMax()
                val targetClass = y.indexOfMax()
                if (predClass == targetClass) "✓" else "✗"
            }
            sb.appendLine("#${i+1}: pred=[$predStr] target=[$targetStr] $match")
        }
        return sb.toString()
    }

    // ── Manual Prediction ─────────────────────────────────────────────────────

    private fun setupPrediction() {
        val net = ResultsHolder.network ?: return
        val inputSize = net.layerConfigs.first().neurons

        binding.predictionHint.text =
            "Enter $inputSize comma-separated values (matching your feature columns):"

        binding.predictBtn.setOnClickListener {
            val text = binding.predictionInput.text?.toString()?.trim() ?: ""
            val values = text.split(",").mapNotNull { it.trim().toDoubleOrNull() }

            if (values.size != inputSize) {
                Toast.makeText(
                    this,
                    "Expected $inputSize values, got ${values.size}",
                    Toast.LENGTH_SHORT
                ).show()
                return@setOnClickListener
            }

            val input = values.toDoubleArray()
            val output = net.predict(input)

            val outputStr = when {
                output.size == 1 -> "%.6f".format(output[0])
                else -> output.mapIndexed { i, v -> "Class $i: ${"%.4f".format(v)}" }.joinToString("\n")
            }
            binding.predictionResult.text = "Prediction:\n$outputStr"
            binding.predictionResult.visibility = View.VISIBLE
            binding.resultContainer.visibility  = View.VISIBLE
        }
    }

    private fun DoubleArray.indexOfMax(): Int {
        var best = 0
        for (i in 1 until size) if (this[i] > this[best]) best = i
        return best
    }
}
