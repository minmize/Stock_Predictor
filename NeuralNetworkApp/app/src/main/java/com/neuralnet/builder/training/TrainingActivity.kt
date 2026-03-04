package com.neuralnet.builder.training

import android.content.Intent
import android.graphics.Color
import android.os.Bundle
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.github.mikephil.charting.components.Legend
import com.github.mikephil.charting.data.*
import com.github.mikephil.charting.interfaces.datasets.ILineDataSet
import com.neuralnet.builder.data.DataHolder
import com.neuralnet.builder.databinding.ActivityTrainingBinding
import com.neuralnet.builder.engine.*
import com.neuralnet.builder.results.ResultsActivity
import kotlinx.coroutines.*

class TrainingActivity : AppCompatActivity() {

    private lateinit var binding: ActivityTrainingBinding
    private var trainingJob: Job? = null
    private var network: NeuralNetwork? = null

    private val trainLossEntries = mutableListOf<Entry>()
    private val valLossEntries   = mutableListOf<Entry>()
    private val trainAccEntries  = mutableListOf<Entry>()
    private val valAccEntries    = mutableListOf<Entry>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityTrainingBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setupToolbar()
        setupCharts()
        setupButtons()
        buildNetwork()
    }

    override fun onDestroy() {
        super.onDestroy()
        trainingJob?.cancel()
    }

    // ── Setup ─────────────────────────────────────────────────────────────────

    private fun setupToolbar() {
        setSupportActionBar(binding.toolbar)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        supportActionBar?.title = "Training"
        binding.toolbar.setNavigationOnClickListener {
            trainingJob?.cancel()
            onBackPressedDispatcher.onBackPressed()
        }
    }

    private fun setupCharts() {
        listOf(binding.lossChart, binding.accChart).forEach { chart ->
            chart.apply {
                description.isEnabled = false
                setTouchEnabled(true)
                isDragEnabled = true
                setScaleEnabled(true)
                setPinchZoom(true)
                setBackgroundColor(Color.TRANSPARENT)
                xAxis.textColor = 0xFFCCCCCC.toInt()
                axisLeft.textColor = 0xFFCCCCCC.toInt()
                axisRight.isEnabled = false
                legend.textColor = 0xFFCCCCCC.toInt()
                legend.form = Legend.LegendForm.LINE
            }
        }
    }

    private fun setupButtons() {
        binding.startBtn.setOnClickListener { startTraining() }
        binding.stopBtn.setOnClickListener  { stopTraining() }
        binding.resultsBtn.setOnClickListener { navigateToResults() }

        binding.stopBtn.isEnabled = false
        binding.resultsBtn.isEnabled = false
    }

    // ── Build Network ─────────────────────────────────────────────────────────

    private fun buildNetwork() {
        val sizes  = intent.getIntArrayExtra("layer_sizes") ?: return
        val acts   = intent.getStringArrayExtra("layer_acts") ?: return
        val drops  = intent.getFloatArrayExtra("layer_drops") ?: FloatArray(sizes.size)
        val lr     = intent.getDoubleExtra("learning_rate", 0.001)
        val loss   = LossType.valueOf(intent.getStringExtra("loss_type") ?: "MSE")

        val layers = sizes.indices.map { i ->
            LayerConfig(sizes[i], ActivationType.valueOf(acts[i]), drops[i])
        }
        network = NeuralNetwork(layers, loss, lr)

        binding.networkSummaryText.text = network!!.summary()
        binding.epochsTotal.text = "/ ${intent.getIntExtra("epochs", 50)}"
    }

    // ── Training ──────────────────────────────────────────────────────────────

    private fun startTraining() {
        val net       = network ?: return
        val trainData = DataHolder.trainSet
        val valData   = DataHolder.valSet
        val epochs    = intent.getIntExtra("epochs", 50)
        val batchSize = intent.getIntExtra("batch_size", 32)

        if (trainData.isEmpty()) {
            binding.logText.append("ERROR: No training data found. Go back and load data.\n")
            return
        }

        // Input size check
        val expectedInput = net.layerConfigs.first().neurons
        if (trainData.first().first.size != expectedInput) {
            binding.logText.append(
                "ERROR: Network input size ($expectedInput) doesn't match feature count " +
                "(${trainData.first().first.size}). Go back and adjust.\n"
            )
            return
        }

        trainLossEntries.clear(); valLossEntries.clear()
        trainAccEntries.clear();  valAccEntries.clear()

        binding.startBtn.isEnabled   = false
        binding.stopBtn.isEnabled    = true
        binding.resultsBtn.isEnabled = false
        binding.progressBar.max      = epochs

        trainingJob = lifecycleScope.launch(Dispatchers.Default) {
            var shouldContinue = true

            net.train(trainData, valData, epochs, batchSize) { result ->
                shouldContinue = isActive
                if (!shouldContinue) return@train false

                // Post UI updates on main thread
                launch(Dispatchers.Main) { updateUI(result, epochs) }

                // Small yield to prevent UI freeze
                Thread.sleep(1)
                shouldContinue
            }

            launch(Dispatchers.Main) { onTrainingComplete() }
        }
    }

    private fun updateUI(result: EpochResult, totalEpochs: Int) {
        // Progress
        binding.epochProgress.text = result.epoch.toString()
        binding.progressBar.progress = result.epoch

        // Loss chart
        trainLossEntries.add(Entry(result.epoch.toFloat(), result.trainLoss.toFloat()))
        result.valLoss?.let { valLossEntries.add(Entry(result.epoch.toFloat(), it.toFloat())) }

        val lossSets = mutableListOf<ILineDataSet>(
            buildDataSet(trainLossEntries, "Train Loss", Color.parseColor("#4FC3F7"))
        )
        if (valLossEntries.isNotEmpty())
            lossSets.add(buildDataSet(valLossEntries, "Val Loss", Color.parseColor("#EF9A9A")))

        binding.lossChart.data = LineData(lossSets)
        binding.lossChart.invalidate()

        // Accuracy chart
        result.trainAccuracy?.let { trainAccEntries.add(Entry(result.epoch.toFloat(), (it * 100).toFloat())) }
        result.valAccuracy?.let   { valAccEntries.add(Entry(result.epoch.toFloat(),   (it * 100).toFloat())) }

        if (trainAccEntries.isNotEmpty()) {
            val accSets = mutableListOf<ILineDataSet>(
                buildDataSet(trainAccEntries, "Train Acc %", Color.parseColor("#A5D6A7"))
            )
            if (valAccEntries.isNotEmpty())
                accSets.add(buildDataSet(valAccEntries, "Val Acc %", Color.parseColor("#FFCC80")))
            binding.accChart.data = LineData(accSets)
            binding.accChart.invalidate()
            binding.accCard.visibility = View.VISIBLE
        }

        // Live metrics
        binding.metricTrainLoss.text = "Train Loss: ${"%.5f".format(result.trainLoss)}"
        result.valLoss?.let    { binding.metricValLoss.text = "Val Loss:   ${"%.5f".format(it)}" }
        result.trainAccuracy?.let { binding.metricTrainAcc.text = "Train Acc:  ${"%.2f".format(it * 100)}%" }
        result.valAccuracy?.let   { binding.metricValAcc.text   = "Val Acc:    ${"%.2f".format(it * 100)}%" }

        // Log (every 5 epochs or first/last)
        if (result.epoch % 5 == 0 || result.epoch == 1 || result.epoch == totalEpochs) {
            val accText = result.trainAccuracy?.let { " | Acc: ${"%.1f".format(it * 100)}%" } ?: ""
            binding.logText.append("Epoch ${result.epoch}/$totalEpochs — Loss: ${"%.5f".format(result.trainLoss)}$accText\n")
            binding.logScroll.post { binding.logScroll.fullScroll(android.view.View.FOCUS_DOWN) }
        }
    }

    private fun buildDataSet(entries: List<Entry>, label: String, color: Int): LineDataSet {
        return LineDataSet(entries, label).apply {
            this.color = color
            setCircleColor(color)
            lineWidth = 2f
            circleRadius = 2f
            setDrawValues(false)
            mode = LineDataSet.Mode.CUBIC_BEZIER
        }
    }

    private fun stopTraining() {
        trainingJob?.cancel()
        trainingJob = null
        binding.startBtn.isEnabled   = true
        binding.stopBtn.isEnabled    = false
        binding.resultsBtn.isEnabled = true
        binding.logText.append("⏹ Training stopped.\n")
    }

    private fun onTrainingComplete() {
        binding.startBtn.isEnabled   = true
        binding.stopBtn.isEnabled    = false
        binding.resultsBtn.isEnabled = true
        binding.logText.append("✅ Training complete!\n")
        binding.logScroll.post { binding.logScroll.fullScroll(android.view.View.FOCUS_DOWN) }

        // Save to holder for results screen
        ResultsHolder.network = network
        ResultsHolder.trainLoss = trainLossEntries.map { it.y.toDouble() }
        ResultsHolder.valLoss   = valLossEntries.map   { it.y.toDouble() }
    }

    private fun navigateToResults() {
        if (network == null) return
        ResultsHolder.network = network
        startActivity(Intent(this, ResultsActivity::class.java))
    }
}

/** Pass training artifacts to the Results screen. */
object ResultsHolder {
    var network: NeuralNetwork? = null
    var trainLoss: List<Double> = emptyList()
    var valLoss: List<Double>   = emptyList()
}
