package com.neuralnet.builder.data

import android.app.Activity
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.view.View
import android.widget.ArrayAdapter
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.chip.Chip
import com.neuralnet.builder.databinding.ActivityDataInputBinding
import com.neuralnet.builder.training.TrainingActivity

class DataInputActivity : AppCompatActivity() {

    private lateinit var binding: ActivityDataInputBinding
    private var rawDataset: RawDataset? = null
    private var fileUri: Uri? = null

    // ── File Picker ────────────────────────────────────────────────────────────

    private val filePicker = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) {
            fileUri = uri
            contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION)
            loadFile(uri)
        }
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityDataInputBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setupToolbar()
        setupFilePicker()
        setupSpinners()
        setupSliders()
        setupNextButton()

        showEmptyState()
    }

    private fun setupToolbar() {
        setSupportActionBar(binding.toolbar)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        supportActionBar?.title = "Load Data"
        binding.toolbar.setNavigationOnClickListener { onBackPressedDispatcher.onBackPressed() }
    }

    // ── File Loading ──────────────────────────────────────────────────────────

    private fun setupFilePicker() {
        binding.uploadBtn.setOnClickListener {
            filePicker.launch(arrayOf(
                "text/csv", "text/plain", "text/comma-separated-values",
                "application/csv", "application/octet-stream"
            ))
        }
        binding.sampleDataBtn.setOnClickListener { loadSampleData() }
    }

    private fun loadFile(uri: Uri) {
        binding.loadingIndicator.visibility = View.VISIBLE
        binding.dataCard.visibility = View.GONE

        Thread {
            val result = DataLoader.loadFromUri(this, uri)
            runOnUiThread {
                binding.loadingIndicator.visibility = View.GONE
                if (result == null) {
                    Toast.makeText(this, "Could not read file. Make sure it's a CSV.", Toast.LENGTH_LONG).show()
                    return@runOnUiThread
                }
                rawDataset = result
                onDataLoaded(result)
            }
        }.start()
    }

    private fun loadSampleData() {
        // Generate XOR dataset as sample
        val headers = listOf("x1", "x2", "xor")
        val rows = mutableListOf<List<String>>()
        val rng = kotlin.random.Random(42)
        for (i in 0..199) {
            val x1 = rng.nextInt(2)
            val x2 = rng.nextInt(2)
            rows.add(listOf(x1.toString(), x2.toString(), (x1 xor x2).toString()))
        }
        rawDataset = RawDataset(headers, rows)
        onDataLoaded(rawDataset!!)
        Toast.makeText(this, "XOR sample dataset loaded (200 rows)", Toast.LENGTH_SHORT).show()
    }

    private fun onDataLoaded(raw: RawDataset) {
        binding.dataCard.visibility = View.VISIBLE
        binding.emptyState.visibility = View.GONE
        binding.columnSelectionCard.visibility = View.VISIBLE

        val fileName = fileUri?.lastPathSegment ?: "Sample Data"
        binding.fileNameText.text = fileName
        binding.dataStatsText.text = "${raw.rowCount} rows × ${raw.colCount} columns"

        // Preview first 5 rows
        binding.previewText.text = buildPreviewText(raw)

        // Column stats
        showColumnStats(raw)

        // Populate column selectors
        populateColumnChips(raw.headers)
    }

    private fun buildPreviewText(raw: RawDataset): String {
        val sb = StringBuilder()
        sb.append(raw.headers.joinToString(" | "))
        sb.append("\n")
        sb.append("─".repeat(60))
        sb.append("\n")
        raw.rows.take(5).forEach { row ->
            sb.append(row.joinToString(" | ") { it.take(8).padStart(8) })
            sb.append("\n")
        }
        if (raw.rowCount > 5) sb.append("  ... ${raw.rowCount - 5} more rows")
        return sb.toString()
    }

    private fun showColumnStats(raw: RawDataset) {
        val stats = DataProcessor.allColumnStats(raw)
        val sb = StringBuilder()
        stats.forEach { (name, st) ->
            sb.appendLine("$name: mean=${"%+.2f".format(st.mean)}, std=${"%+.2f".format(st.std)}, min=${"%+.2f".format(st.min)}, max=${"%+.2f".format(st.max)}")
        }
        binding.statsText.text = sb.toString().ifBlank { "No numeric columns detected" }
    }

    // ── Column Selection ──────────────────────────────────────────────────────

    private fun populateColumnChips(headers: List<String>) {
        binding.featureChipGroup.removeAllViews()
        binding.targetChipGroup.removeAllViews()

        headers.forEachIndexed { index, name ->
            // Feature chip
            val fChip = Chip(this).apply {
                text = name
                isCheckable = true
                isChecked = index < headers.size - 1  // All except last = features by default
            }
            binding.featureChipGroup.addView(fChip)

            // Target chip
            val tChip = Chip(this).apply {
                text = name
                isCheckable = true
                isChecked = index == headers.size - 1  // Last column = target by default
            }
            binding.targetChipGroup.addView(tChip)
        }
    }

    private fun getSelectedFeatures(): List<String> {
        val raw = rawDataset ?: return emptyList()
        return raw.headers.filterIndexed { i, _ ->
            (binding.featureChipGroup.getChildAt(i) as? Chip)?.isChecked == true
        }
    }

    private fun getSelectedTargets(): List<String> {
        val raw = rawDataset ?: return emptyList()
        return raw.headers.filterIndexed { i, _ ->
            (binding.targetChipGroup.getChildAt(i) as? Chip)?.isChecked == true
        }
    }

    // ── Spinners & Sliders ────────────────────────────────────────────────────

    private fun setupSpinners() {
        val normOptions = NormalisationMethod.values().map { it.displayName }
        binding.normSpinner.setAdapter(ArrayAdapter(this, android.R.layout.simple_dropdown_item_1line, normOptions))
        binding.normSpinner.setText(NormalisationMethod.MIN_MAX.displayName, false)

        val missingOptions = MissingValueStrategy.values().map { it.displayName }
        binding.missingSpinner.setAdapter(ArrayAdapter(this, android.R.layout.simple_dropdown_item_1line, missingOptions))
        binding.missingSpinner.setText(MissingValueStrategy.DROP_ROW.displayName, false)

        val outlierOptions = OutlierMethod.values().map { it.displayName }
        binding.outlierSpinner.setAdapter(ArrayAdapter(this, android.R.layout.simple_dropdown_item_1line, outlierOptions))
        binding.outlierSpinner.setText(OutlierMethod.NONE.displayName, false)

        val samplingOptions = SamplingMethod.values().map { it.displayName }
        binding.samplingSpinner.setAdapter(ArrayAdapter(this, android.R.layout.simple_dropdown_item_1line, samplingOptions))
        binding.samplingSpinner.setText(SamplingMethod.ALL.displayName, false)

        binding.samplingSpinner.setOnItemClickListener { _, _, idx, _ ->
            val method = SamplingMethod.values()[idx]
            binding.sampleSizeCard.visibility =
                if (method != SamplingMethod.ALL) View.VISIBLE else View.GONE
        }
    }

    private fun setupSliders() {
        binding.trainSplitSlider.addOnChangeListener { _, value, _ ->
            binding.trainSplitLabel.text = "Train/Val Split: ${"%.0f".format(value * 100)}% / ${"%.0f".format((1 - value) * 100)}%"
        }
        binding.trainSplitSlider.value = 0.8f
        binding.trainSplitLabel.text = "Train/Val Split: 80% / 20%"
    }

    // ── Next ──────────────────────────────────────────────────────────────────

    private fun setupNextButton() {
        binding.nextBtn.setOnClickListener {
            if (validateAndProceed()) proceedToTraining()
        }
    }

    private fun validateAndProceed(): Boolean {
        if (rawDataset == null) {
            Toast.makeText(this, "Please load a data file first", Toast.LENGTH_SHORT).show()
            return false
        }
        if (getSelectedFeatures().isEmpty()) {
            Toast.makeText(this, "Select at least one feature column", Toast.LENGTH_SHORT).show()
            return false
        }
        if (getSelectedTargets().isEmpty()) {
            Toast.makeText(this, "Select a target column", Toast.LENGTH_SHORT).show()
            return false
        }
        return true
    }

    private fun proceedToTraining() {
        val raw = rawDataset ?: return
        val features = getSelectedFeatures()
        val targets  = getSelectedTargets()

        val normText = binding.normSpinner.text.toString()
        val normMethod = NormalisationMethod.values().firstOrNull { it.displayName == normText }
            ?: NormalisationMethod.MIN_MAX

        val missingText = binding.missingSpinner.text.toString()
        val missingMethod = MissingValueStrategy.values().firstOrNull { it.displayName == missingText }
            ?: MissingValueStrategy.DROP_ROW

        val outlierText = binding.outlierSpinner.text.toString()
        val outlierMethod = OutlierMethod.values().firstOrNull { it.displayName == outlierText }
            ?: OutlierMethod.NONE

        val samplingText = binding.samplingSpinner.text.toString()
        val samplingMethod = SamplingMethod.values().firstOrNull { it.displayName == samplingText }
            ?: SamplingMethod.ALL

        val sampleSize = binding.sampleSizeInput.text?.toString()?.toIntOrNull() ?: 1000

        val opts = ProcessingOptions(
            featureColumns    = features,
            targetColumns     = targets,
            samplingMethod    = samplingMethod,
            sampleSize        = sampleSize,
            normalisation     = normMethod,
            missingStrategy   = missingMethod,
            outlierMethod     = outlierMethod,
            trainSplitFraction = binding.trainSplitSlider.value.toDouble()
        )

        val result = DataProcessor.process(raw, opts)
        if (result == null) {
            Toast.makeText(this, "Could not process data. Check column selection.", Toast.LENGTH_LONG).show()
            return
        }
        val (trainSet, valSet, _) = result

        if (trainSet.size < 2) {
            Toast.makeText(this, "Too few rows after processing (${trainSet.size}). Adjust filters.", Toast.LENGTH_LONG).show()
            return
        }

        // Serialize compact for intent (keep small — pass by reference via singleton)
        DataHolder.trainSet = trainSet
        DataHolder.valSet   = valSet

        val intent = Intent(this, TrainingActivity::class.java)
        // Forward all network config extras from the previous screen
        intent.putExtras(this.intent)
        startActivity(intent)
    }

    private fun showEmptyState() {
        binding.emptyState.visibility = View.VISIBLE
        binding.dataCard.visibility   = View.GONE
    }
}

/** Simple in-memory singleton to pass large datasets between activities without Parcelable overhead. */
object DataHolder {
    var trainSet: List<Pair<DoubleArray, DoubleArray>> = emptyList()
    var valSet:   List<Pair<DoubleArray, DoubleArray>> = emptyList()
}
