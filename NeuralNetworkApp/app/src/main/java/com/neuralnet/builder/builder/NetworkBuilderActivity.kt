package com.neuralnet.builder.builder

import android.content.Intent
import android.os.Bundle
import android.text.Editable
import android.text.TextWatcher
import android.view.LayoutInflater
import android.view.ViewGroup
import android.widget.ArrayAdapter
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.ItemTouchHelper
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.neuralnet.builder.data.DataInputActivity
import com.neuralnet.builder.databinding.ActivityNetworkBuilderBinding
import com.neuralnet.builder.databinding.ItemLayerCardBinding
import com.neuralnet.builder.engine.ActivationType
import com.neuralnet.builder.engine.LayerConfig
import com.neuralnet.builder.engine.LossType
import com.neuralnet.builder.engine.NeuralNetwork

class NetworkBuilderActivity : AppCompatActivity() {

    private lateinit var binding: ActivityNetworkBuilderBinding
    private lateinit var layerAdapter: LayerCardAdapter
    private val layers = mutableListOf<LayerConfig>()

    companion object {
        const val EXTRA_QUICK_START = "quick_start"
        const val EXTRA_NETWORK_CONFIG = "network_config"
        const val EXTRA_LEARNING_RATE = "learning_rate"
        const val EXTRA_EPOCHS = "epochs"
        const val EXTRA_BATCH_SIZE = "batch_size"
        const val EXTRA_LOSS_TYPE = "loss_type"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityNetworkBuilderBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setupToolbar()
        setupLayerList()
        setupTextInput()
        setupControls()
        setupLossSpinner()

        if (intent.getBooleanExtra(EXTRA_QUICK_START, false)) {
            loadQuickStartConfig()
        } else {
            loadDefaultConfig()
        }
    }

    private fun setupToolbar() {
        setSupportActionBar(binding.toolbar)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        supportActionBar?.title = "Build Network"
        binding.toolbar.setNavigationOnClickListener { onBackPressedDispatcher.onBackPressed() }
    }

    // ── Layer List ────────────────────────────────────────────────────────────

    private fun setupLayerList() {
        layerAdapter = LayerCardAdapter(layers,
            onRemove = { pos ->
                if (layers.size > 2) {
                    layers.removeAt(pos)
                    layerAdapter.notifyItemRemoved(pos)
                    layerAdapter.notifyItemRangeChanged(pos, layers.size)
                    syncTextFromLayers()
                    updateSummary()
                } else {
                    Toast.makeText(this, "Network needs at least 2 layers", Toast.LENGTH_SHORT).show()
                }
            },
            onChange = { syncTextFromLayers(); updateSummary() }
        )
        binding.layerRecycler.layoutManager = LinearLayoutManager(this)
        binding.layerRecycler.adapter = layerAdapter

        // Drag to reorder (not for input/output)
        val callback = object : ItemTouchHelper.SimpleCallback(
            ItemTouchHelper.UP or ItemTouchHelper.DOWN, 0
        ) {
            override fun onMove(rv: RecyclerView, from: RecyclerView.ViewHolder, to: RecyclerView.ViewHolder): Boolean {
                val fromPos = from.adapterPosition
                val toPos   = to.adapterPosition
                if (fromPos == 0 || toPos == 0 || fromPos == layers.size - 1 || toPos == layers.size - 1)
                    return false
                layers.add(toPos, layers.removeAt(fromPos))
                layerAdapter.notifyItemMoved(fromPos, toPos)
                syncTextFromLayers()
                updateSummary()
                return true
            }
            override fun onSwiped(viewHolder: RecyclerView.ViewHolder, direction: Int) {}
        }
        ItemTouchHelper(callback).attachToRecyclerView(binding.layerRecycler)
    }

    // ── Text Input Parsing ────────────────────────────────────────────────────

    private fun setupTextInput() {
        binding.structureInput.addTextChangedListener(object : TextWatcher {
            override fun afterTextChanged(s: Editable?) = parseStructureText(s.toString())
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}
        })

        binding.exampleChips.setOnCheckedStateChangeListener { group, checkedIds ->
            val example = when {
                checkedIds.contains(binding.chip2class.id)    -> "2-32-16-1"
                checkedIds.contains(binding.chipMulti.id)     -> "4-64-32-10"
                checkedIds.contains(binding.chipRegress.id)   -> "6-128-64-32-1"
                checkedIds.contains(binding.chipDeep.id)      -> "10-256-128-64-32-1"
                else -> return@setOnCheckedStateChangeListener
            }
            binding.structureInput.setText(example)
        }
    }

    private fun parseStructureText(text: String) {
        val parts = text.trim().split("-").mapNotNull { it.trim().toIntOrNull() }
        if (parts.size < 2) {
            binding.parseError.text = "Enter at least 2 numbers separated by dashes (e.g. 4-32-1)"
            return
        }
        if (parts.any { it < 1 || it > 4096 }) {
            binding.parseError.text = "Each layer size must be between 1 and 4096"
            return
        }
        binding.parseError.text = ""

        layers.clear()
        parts.forEachIndexed { i, size ->
            val activation = when {
                i == parts.size - 1 -> guessOutputActivation()
                else -> ActivationType.RELU
            }
            layers.add(LayerConfig(size, activation))
        }
        layerAdapter.notifyDataSetChanged()
        updateSummary()
    }

    private fun syncTextFromLayers() {
        val text = layers.joinToString("-") { it.neurons.toString() }
        binding.structureInput.removeTextChangedListener(null)
        if (binding.structureInput.text.toString() != text)
            binding.structureInput.setText(text)
    }

    // ── Controls ──────────────────────────────────────────────────────────────

    private fun setupControls() {
        binding.addLayerBtn.setOnClickListener {
            val insertAt = (layers.size - 1).coerceAtLeast(1)
            layers.add(insertAt, LayerConfig(64, ActivationType.RELU))
            layerAdapter.notifyItemInserted(insertAt)
            layerAdapter.notifyItemRangeChanged(insertAt, layers.size)
            syncTextFromLayers()
            updateSummary()
        }

        binding.nextBtn.setOnClickListener {
            if (validateNetwork()) proceedToData()
        }

        binding.btnSummary.setOnClickListener { showNetworkSummary() }
    }

    private fun setupLossSpinner() {
        val lossOptions = LossType.values().map { it.displayName }
        val adapter = ArrayAdapter(this, android.R.layout.simple_dropdown_item_1line, lossOptions)
        binding.lossSpinner.setAdapter(adapter)
        binding.lossSpinner.setText(LossType.MSE.displayName, false)
    }

    // ── Defaults ──────────────────────────────────────────────────────────────

    private fun loadDefaultConfig() {
        binding.structureInput.setText("4-64-32-1")
        binding.lrInput.setText("0.001")
        binding.epochsInput.setText("50")
        binding.batchInput.setText("32")
    }

    private fun loadQuickStartConfig() {
        binding.structureInput.setText("2-16-8-1")
        binding.lrInput.setText("0.01")
        binding.epochsInput.setText("30")
        binding.batchInput.setText("16")
        binding.lossSpinner.setText(LossType.MSE.displayName, false)
        Toast.makeText(this, "Quick Start config loaded!", Toast.LENGTH_SHORT).show()
    }

    private fun guessOutputActivation(): ActivationType {
        val lossText = binding.lossSpinner.text?.toString() ?: ""
        return when {
            lossText.contains("Binary")       -> ActivationType.SIGMOID
            lossText.contains("Categorical")  -> ActivationType.SOFTMAX
            else                              -> ActivationType.LINEAR
        }
    }

    // ── Validation & Navigation ───────────────────────────────────────────────

    private fun validateNetwork(): Boolean {
        if (layers.size < 2) {
            Toast.makeText(this, "Add at least 2 layers", Toast.LENGTH_SHORT).show()
            return false
        }
        val lr = binding.lrInput.text?.toString()?.toDoubleOrNull()
        if (lr == null || lr <= 0) {
            binding.lrInput.error = "Enter a positive learning rate (e.g. 0.001)"
            return false
        }
        val epochs = binding.epochsInput.text?.toString()?.toIntOrNull()
        if (epochs == null || epochs < 1) {
            binding.epochsInput.error = "Enter a number of epochs (e.g. 50)"
            return false
        }
        val batch = binding.batchInput.text?.toString()?.toIntOrNull()
        if (batch == null || batch < 1) {
            binding.batchInput.error = "Enter a batch size (e.g. 32)"
            return false
        }
        return true
    }

    private fun proceedToData() {
        val lossName = binding.lossSpinner.text?.toString() ?: LossType.MSE.displayName
        val lossType = LossType.values().firstOrNull { it.displayName == lossName } ?: LossType.MSE

        // Ensure output activation matches loss
        val updatedLayers = layers.toMutableList()
        val lastIndex = updatedLayers.size - 1
        val outputAct = when (lossType) {
            LossType.BINARY_CROSS_ENTROPY        -> ActivationType.SIGMOID
            LossType.CATEGORICAL_CROSS_ENTROPY   -> ActivationType.SOFTMAX
            LossType.MSE                         -> ActivationType.LINEAR
        }
        updatedLayers[lastIndex] = updatedLayers[lastIndex].copy(activation = outputAct)

        val intent = Intent(this, DataInputActivity::class.java).apply {
            putExtra(EXTRA_LEARNING_RATE, binding.lrInput.text.toString().toDouble())
            putExtra(EXTRA_EPOCHS, binding.epochsInput.text.toString().toInt())
            putExtra(EXTRA_BATCH_SIZE, binding.batchInput.text.toString().toInt())
            putExtra(EXTRA_LOSS_TYPE, lossType.name)
            // Pack layer config as two parallel int arrays + string array
            putExtra("layer_sizes", updatedLayers.map { it.neurons }.toIntArray())
            putExtra("layer_acts",  updatedLayers.map { it.activation.name }.toTypedArray())
            putExtra("layer_drops", updatedLayers.map { it.dropout }.toFloatArray())
        }
        startActivity(intent)
    }

    private fun showNetworkSummary() {
        if (layers.size < 2) { Toast.makeText(this, "Build a network first", Toast.LENGTH_SHORT).show(); return }
        val lossName = binding.lossSpinner.text?.toString() ?: LossType.MSE.displayName
        val lossType = LossType.values().firstOrNull { it.displayName == lossName } ?: LossType.MSE
        val lr = binding.lrInput.text?.toString()?.toDoubleOrNull() ?: 0.001
        val net = NeuralNetwork(layers, lossType, lr)
        MaterialAlertDialogBuilder(this)
            .setTitle("Network Summary")
            .setMessage(net.summary())
            .setPositiveButton("OK", null)
            .show()
    }

    private fun updateSummary() {
        if (layers.size < 2) return
        var total = 0
        layers.forEachIndexed { i, cfg ->
            if (i > 0) total += layers[i - 1].neurons * cfg.neurons + cfg.neurons
        }
        binding.paramCount.text = "Parameters: $total"
        binding.layerCount.text = "Layers: ${layers.size} (${layers.size - 2} hidden)"
    }
}

// ─── Layer Card Adapter ───────────────────────────────────────────────────────

class LayerCardAdapter(
    private val layers: MutableList<LayerConfig>,
    private val onRemove: (Int) -> Unit,
    private val onChange: () -> Unit
) : RecyclerView.Adapter<LayerCardAdapter.VH>() {

    inner class VH(val b: ItemLayerCardBinding) : RecyclerView.ViewHolder(b.root)

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int) =
        VH(ItemLayerCardBinding.inflate(LayoutInflater.from(parent.context), parent, false))

    override fun getItemCount() = layers.size

    override fun onBindViewHolder(holder: VH, position: Int) {
        val cfg = layers[position]
        val b   = holder.b
        val isInput  = position == 0
        val isOutput = position == layers.size - 1

        b.layerLabel.text = when {
            isInput  -> "Input Layer"
            isOutput -> "Output Layer"
            else     -> "Hidden Layer $position"
        }
        b.layerBadge.text = when {
            isInput  -> "IN"
            isOutput -> "OUT"
            else     -> "$position"
        }

        b.neuronInput.setText(cfg.neurons.toString())
        b.neuronInput.addTextChangedListener(object : android.text.TextWatcher {
            override fun afterTextChanged(s: android.text.Editable?) {
                val n = s?.toString()?.toIntOrNull()
                if (n != null && n in 1..4096) {
                    layers[holder.adapterPosition] = layers[holder.adapterPosition].copy(neurons = n)
                    onChange()
                }
            }
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}
        })

        val actNames = ActivationType.values().map { it.displayName }
        val actAdapter = ArrayAdapter(b.root.context, android.R.layout.simple_dropdown_item_1line, actNames)
        b.activationSpinner.setAdapter(actAdapter)
        b.activationSpinner.setText(cfg.activation.displayName, false)
        b.activationSpinner.setOnItemClickListener { _, _, idx, _ ->
            layers[holder.adapterPosition] = layers[holder.adapterPosition].copy(activation = ActivationType.values()[idx])
            onChange()
        }

        b.dropoutSlider.value = cfg.dropout
        b.dropoutLabel.text = "Dropout: ${"%.0f".format(cfg.dropout * 100)}%"
        b.dropoutSlider.addOnChangeListener { _, value, fromUser ->
            if (fromUser) {
                layers[holder.adapterPosition] = layers[holder.adapterPosition].copy(dropout = value)
                b.dropoutLabel.text = "Dropout: ${"%.0f".format(value * 100)}%"
                onChange()
            }
        }

        // Can't delete input or output layers
        b.removeBtn.isEnabled = !isInput && !isOutput
        b.removeBtn.alpha = if (b.removeBtn.isEnabled) 1f else 0.3f
        b.removeBtn.setOnClickListener { onRemove(holder.adapterPosition) }

        // Hint text for activation
        b.activationHint.text = cfg.activation.description
    }
}
