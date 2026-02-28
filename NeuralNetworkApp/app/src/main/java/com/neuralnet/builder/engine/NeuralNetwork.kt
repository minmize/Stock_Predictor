package com.neuralnet.builder.engine

import kotlin.math.sqrt
import kotlin.random.Random

/**
 * Configuration for a single layer in the network.
 */
data class LayerConfig(
    val neurons: Int,
    val activation: ActivationType,
    val dropout: Float = 0f   // Dropout rate 0.0–1.0 (0 = no dropout)
)

/**
 * Snapshot of a single training epoch for reporting progress.
 */
data class EpochResult(
    val epoch: Int,
    val trainLoss: Double,
    val valLoss: Double?,
    val trainAccuracy: Double?,
    val valAccuracy: Double?
)

/**
 * Full neural network implementation using pure Kotlin.
 *
 * Architecture: dense (fully-connected) feed-forward network.
 * Training:     mini-batch stochastic gradient descent with momentum.
 *
 * Indexing convention (L = number of weight matrices = layerConfigs.size - 1):
 *   activations[0]      = input
 *   activations[L]      = output
 *   weights[l]          = weight matrix between activations[l] and activations[l+1]
 *                         shape: [activations[l].size][activations[l+1].size]
 *   preActivations[l]   = z values for computing activations[l+1]
 *   deltas[l]           = error signal at activations[l+1] (for weight update of weights[l])
 */
class NeuralNetwork(
    val layerConfigs: List<LayerConfig>,
    val lossType: LossType = LossType.MSE,
    var learningRate: Double = 0.001,
    private val momentum: Double = 0.9,
    private val seed: Long = 42L
) {
    private val rng = Random(seed)

    // Trainable parameters
    private val weights: List<Array<DoubleArray>>
    private val biases: List<DoubleArray>

    // Momentum accumulators (same shape as weights/biases)
    private val wMomentum: List<Array<DoubleArray>>
    private val bMomentum: List<DoubleArray>

    val numLayers get() = layerConfigs.size
    val numWeightMatrices get() = layerConfigs.size - 1

    // ─── Initialisation ───────────────────────────────────────────────────────

    init {
        require(layerConfigs.size >= 2) { "Network must have at least 2 layers (input + output)." }

        val w = mutableListOf<Array<DoubleArray>>()
        val b = mutableListOf<DoubleArray>()
        val wm = mutableListOf<Array<DoubleArray>>()
        val bm = mutableListOf<DoubleArray>()

        for (l in 0 until layerConfigs.size - 1) {
            val fanIn  = layerConfigs[l].neurons
            val fanOut = layerConfigs[l + 1].neurons

            // He initialisation (good for ReLU); falls back gracefully for others
            val scale = sqrt(2.0 / fanIn)

            w.add(Array(fanIn) { DoubleArray(fanOut) { rng.nextGaussian() * scale } })
            b.add(DoubleArray(fanOut) { 0.0 })
            wm.add(Array(fanIn) { DoubleArray(fanOut) { 0.0 } })
            bm.add(DoubleArray(fanOut) { 0.0 })
        }

        weights    = w
        biases     = b
        wMomentum  = wm
        bMomentum  = bm
    }

    // ─── Forward Pass ─────────────────────────────────────────────────────────

    /**
     * Run the network on a single sample.
     * Returns (activations, preActivations) needed for backprop.
     */
    private fun forwardWithCache(input: DoubleArray): Pair<List<DoubleArray>, List<DoubleArray>> {
        val activations    = mutableListOf(input)
        val preActivations = mutableListOf<DoubleArray>()

        for (l in 0 until numWeightMatrices) {
            val prevAct = activations[l]
            val fanOut  = layerConfigs[l + 1].neurons
            val z = DoubleArray(fanOut) { j ->
                biases[l][j] + prevAct.indices.sumOf { i -> prevAct[i] * weights[l][i][j] }
            }
            preActivations.add(z)
            activations.add(ActivationFunctions.applyVector(z, layerConfigs[l + 1].activation))
        }

        return activations to preActivations
    }

    /** Inference-only forward pass (no cache). */
    fun predict(input: DoubleArray): DoubleArray {
        var a = input
        for (l in 0 until numWeightMatrices) {
            val fanOut = layerConfigs[l + 1].neurons
            val z = DoubleArray(fanOut) { j ->
                biases[l][j] + a.indices.sumOf { i -> a[i] * weights[l][i][j] }
            }
            a = ActivationFunctions.applyVector(z, layerConfigs[l + 1].activation)
        }
        return a
    }

    // ─── Backward Pass ────────────────────────────────────────────────────────

    /**
     * Train on a single (input, target) pair.
     * Returns the loss value for logging.
     */
    private fun trainSample(input: DoubleArray, target: DoubleArray): Double {
        val (activations, preActivations) = forwardWithCache(input)
        val output = activations.last()
        val loss   = LossFunctions.compute(output, target, lossType)

        // deltas[l] = error signal at activations[l+1], used to update weights[l]
        val deltas = arrayOfNulls<DoubleArray>(numWeightMatrices)

        // Output layer delta
        deltas[numWeightMatrices - 1] = LossFunctions.outputDelta(
            output, target, preActivations.last(),
            layerConfigs.last().activation, lossType
        )

        // Hidden layer deltas (back-propagate)
        for (l in numWeightMatrices - 2 downTo 0) {
            val nextDelta = deltas[l + 1]!!
            val fanMid    = layerConfigs[l + 1].neurons
            deltas[l] = DoubleArray(fanMid) { j ->
                // Propagate error through weights[l+1][j][*]
                val propagated = weights[l + 1][j].indices.sumOf { k ->
                    weights[l + 1][j][k] * nextDelta[k]
                }
                propagated * ActivationFunctions.derivative(
                    preActivations[l][j], activations[l + 1][j],
                    layerConfigs[l + 1].activation
                )
            }
        }

        // Weight updates with SGD momentum
        for (l in 0 until numWeightMatrices) {
            val delta = deltas[l]!!
            for (i in weights[l].indices) {
                for (j in weights[l][i].indices) {
                    val grad = activations[l][i] * delta[j]
                    wMomentum[l][i][j] = momentum * wMomentum[l][i][j] + (1 - momentum) * grad
                    weights[l][i][j] -= learningRate * wMomentum[l][i][j]
                }
            }
            for (j in biases[l].indices) {
                bMomentum[l][j] = momentum * bMomentum[l][j] + (1 - momentum) * delta[j]
                biases[l][j] -= learningRate * bMomentum[l][j]
            }
        }

        return loss
    }

    // ─── Mini-Batch Training ──────────────────────────────────────────────────

    /**
     * Train for one epoch over the given dataset.
     * @param data      List of (input, target) pairs
     * @param batchSize Mini-batch size (1 = online SGD)
     * @param shuffle   Randomly shuffle data each epoch
     */
    fun trainEpoch(
        data: List<Pair<DoubleArray, DoubleArray>>,
        batchSize: Int = 32,
        shuffle: Boolean = true
    ): Double {
        val dataset = if (shuffle) data.shuffled(rng) else data
        var totalLoss = 0.0

        dataset.chunked(batchSize).forEach { batch ->
            batch.forEach { (x, y) ->
                totalLoss += trainSample(x, y)
            }
        }

        return totalLoss / dataset.size
    }

    /**
     * Full training loop.
     * Calls [onEpochEnd] after each epoch with an [EpochResult].
     * Returns early if [onEpochEnd] returns false (stop signal).
     */
    fun train(
        trainData: List<Pair<DoubleArray, DoubleArray>>,
        valData: List<Pair<DoubleArray, DoubleArray>>? = null,
        epochs: Int = 50,
        batchSize: Int = 32,
        onEpochEnd: (EpochResult) -> Boolean = { true }
    ) {
        for (epoch in 1..epochs) {
            val trainLoss = trainEpoch(trainData, batchSize)

            val valLoss = valData?.let { vd ->
                vd.sumOf { (x, y) -> LossFunctions.compute(predict(x), y, lossType) } / vd.size
            }

            val trainAcc  = computeAccuracy(trainData)
            val valAcc    = valData?.let { computeAccuracy(it) }

            val result = EpochResult(epoch, trainLoss, valLoss, trainAcc, valAcc)
            if (!onEpochEnd(result)) break
        }
    }

    // ─── Metrics ──────────────────────────────────────────────────────────────

    /** Classification accuracy (works for both binary and multi-class). */
    private fun computeAccuracy(data: List<Pair<DoubleArray, DoubleArray>>): Double? {
        if (lossType == LossType.MSE) return null  // Regression, no accuracy
        var correct = 0
        data.forEach { (x, y) ->
            val pred = predict(x)
            correct += if (pred.argmax() == y.argmax()) 1 else 0
        }
        return correct.toDouble() / data.size
    }

    /** Evaluate on a dataset: returns (loss, accuracy). */
    fun evaluate(data: List<Pair<DoubleArray, DoubleArray>>): Pair<Double, Double?> {
        val loss = data.sumOf { (x, y) -> LossFunctions.compute(predict(x), y, lossType) } / data.size
        val acc  = computeAccuracy(data)
        return loss to acc
    }

    // ─── Serialisation ────────────────────────────────────────────────────────

    /** Export model to a simple JSON-compatible map for saving. */
    fun toMap(): Map<String, Any> = mapOf(
        "layerConfigs" to layerConfigs.map { mapOf("neurons" to it.neurons, "activation" to it.activation.name) },
        "lossType"     to lossType.name,
        "learningRate" to learningRate,
        "weights"      to weights.map { layer -> layer.map { row -> row.toList() } },
        "biases"       to biases.map { it.toList() }
    )

    // ─── Summary ──────────────────────────────────────────────────────────────

    fun summary(): String = buildString {
        appendLine("═══════════════════════════════════════")
        appendLine("  Neural Network Architecture Summary")
        appendLine("═══════════════════════════════════════")
        var totalParams = 0
        layerConfigs.forEachIndexed { i, cfg ->
            val layerName = when (i) {
                0                        -> "Input Layer   "
                layerConfigs.size - 1    -> "Output Layer  "
                else                     -> "Hidden Layer $i"
            }
            val params = if (i == 0) 0 else {
                val prev = layerConfigs[i - 1].neurons
                prev * cfg.neurons + cfg.neurons
            }
            totalParams += params
            appendLine("  $layerName | ${cfg.neurons} neurons | ${cfg.activation.displayName} | params: $params")
        }
        appendLine("───────────────────────────────────────")
        appendLine("  Total trainable parameters: $totalParams")
        appendLine("  Loss function: ${lossType.displayName}")
        appendLine("  Learning rate: $learningRate")
        appendLine("═══════════════════════════════════════")
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

private fun DoubleArray.argmax(): Int {
    var best = 0
    for (i in 1 until size) if (this[i] > this[best]) best = i
    return best
}

private fun Random.nextGaussian(): Double {
    // Box-Muller transform
    val u = nextDouble(1e-10, 1.0)
    val v = nextDouble(0.0, 1.0)
    return kotlin.math.sqrt(-2.0 * kotlin.math.ln(u)) * kotlin.math.cos(2.0 * Math.PI * v)
}
