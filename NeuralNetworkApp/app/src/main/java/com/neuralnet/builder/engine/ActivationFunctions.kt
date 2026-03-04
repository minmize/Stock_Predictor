package com.neuralnet.builder.engine

import kotlin.math.exp
import kotlin.math.max
import kotlin.math.tanh

/**
 * Supported activation functions.
 * Each controls how a neuron "fires" — a key concept in neural networks.
 */
enum class ActivationType(val displayName: String, val description: String) {
    RELU("ReLU", "Best for hidden layers. Fast and avoids vanishing gradients."),
    SIGMOID("Sigmoid", "Outputs 0–1. Good for binary classification output."),
    TANH("Tanh", "Outputs -1 to 1. Often better than sigmoid for hidden layers."),
    SOFTMAX("Softmax", "Converts outputs to probabilities. Use for multi-class output."),
    LINEAR("Linear", "No transformation. Use for regression output layers.")
}

/**
 * Supported loss (cost) functions.
 * A loss function measures how wrong the network's predictions are.
 */
enum class LossType(val displayName: String, val description: String) {
    MSE("Mean Squared Error", "Best for regression tasks (predicting numbers)."),
    BINARY_CROSS_ENTROPY("Binary Cross-Entropy", "Best for yes/no classification."),
    CATEGORICAL_CROSS_ENTROPY("Categorical Cross-Entropy", "Best for multi-class classification.")
}

object ActivationFunctions {

    /** Apply activation function to a single value */
    fun apply(x: Double, type: ActivationType): Double = when (type) {
        ActivationType.RELU -> max(0.0, x)
        ActivationType.SIGMOID -> 1.0 / (1.0 + exp(-x.coerceIn(-500.0, 500.0)))
        ActivationType.TANH -> tanh(x)
        ActivationType.LINEAR -> x
        ActivationType.SOFTMAX -> x // Softmax is handled vector-wise below
    }

    /** Apply activation to a whole vector (needed for softmax) */
    fun applyVector(z: DoubleArray, type: ActivationType): DoubleArray = when (type) {
        ActivationType.SOFTMAX -> {
            val maxZ = z.max()
            val exps = DoubleArray(z.size) { exp(z[it] - maxZ) }
            val sum = exps.sum().coerceAtLeast(1e-10)
            DoubleArray(z.size) { exps[it] / sum }
        }
        else -> DoubleArray(z.size) { apply(z[it], type) }
    }

    /**
     * Derivative of the activation function (for backpropagation).
     * @param z the pre-activation value
     * @param a the post-activation value (already computed, avoids recomputing)
     */
    fun derivative(z: Double, a: Double, type: ActivationType): Double = when (type) {
        ActivationType.RELU -> if (z > 0.0) 1.0 else 0.0
        ActivationType.SIGMOID -> a * (1.0 - a)   // σ(z) * (1 - σ(z))
        ActivationType.TANH -> 1.0 - a * a         // 1 - tanh²(z)
        ActivationType.LINEAR -> 1.0
        ActivationType.SOFTMAX -> 1.0              // Combined with cross-entropy loss derivative
    }
}

object LossFunctions {

    /** Compute scalar loss value */
    fun compute(predicted: DoubleArray, target: DoubleArray, type: LossType): Double {
        return when (type) {
            LossType.MSE -> {
                predicted.indices.sumOf { i ->
                    val diff = predicted[i] - target[i]
                    diff * diff
                } / predicted.size
            }
            LossType.BINARY_CROSS_ENTROPY -> {
                -predicted.indices.sumOf { i ->
                    val p = predicted[i].coerceIn(1e-10, 1 - 1e-10)
                    target[i] * Math.log(p) + (1 - target[i]) * Math.log(1 - p)
                } / predicted.size
            }
            LossType.CATEGORICAL_CROSS_ENTROPY -> {
                -predicted.indices.sumOf { i ->
                    val p = predicted[i].coerceIn(1e-10, 1.0)
                    target[i] * Math.log(p)
                }
            }
        }
    }

    /**
     * Gradient of the loss w.r.t. the output layer's pre-activation (z).
     * For softmax + cross-entropy this simplifies to (predicted - target).
     */
    fun outputDelta(
        predicted: DoubleArray,
        target: DoubleArray,
        preActivation: DoubleArray,
        activation: ActivationType,
        type: LossType
    ): DoubleArray {
        return when {
            // Softmax + cross-entropy → elegant simplification: δ = predicted - target
            activation == ActivationType.SOFTMAX &&
                    (type == LossType.CATEGORICAL_CROSS_ENTROPY || type == LossType.BINARY_CROSS_ENTROPY) ->
                DoubleArray(predicted.size) { i -> predicted[i] - target[i] }

            // Sigmoid + BCE → δ = predicted - target
            activation == ActivationType.SIGMOID && type == LossType.BINARY_CROSS_ENTROPY ->
                DoubleArray(predicted.size) { i -> predicted[i] - target[i] }

            // MSE general case
            else -> DoubleArray(predicted.size) { i ->
                val dLoss = 2.0 * (predicted[i] - target[i]) / predicted.size
                val dAct = ActivationFunctions.derivative(preActivation[i], predicted[i], activation)
                dLoss * dAct
            }
        }
    }
}
