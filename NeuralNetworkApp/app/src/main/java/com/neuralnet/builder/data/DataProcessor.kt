package com.neuralnet.builder.data

import android.content.Context
import android.net.Uri
import kotlin.math.abs
import kotlin.math.pow
import kotlin.math.sqrt

// ─── Data Model ───────────────────────────────────────────────────────────────

/** A loaded dataset: column names + rows of raw String values. */
data class RawDataset(
    val headers: List<String>,
    val rows: List<List<String>>
) {
    val rowCount get() = rows.size
    val colCount get() = headers.size
}

/** Fully numeric dataset ready for training. */
data class ProcessedDataset(
    val featureNames: List<String>,
    val targetName: String,
    val X: List<DoubleArray>,   // Feature matrix (N samples × F features)
    val y: List<DoubleArray>,   // Target matrix (N samples × T targets)
    val normParams: NormParams? // Saved normalisation params for inference
)

data class NormParams(
    val featureMins: DoubleArray,
    val featureMaxs: DoubleArray,
    val featureMeans: DoubleArray,
    val featureStds: DoubleArray
)

// ─── Options ──────────────────────────────────────────────────────────────────

enum class SamplingMethod(val displayName: String) {
    ALL("Use all data"),
    RANDOM("Random sample"),
    FIRST_N("First N rows"),
    LAST_N("Last N rows"),
    STRATIFIED("Stratified sample")
}

enum class NormalisationMethod(val displayName: String) {
    NONE("None"),
    MIN_MAX("Min-Max (0 to 1)"),
    Z_SCORE("Z-Score (mean=0, std=1)"),
    ROBUST("Robust (median/IQR)")
}

enum class MissingValueStrategy(val displayName: String) {
    DROP_ROW("Drop rows with missing values"),
    FILL_MEAN("Fill with column mean"),
    FILL_MEDIAN("Fill with column median"),
    FILL_MODE("Fill with column mode"),
    FILL_ZERO("Fill with 0")
}

enum class OutlierMethod(val displayName: String) {
    NONE("Keep all data"),
    IQR("Remove IQR outliers (1.5×)"),
    Z_SCORE("Remove Z-score outliers (|z|>3)"),
    PERCENTILE("Clip to 1st–99th percentile")
}

data class ProcessingOptions(
    val featureColumns: List<String>           = emptyList(),
    val targetColumns: List<String>            = emptyList(),
    val samplingMethod: SamplingMethod         = SamplingMethod.ALL,
    val sampleSize: Int                        = 1000,
    val sampleFraction: Double                 = 1.0,
    val normalisation: NormalisationMethod     = NormalisationMethod.MIN_MAX,
    val missingStrategy: MissingValueStrategy  = MissingValueStrategy.DROP_ROW,
    val outlierMethod: OutlierMethod           = OutlierMethod.NONE,
    val trainSplitFraction: Double             = 0.8,
    val shuffleBeforeSplit: Boolean            = true,
    val randomSeed: Long                       = 42L
)

// ─── Loader ───────────────────────────────────────────────────────────────────

object DataLoader {

    /** Load a CSV or TSV from a content URI. Returns null on failure. */
    fun loadFromUri(context: Context, uri: Uri): RawDataset? = try {
        val text = context.contentResolver.openInputStream(uri)?.bufferedReader()?.readText()
            ?: return null

        val lines = text.lines().filter { it.isNotBlank() }
        if (lines.size < 2) return null

        val sep = detectSeparator(lines[0])
        val headers = lines[0].splitCsv(sep)
        val rows    = lines.drop(1).map { it.splitCsv(sep) }
            .filter { it.size == headers.size }

        RawDataset(headers, rows)
    } catch (_: Exception) { null }

    private fun detectSeparator(line: String): Char {
        val counts = mapOf(',' to line.count { it == ',' },
                           '\t' to line.count { it == '\t' },
                           ';' to line.count { it == ';' })
        return counts.maxByOrNull { it.value }?.key ?: ','
    }

    private fun String.splitCsv(sep: Char): List<String> {
        val result = mutableListOf<String>()
        var inQuotes = false
        val current = StringBuilder()
        for (ch in this) {
            when {
                ch == '"' -> inQuotes = !inQuotes
                ch == sep && !inQuotes -> { result.add(current.toString().trim()); current.clear() }
                else -> current.append(ch)
            }
        }
        result.add(current.toString().trim())
        return result
    }
}

// ─── Processor ────────────────────────────────────────────────────────────────

object DataProcessor {

    // ── Summary Statistics ───────────────────────────────────────────────────

    fun columnStats(raw: RawDataset, column: String): ColumnStats? {
        val idx = raw.headers.indexOf(column).takeIf { it >= 0 } ?: return null
        val values = raw.rows.mapNotNull { row -> row.getOrNull(idx)?.toDoubleOrNull() }
        if (values.isEmpty()) return null
        return ColumnStats.from(column, values)
    }

    fun allColumnStats(raw: RawDataset): Map<String, ColumnStats> =
        raw.headers.associateWith { col ->
            val idx = raw.headers.indexOf(col)
            val values = raw.rows.mapNotNull { it.getOrNull(idx)?.toDoubleOrNull() }
            ColumnStats.from(col, values)
        }.filterValues { it.count > 0 }

    // ── Missing Value Handling ────────────────────────────────────────────────

    private fun handleMissing(
        raw: RawDataset,
        cols: List<String>,
        strategy: MissingValueStrategy
    ): RawDataset {
        val indices = cols.map { raw.headers.indexOf(it) }.filter { it >= 0 }

        return when (strategy) {
            MissingValueStrategy.DROP_ROW -> {
                val cleaned = raw.rows.filter { row ->
                    indices.all { i -> row.getOrNull(i)?.toDoubleOrNull() != null }
                }
                raw.copy(rows = cleaned)
            }
            else -> {
                // Compute fill values per column
                val fillValues = indices.associateWith { idx ->
                    val vals = raw.rows.mapNotNull { it.getOrNull(idx)?.toDoubleOrNull() }
                    when (strategy) {
                        MissingValueStrategy.FILL_MEAN   -> vals.average()
                        MissingValueStrategy.FILL_MEDIAN -> median(vals)
                        MissingValueStrategy.FILL_MODE   -> mode(vals)
                        MissingValueStrategy.FILL_ZERO   -> 0.0
                        else -> 0.0
                    }
                }
                val newRows = raw.rows.map { row ->
                    row.mapIndexed { i, v ->
                        if (i in fillValues && v.toDoubleOrNull() == null)
                            fillValues[i].toString()
                        else v
                    }
                }
                raw.copy(rows = newRows)
            }
        }
    }

    // ── Sampling ─────────────────────────────────────────────────────────────

    private fun sample(rows: List<List<String>>, opts: ProcessingOptions): List<List<String>> {
        val rng = kotlin.random.Random(opts.randomSeed)
        return when (opts.samplingMethod) {
            SamplingMethod.ALL        -> rows
            SamplingMethod.RANDOM     -> rows.shuffled(rng).take(opts.sampleSize)
            SamplingMethod.FIRST_N    -> rows.take(opts.sampleSize)
            SamplingMethod.LAST_N     -> rows.takeLast(opts.sampleSize)
            SamplingMethod.STRATIFIED -> stratifiedSample(rows, opts.sampleSize, rng)
        }
    }

    private fun stratifiedSample(
        rows: List<List<String>>,
        n: Int,
        rng: kotlin.random.Random
    ): List<List<String>> {
        // Group by last column (assumed label); proportional sampling
        val groups = rows.groupBy { it.lastOrNull() ?: "" }
        return groups.flatMap { (_, g) ->
            val take = (n.toDouble() * g.size / rows.size).toInt().coerceAtLeast(1)
            g.shuffled(rng).take(take)
        }
    }

    // ── Outlier Removal ──────────────────────────────────────────────────────

    private fun removeOutliers(
        values: List<Double>,
        method: OutlierMethod
    ): List<Int> { // Returns indices to keep
        return when (method) {
            OutlierMethod.NONE -> values.indices.toList()
            OutlierMethod.IQR  -> {
                val sorted = values.sorted()
                val q1 = percentile(sorted, 25.0)
                val q3 = percentile(sorted, 75.0)
                val iqr = q3 - q1
                val lo  = q1 - 1.5 * iqr
                val hi  = q3 + 1.5 * iqr
                values.indices.filter { values[it] in lo..hi }
            }
            OutlierMethod.Z_SCORE -> {
                val mean = values.average()
                val std  = stdDev(values)
                if (std == 0.0) return values.indices.toList()
                values.indices.filter { abs((values[it] - mean) / std) <= 3.0 }
            }
            OutlierMethod.PERCENTILE -> {
                val p1  = percentile(values.sorted(), 1.0)
                val p99 = percentile(values.sorted(), 99.0)
                values.indices.filter { values[it] in p1..p99 }
            }
        }
    }

    // ── Normalisation ────────────────────────────────────────────────────────

    private fun buildNormParams(X: List<DoubleArray>): NormParams {
        val nFeatures = X[0].size
        val mins  = DoubleArray(nFeatures)
        val maxs  = DoubleArray(nFeatures)
        val means = DoubleArray(nFeatures)
        val stds  = DoubleArray(nFeatures)

        for (f in 0 until nFeatures) {
            val col = X.map { it[f] }
            mins[f]  = col.min()
            maxs[f]  = col.max()
            means[f] = col.average()
            stds[f]  = stdDev(col).coerceAtLeast(1e-10)
        }
        return NormParams(mins, maxs, means, stds)
    }

    fun normalise(row: DoubleArray, params: NormParams, method: NormalisationMethod): DoubleArray =
        when (method) {
            NormalisationMethod.NONE    -> row
            NormalisationMethod.MIN_MAX -> DoubleArray(row.size) { i ->
                val range = params.featureMaxs[i] - params.featureMins[i]
                if (range == 0.0) 0.0 else (row[i] - params.featureMins[i]) / range
            }
            NormalisationMethod.Z_SCORE -> DoubleArray(row.size) { i ->
                (row[i] - params.featureMeans[i]) / params.featureStds[i]
            }
            NormalisationMethod.ROBUST  -> DoubleArray(row.size) { i ->
                val range = (params.featureMaxs[i] - params.featureMins[i]).coerceAtLeast(1e-10)
                (row[i] - params.featureMeans[i]) / range
            }
        }

    // ── Main Processing Pipeline ─────────────────────────────────────────────

    /**
     * Full pipeline: missing values → sampling → outlier removal → feature extraction → normalise → split.
     * Returns (trainSet, valSet, processedDataset).
     */
    fun process(
        raw: RawDataset,
        opts: ProcessingOptions
    ): Triple<List<Pair<DoubleArray, DoubleArray>>, List<Pair<DoubleArray, DoubleArray>>, ProcessedDataset>? {

        val featureCols = opts.featureColumns.filter { raw.headers.contains(it) }
        val targetCols  = opts.targetColumns.filter  { raw.headers.contains(it) }
        if (featureCols.isEmpty() || targetCols.isEmpty()) return null

        val allCols = featureCols + targetCols

        // 1. Handle missing values
        val cleaned = handleMissing(raw, allCols, opts.missingStrategy)

        // 2. Sampling
        val sampled = sample(cleaned.rows, opts)

        // 3. Parse to numbers
        val featIdx   = featureCols.map { cleaned.headers.indexOf(it) }
        val targetIdx = targetCols.map  { cleaned.headers.indexOf(it) }

        val numericRows = sampled.mapNotNull { row ->
            val feats = featIdx.map { row.getOrNull(it)?.toDoubleOrNull() ?: return@mapNotNull null }
            val tgts  = targetIdx.map { row.getOrNull(it)?.toDoubleOrNull() ?: return@mapNotNull null }
            feats.toDoubleArray() to tgts.toDoubleArray()
        }
        if (numericRows.isEmpty()) return null

        // 4. Outlier removal (first feature column drives the filter)
        val firstFeatureVals = numericRows.map { it.first[0] }
        val keepIdx = removeOutliers(firstFeatureVals, opts.outlierMethod).toSet()
        val filtered = numericRows.filterIndexed { i, _ -> i in keepIdx }

        // 5. Build normalisation params from features
        val Xraw = filtered.map { it.first }
        val norm = if (opts.normalisation != NormalisationMethod.NONE) buildNormParams(Xraw) else null

        // 6. Normalise features
        val Xnorm = if (norm != null) Xraw.map { normalise(it, norm, opts.normalisation) }
                    else Xraw

        val Y = filtered.map { it.second }

        // 7. Shuffle + split
        val rng = kotlin.random.Random(opts.randomSeed)
        val data: List<Pair<DoubleArray, DoubleArray>> =
            if (opts.shuffleBeforeSplit)
                Xnorm.zip(Y).shuffled(rng)
            else
                Xnorm.zip(Y)

        val splitAt = (data.size * opts.trainSplitFraction).toInt().coerceAtLeast(1)
        val trainSet = data.subList(0, splitAt)
        val valSet   = data.subList(splitAt, data.size)

        val fullDataset = ProcessedDataset(featureCols, targetCols.first(), Xnorm, Y, norm)
        return Triple(trainSet, valSet, fullDataset)
    }

    // ── Statistical helpers ──────────────────────────────────────────────────

    fun mean(values: List<Double>): Double = if (values.isEmpty()) 0.0 else values.average()

    fun median(values: List<Double>): Double {
        if (values.isEmpty()) return 0.0
        val sorted = values.sorted()
        val mid = sorted.size / 2
        return if (sorted.size % 2 == 0) (sorted[mid - 1] + sorted[mid]) / 2 else sorted[mid]
    }

    fun mode(values: List<Double>): Double =
        values.groupingBy { it }.eachCount().maxByOrNull { it.value }?.key ?: 0.0

    fun stdDev(values: List<Double>): Double {
        if (values.size < 2) return 0.0
        val mean = values.average()
        return sqrt(values.sumOf { (it - mean).pow(2) } / (values.size - 1))
    }

    fun percentile(sorted: List<Double>, p: Double): Double {
        if (sorted.isEmpty()) return 0.0
        val idx = (p / 100.0 * (sorted.size - 1)).coerceIn(0.0, (sorted.size - 1).toDouble())
        val lo  = sorted[idx.toInt()]
        val hi  = sorted.getOrElse(idx.toInt() + 1) { lo }
        return lo + (hi - lo) * (idx - idx.toInt())
    }

    fun correlationMatrix(X: List<DoubleArray>): Array<DoubleArray> {
        val n = X[0].size
        return Array(n) { i ->
            DoubleArray(n) { j -> pearsonCorrelation(X.map { it[i] }, X.map { it[j] }) }
        }
    }

    fun pearsonCorrelation(a: List<Double>, b: List<Double>): Double {
        val meanA = a.average(); val meanB = b.average()
        val num = a.indices.sumOf { (a[it] - meanA) * (b[it] - meanB) }
        val denA = sqrt(a.sumOf { (it - meanA).pow(2) })
        val denB = sqrt(b.sumOf { (it - meanB).pow(2) })
        return if (denA * denB == 0.0) 0.0 else num / (denA * denB)
    }
}

// ─── Column Statistics ────────────────────────────────────────────────────────

data class ColumnStats(
    val name: String,
    val count: Int,
    val missing: Int,
    val mean: Double,
    val median: Double,
    val mode: Double,
    val std: Double,
    val min: Double,
    val max: Double,
    val q1: Double,
    val q3: Double
) {
    val isNumeric get() = count > 0

    fun describe(): String = buildString {
        appendLine("Column: $name")
        appendLine("  Count:   $count  |  Missing: $missing")
        appendLine("  Mean:    ${"%.4f".format(mean)}")
        appendLine("  Median:  ${"%.4f".format(median)}")
        appendLine("  Mode:    ${"%.4f".format(mode)}")
        appendLine("  Std Dev: ${"%.4f".format(std)}")
        appendLine("  Min:     ${"%.4f".format(min)}  |  Max: ${"%.4f".format(max)}")
        appendLine("  Q1:      ${"%.4f".format(q1)}  |  Q3:  ${"%.4f".format(q3)}")
    }

    companion object {
        fun from(name: String, values: List<Double>): ColumnStats {
            if (values.isEmpty()) return ColumnStats(name, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            val sorted = values.sorted()
            return ColumnStats(
                name    = name,
                count   = values.size,
                missing = 0,
                mean    = DataProcessor.mean(values),
                median  = DataProcessor.median(values),
                mode    = DataProcessor.mode(values),
                std     = DataProcessor.stdDev(values),
                min     = sorted.first(),
                max     = sorted.last(),
                q1      = DataProcessor.percentile(sorted, 25.0),
                q3      = DataProcessor.percentile(sorted, 75.0)
            )
        }
    }
}
