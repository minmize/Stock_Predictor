package com.neuralnet.builder.tutorial

data class TutorialPage(
    val title: String,
    val emoji: String,
    val body: String,       // Markdown-lite text shown in the app
    val visual: TutorialVisual = TutorialVisual.NONE,
    val tip: String = ""
)

enum class TutorialVisual {
    NONE,
    NEURON_DIAGRAM,
    LAYER_DIAGRAM,
    NETWORK_DIAGRAM,
    TRAINING_LOOP,
    ACTIVATION_GRAPH,
    LOSS_CURVE
}

object TutorialData {

    val chapters: List<TutorialChapter> = listOf(

        TutorialChapter(
            title = "What is AI?",
            icon = "🤖",
            pages = listOf(
                TutorialPage(
                    title = "Artificial Intelligence Basics",
                    emoji = "🧠",
                    body = """
Artificial Intelligence (AI) means giving computers the ability to do tasks that normally require human intelligence.

Examples of AI in daily life:
  • Face unlock on your phone
  • Spam filters in email
  • Music recommendations on Spotify
  • Autocomplete when you type

AI is a huge field. One of its most powerful tools is called a Neural Network.
                    """.trimIndent(),
                    visual = TutorialVisual.NONE,
                    tip = "AI doesn't mean robots! Most AI is software running quietly on servers."
                ),
                TutorialPage(
                    title = "Machine Learning vs. Programming",
                    emoji = "🔄",
                    body = """
Traditional programming:
  You write rules → Computer follows rules → Output

Machine Learning:
  You give examples → Computer learns rules → Output

With machine learning, instead of telling the computer how to recognise a cat, you show it thousands of cat photos and it figures out the rules itself!

Neural networks are one type of machine learning model.
                    """.trimIndent(),
                    tip = "Neural networks are especially good when the rules are too complex to write by hand."
                )
            )
        ),

        TutorialChapter(
            title = "Neurons",
            icon = "⚡",
            pages = listOf(
                TutorialPage(
                    title = "The Biological Inspiration",
                    emoji = "🧬",
                    body = """
Your brain has ~86 billion neurons (nerve cells). Each neuron:
  1. Receives signals from other neurons
  2. Adds them up
  3. Fires its own signal if the total is strong enough

Artificial neurons work the same way — just with numbers!
                    """.trimIndent(),
                    visual = TutorialVisual.NEURON_DIAGRAM,
                    tip = "Each artificial neuron does only one simple calculation, but billions working together create intelligence!"
                ),
                TutorialPage(
                    title = "How an Artificial Neuron Works",
                    emoji = "⚙️",
                    body = """
An artificial neuron does three steps:

Step 1 — Receive inputs:
  Each input (x₁, x₂, x₃...) has a weight (w₁, w₂, w₃...)
  Weights say how important each input is.

Step 2 — Weighted sum:
  z = w₁×x₁ + w₂×x₂ + w₃×x₃ + bias

Step 3 — Activation:
  output = activation_function(z)

The activation function decides whether the neuron "fires" and how strongly.
                    """.trimIndent(),
                    visual = TutorialVisual.NEURON_DIAGRAM,
                    tip = "The 'bias' is like a threshold. It lets the neuron fire even when all inputs are zero."
                )
            )
        ),

        TutorialChapter(
            title = "Layers",
            icon = "📚",
            pages = listOf(
                TutorialPage(
                    title = "Building a Network from Layers",
                    emoji = "🏗️",
                    body = """
Neurons are organised into layers:

Input Layer:
  Receives raw data. One neuron per feature.
  (e.g., 4 neurons for 4 columns of data)

Hidden Layers:
  Learn patterns. You can have 0, 1, or many.
  More hidden layers = more complex patterns.

Output Layer:
  Produces the final result.
  1 neuron for regression, N neurons for N classes.
                    """.trimIndent(),
                    visual = TutorialVisual.LAYER_DIAGRAM,
                    tip = "A network with no hidden layers can only learn straight-line patterns. Add hidden layers for complex data!"
                ),
                TutorialPage(
                    title = "How Many Neurons?",
                    emoji = "🔢",
                    body = """
Choosing the right size is part science, part art.

Rules of thumb:
  • Hidden layer size: between input and output size
  • Start small, add more if accuracy is low
  • More neurons = more power but slower training

Example architectures:
  Simple:    4 → 8 → 1
  Medium:    10 → 64 → 32 → 1
  Complex:   100 → 256 → 128 → 64 → 10

In this app, just type your structure like: 4-64-32-1
                    """.trimIndent(),
                    visual = TutorialVisual.NETWORK_DIAGRAM,
                    tip = "When in doubt, try: input_size → 64 → 32 → output_size."
                )
            )
        ),

        TutorialChapter(
            title = "Activation Functions",
            icon = "📈",
            pages = listOf(
                TutorialPage(
                    title = "Why Do We Need Activation Functions?",
                    emoji = "🎛️",
                    body = """
Without activation functions, a neural network is just matrix multiplication — it can only learn linear patterns (straight lines).

Activation functions add non-linearity, allowing the network to learn curves, waves, and complex shapes.

Think of it like folding paper. Each activation function is one fold. More layers = more folds = can model almost anything!
                    """.trimIndent(),
                    visual = TutorialVisual.ACTIVATION_GRAPH
                ),
                TutorialPage(
                    title = "The Most Common Activations",
                    emoji = "📊",
                    body = """
ReLU (Rectified Linear Unit):
  output = max(0, x)
  ✓ Fast, works well in hidden layers
  ✓ Best default choice for hidden layers

Sigmoid:
  output = 1 / (1 + e^-x)
  ✓ Outputs 0–1, great for binary output
  ✗ Can slow down learning (vanishing gradient)

Tanh:
  output = tanh(x)  (range: -1 to 1)
  ✓ Usually better than Sigmoid for hidden layers

Softmax:
  Converts outputs to probabilities that sum to 1
  ✓ Use in output layer for multi-class problems

Linear:
  output = x  (no change)
  ✓ Use in output layer for regression
                    """.trimIndent(),
                    visual = TutorialVisual.ACTIVATION_GRAPH,
                    tip = "Quick guide: Hidden layers → ReLU. Binary output → Sigmoid. Multi-class output → Softmax. Regression output → Linear."
                )
            )
        ),

        TutorialChapter(
            title = "Training",
            icon = "🏋️",
            pages = listOf(
                TutorialPage(
                    title = "The Training Loop",
                    emoji = "🔄",
                    body = """
Training is how a network learns. It repeats these steps:

1. Forward Pass
   Feed input data through the network → get a prediction.

2. Calculate Loss
   Compare prediction to the true answer. How wrong is it?

3. Backward Pass (Backpropagation)
   Work backwards to find which weights caused the error.

4. Update Weights
   Nudge each weight slightly to make the prediction better.

Repeat thousands of times. The network gradually gets better!
                    """.trimIndent(),
                    visual = TutorialVisual.TRAINING_LOOP,
                    tip = "One full pass over ALL your training data is called an 'epoch'."
                ),
                TutorialPage(
                    title = "Gradient Descent",
                    emoji = "⛰️",
                    body = """
Imagine you're on a foggy mountain and want to reach the valley (lowest loss). You can't see far, so you take small steps downhill.

That's gradient descent! The "gradient" tells you which direction is downhill, and you take a step in that direction.

Key hyperparameter:
  Learning Rate — size of each step
  • Too large: you overshoot the valley
  • Too small: training takes forever
  • Good default: 0.001

This app uses SGD with Momentum — it remembers which direction you were going and keeps some of that momentum (like a ball rolling downhill).
                    """.trimIndent(),
                    tip = "If your loss is jumping up and down wildly, try a smaller learning rate."
                )
            )
        ),

        TutorialChapter(
            title = "Loss Functions",
            icon = "📉",
            pages = listOf(
                TutorialPage(
                    title = "What is a Loss Function?",
                    emoji = "🎯",
                    body = """
A loss function measures how wrong your network is. Lower loss = better network.

Mean Squared Error (MSE):
  loss = average((prediction - truth)²)
  ✓ Use for regression (predicting numbers)
  Example: predicting house prices

Binary Cross-Entropy:
  loss = -[y·log(p) + (1-y)·log(1-p)]
  ✓ Use for yes/no (0 or 1) classification
  Example: spam detection

Categorical Cross-Entropy:
  loss = -sum(y · log(p))
  ✓ Use for multi-class classification
  Example: digit recognition (0-9)
                    """.trimIndent(),
                    visual = TutorialVisual.LOSS_CURVE,
                    tip = "Match your loss function to your task, not your data!"
                )
            )
        ),

        TutorialChapter(
            title = "Overfitting",
            icon = "⚠️",
            pages = listOf(
                TutorialPage(
                    title = "Overfitting vs Underfitting",
                    emoji = "📏",
                    body = """
Underfitting:
  The model is too simple to learn the pattern.
  Both training and validation loss are high.
  Fix: Add more layers, neurons, or train longer.

Overfitting:
  The model memorised the training data but doesn't generalise.
  Training loss is low, but validation loss is high.
  Fix: Use less data, add dropout, collect more data.

Good fit:
  Both training and validation loss are low and close together.
  This is the goal!

Watching both training and validation loss while training is called "monitoring". This app shows you both!
                    """.trimIndent(),
                    visual = TutorialVisual.LOSS_CURVE,
                    tip = "Always use a validation set. Never make decisions based only on training loss!"
                ),
                TutorialPage(
                    title = "Dropout (Regularisation)",
                    emoji = "🎲",
                    body = """
Dropout is a simple technique to fight overfitting.

During training, dropout randomly "turns off" a fraction of neurons each step. This forces the network to learn redundant representations — it can't rely on any single neuron.

Dropout rate of 0.2 means 20% of neurons are randomly disabled each training step.

In this app, you can set a dropout rate (0–0.5) for any layer in the network builder.

At inference time (predictions), all neurons are active.
                    """.trimIndent(),
                    tip = "A dropout rate of 0.2–0.5 is typical. Too high and the network can't learn."
                )
            )
        ),

        TutorialChapter(
            title = "Your First Model",
            icon = "🚀",
            pages = listOf(
                TutorialPage(
                    title = "Step-by-Step Guide",
                    emoji = "📋",
                    body = """
Ready to build your first neural network? Here's the process:

Step 1 — Prepare your data
  Get a CSV file with your data.
  Make sure one column is your "target" (what to predict).

Step 2 — Build your network
  Tap "Build Network" and type your architecture.
  Example for a simple problem: 4-32-16-1

Step 3 — Load your data
  Upload your CSV, pick feature and target columns.
  Choose how to process missing values.

Step 4 — Train!
  Watch the loss go down as your network learns.
  Stop when validation loss stops improving.

Step 5 — Evaluate
  Check accuracy and make predictions on new data.
                    """.trimIndent(),
                    tip = "Start simple! A small network with clean data beats a huge network with messy data."
                )
            )
        )
    )

    val allPages: List<TutorialPage> get() = chapters.flatMap { it.pages }
}

data class TutorialChapter(
    val title: String,
    val icon: String,
    val pages: List<TutorialPage>
)
