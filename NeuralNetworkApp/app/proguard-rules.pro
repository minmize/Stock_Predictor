# Add project specific ProGuard rules here.

# Keep NeuralNetwork engine classes (used via reflection in saved models)
-keep class com.neuralnet.builder.engine.** { *; }
-keep class com.neuralnet.builder.data.** { *; }

# MPAndroidChart
-keep class com.github.mikephil.charting.** { *; }

# OpenCSV
-keep class com.opencsv.** { *; }

# Gson
-keepattributes Signature
-keepattributes *Annotation*
-keep class com.google.gson.** { *; }
