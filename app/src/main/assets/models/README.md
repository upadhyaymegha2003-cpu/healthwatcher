# On-device models go here

Place a trained model and its manifest side by side, named after the vital sign:

    heart_rate.tflite          heart_rate.properties
    respiratory_rate.tflite    respiratory_rate.properties
    spo2.tflite                spo2.properties
    blood_pressure.tflite      blood_pressure.properties

Generate them with `ml_training/train.py` from REAL labelled recordings (see ml_training/README.md).
With no files here the app uses the original signal-processing algorithms and says so on screen.
