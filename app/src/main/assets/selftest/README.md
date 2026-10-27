`runtime_selftest.tflite` is NOT a health model. It is a 4-input linear function (y = 0.5*x0 - x1 + 2*x2 + 0.25*x3 + 0.1)
used only by the "System check" screen to prove that the TensorFlow Lite runtime loads and runs on this phone.
It lives outside `assets/models/`, so it can never be used as a vital-sign estimator.
