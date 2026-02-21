<!-- To generate/update the Arduino header from the latest TFLite model -->

# Run from this `model/` folder:
xxd -i cough_cnn1_int8.tflite > ../Arduino/main/model_data1.h
