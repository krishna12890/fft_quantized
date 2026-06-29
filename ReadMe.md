This is a quantized model of **256-point FFT(DIF)**. It simulates an input from 12-bit ADC. It is qunatized to 16-bit Fixed-Point. It uses, **Block-Wise** quantization strategy, to take care of scaling of FFT results at each stage.

1. Note there is a requirements.txt file. (make sure to run : *pip install -r requirements.txt*)
2. The best place to run all the simulations in the **./runs** directory. (The python script to run is ./src/fft_quantized.py). (In the runs directory launch : *python3 ../src/fft_quantized.py*)
3. Upon running you will see the following Results (in comparision with the Numpy FFT module) and Plots of the FFT results at **./runs/fft_plot.png**.
**    RMSE : **
** - Mean magnitude : **
** - Normalized RMSE : **
** - SNR Comparision : **
