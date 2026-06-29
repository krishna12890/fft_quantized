This is a quantized model of **256-point FFT(DIF)**.<br> 
It simulates an input from 12-bit ADC. It is qunatized to 16-bit Fixed-Point. It uses, **Block-Wise** quantization strategy, to take care of scaling of FFT results at each stage.

1. Note there is a requirements.txt file. (make sure to install of them : *pip install -r requirements.txt*).
2. The FFT Floating point model takes a sinewave at tone bin defined by the variable *tone_bin* in **./src/python/fft_floating_point.py**.
3. The FFT Quantized model takes a sinewave at tone bin defined by the variable *tone_bin* in **./src/python/fft_quantized.py**.
4. Plots are saved to the **./runs** directory regardless of where you run the scripts from.
5. Upon running **fft_floating_point.py** you will see the following Results (in comparison with the NumPy FFT module) and Plots at **./runs/fft_floating_point_plot.png**.<br>
   **- Matches NumPy : True/False**<br>
6. Upon running **fft_quantized.py** you will see the following Results (in comparison with the NumPy FFT module) and Plots at **./runs/fft_quantized_plot.png**.<br>
   **- RMSE :**<br>
   **- Mean magnitude :**<br>
   **- Normalized RMSE :**<br>
   **- SNR Comparison :**<br>
