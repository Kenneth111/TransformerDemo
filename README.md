# Pytorch nn.Transformer Demo
A demo to predict odd numbers. Given the input [2, 4, 6], this program generates the output [3, 5, 7]. Given the input [100, 102, 104], this program generates the output [101, 103, 105].

Create a folder named "model", where the weights of trained model will be saved, and train the model using
```shell script
python predict_odd_numbers.py
```
The validation loss will be around 1.7.
  
Test the model using
```shell script
python predict_odd_numbers.py --test_model model/xxx.pt
```

Codes in the model.py come from [this notebook](https://colab.research.google.com/drive/1g4ZFCGegOmD-xXL-Ggu7K5LVoJeXYJ75).
