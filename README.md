# PATE Example

The scripts in the folder allow you to train a MNIST model using the PATE differential privacy framework. 
While running this example would give you a working implementation of PATE, an accurate analysis of DP guarantees is still a work in progress.

## Requirements:

* PyTorch
* PySyft

```bash
$ python Main.py
```
Scripts present:
data: Consists of functions for loading datasets.

Main: The file to be run for a complete PATE model.

Model: PyTorch model definition. The same model is used for student and teacher.

Student: Class to handle student functionality such as training and making predictions.

Teacher: Class to handle teacher functionality such as training and making noisy predictions. All the Teacher ensembles are handled in this Class.  

util: Helper functions.  

This training loop is then executed for the student model to effectively learn from the aggregated teacher predictions.  

Inspiration & Acknowledgement: Please note that the design and implementation of this repository were inspired by kamathhrishi/PATE（https://github.com/kamathhrishi/PATE）.
