# seasonal_influenza
This contains the codes necessary to compare various MCMC inference methods for a simple model of seasonal influenza epidemic in England

# Dependencies 
Other than the normal scientific python stack to use this library one needs to install:
1) Pybind11
2) Eigen
I recommend using this library within a virtual environment such as one provided by Anaconda. 
Install Pybind:
`conda install -c conda-forge pybind11 `
Since Eigen is header only thus it is easy to install (see documentaion [here](http://eigen.tuxfamily.org/index.php?title=Main_Page#Download))

To link the model(s) written in C++ and the MCMC code written in Python you have to appropriately modify the `setup.py` file in the `CPP` directory.
# Running the code
Currently the `test_inference_icu.py` file implement Bayesian inference of the parameters of the epidemic model using two: i) RWM MCMC and ii) Adaptive MCMC. The inference uses simulated data for now. To simulate other data than the one used in this example see the `fake_data` directory, `trycode.R` and `cppfun.cpp` file for the simple ICU model.

# Anaconda-free installation instructions
NB: assumes Python 3.

1. Clone the repo:
```git clone https://github.com/sanmitraghosh/seasonal_influenza.git```

1. (Optional) Create a virtualenv and activate it:
```virtualenv env && source env/bin/activate```

1. Install the dependencies:
```pip install numpy matplotlib seaborn scipy pybind11```

1. Download [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) and extract it to somewhere you can find.

1. Replace `/scratch/sanmitra/test_sanmitra/eigen3` with `/PATH/TO/YOUR/eigen` in **both** places in `CPP/setup.py`.

1. Run the code!
```python seasonal_influenze/test_inference_icu.py```
