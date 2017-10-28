# 5615_P03B

## Structure of code
your_prefer_package_name/<br/>
|-- README.md<br/>
|-- __init__.py<br/>
|-- Influ.py<br/>
|-- influence/<br/>
|&nbsp;&nbsp;&nbsp;&nbsp; |-- smooth_hinge.py<br/>
|&nbsp;&nbsp;&nbsp;&nbsp;   |-- image_utils.py<br/>
|&nbsp;&nbsp;&nbsp;&nbsp;   |-- hessians.py<br/>
|&nbsp;&nbsp;&nbsp;&nbsp;   |-- genericNeuralNet.py<br/>
|&nbsp;&nbsp;&nbsp;&nbsp;   |-- inception_v3.py<br/>
|&nbsp;&nbsp;&nbsp;&nbsp;   |-- dataset.py<br/>
|&nbsp;&nbsp;&nbsp;&nbsp;   |-- __init__.py<br/>
|-- unittest/<br/>
|&nbsp;&nbsp;&nbsp;&nbsp;   |-- test.py<br/>
|-- scripts/<br/>
|&nbsp;&nbsp;&nbsp;&nbsp;   |-- data_transform.py<br/>
|&nbsp;&nbsp;&nbsp;&nbsp;   |-- rbf_test_fig.py<br/>
|&nbsp;&nbsp;&nbsp;&nbsp;   |-- load_animals.py<br/>
|&nbsp;&nbsp;&nbsp;&nbsp;   |-- rbf_test.py<br/>
|&nbsp;&nbsp;&nbsp;&nbsp;   |-- __init__.py<br/>
|-- source_datasets/<br/>
|&nbsp;&nbsp;&nbsp;&nbsp;   |-- supermarket_600.csv<br/>

### Prerequisites
+ Numpy
+ Scipy
+ Scikit-learn
+ Pandas
+ Tensorflow=v1.1.1
+ Keras=v2.0.4
+ Spacy
+ h5py=v2.7.0
+ Matplotlib
+ Seaborn<br/>
If you find it is hard to build the environment, you can also find the dockerfile provided by author of the paper in here: https://hub.docker.com/r/pangwei/tf1.1/
For tensorflow, the cpu version should be enough, but if you need to train with a large amount of data, it is better to use gpu version and then you will need cnn adn cuda kits from nvidia and a gpu as well.

### Usage of code
To use the code,<br/> 
1. Download the code
2. Put the code in a file with the name you prefer.
3. Now you can import the Influ into your script<br/>
The structure of this package and your scripts should be like:
```
a_directory
|-- project_code
|-- your_scripts
|-- your_dataset
```
*project_code is a directory for storing the code from this project

In your scripts:
```
from project_code.Influ import Influ
influ = Influ()
influ.load_data(your_dataset)
influ.convert(features_you_choose, label_you_choose)
influ.cal_influe()
```

### Documentation
In the Influ:
+ load_data(filename)<br/>
Load the data from your dataset, please be noticed that due to the design, the value of your label should be 1 and 2.
The format of dataset can be csv, txt or xlsx

+ convert(feature, label)
'feature' is a list of the features you choose. 'label' is your chosen label.This method will convert your dataset into a format that can be read by tensorflow and it will generate a compressed file called 'fake_data' which will be later read.

+ cal_influe(test_idx, gamma)
This will start training and compute influence function and display the plot for visualization automatically.
test_idx: This is the test data you choose as a standard to compute the euclidean distance for visualization.
gamma: This is the parameter for rbf kernel.

+ visualization(scale)
This will generate the plot again if you are not satisfied with the automatically generated plot. The reason for this is because the default scale for x axis and y axis is both 0.03. If you want to have better view on the distribution, you can set the scale by yoursef.

### Test
Test codes are situated in unittest file. But if you want to run the test, you have to copy it and place it like this:
```
a_directory
|-- project_code
|-- your_scripts
|-- test.py(this is the test code)
|-- your_dataset
```

