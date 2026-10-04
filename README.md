README



Reproducibility Package for Manuscript Submission



This package contains the code, training/validation data, and testing data used for the manuscript experiments. The material is arranged into three main folders:



1\. Code

2\. Experimental Reconstruction

3\. Parametric Generalization



1\. Code



The `Code` folder contains all model training and inference scripts. It is divided into two main parts:





1\. Code

1.1 Experimental Reconstruction

1.2 Parametric Generalization

```



Each of these folders contains four model folders:





DC-LSTM

├── model\_training\_code.py

└── model\_inference\_code.py



FiLM-LSTM

├── model\_training\_code.py

└── model\_inference\_code.py



FiLM-PhyLSTM

├── model\_training\_code.py

└── model\_inference\_code.py



PhyLSTM

├── model\_training\_code.py

└── model\_inference\_code.py



The four model folders correspond to the four model variants used in the manuscript. Each model folder contains one training script and one inference script.



The training script is used for training the model. The inference script is used for testing the trained model and generating the evaluation summary.





2\. Experimental Reconstruction Folder



The `Experimental Reconstruction` folder contains the data used for the experimental reconstruction study.



It is divided into three structural configuration folders:



Experimental Reconstruction

├── D60V

├── D75V

└── D100V



Each structural configuration folder is further divided into two folders:



Structural Configuration X/

├── Test

├── Train and Validation

```



2.1 Test Folder



The `Test` folder contains the testing files used for evaluation purpose. These files are used only for testing and are not used during model training.



2.2 Train and Validation Folder



The `Train and Validation` folder contains the DBE and MCE records used for training and validation.



Train and Validation

├── DBE

└── MCE

```

The training code uses these DBE and MCE files to construct the training and validation datasets.

\---



3\. Parametric Generalization Folder



The `Parametric Generalization` folder contains the data used for the parametric generalization study.



It is divided into three structural configuration folders:



Parametric Generalization

├── D60V

├── D75V

└── D100V

```



Each structural configuration folder is further divided into two folders:



Each structural configuration folder contains:

├── Training and Validation

├── Testing

```

The `Training and Validation` folder contains DBE and MCE data.



Training and Validation

├── DBE

└── MCE

```

The `DBE` and `MCE` folders are each divided into four folders corresponding to the known parameter cases used during training:



DBE

├── C348 and a0.30

├── C348 and a0.80

├── C870 and a0.30

└── C870 and a0.80



MCE

├── C348 and a0.30

├── C348 and a0.80

├── C870 and a0.30

└── C870 and a0.80

```



The training code asks the user to select files according to the folder name. Therefore, when the training script asks for a specific case, select the files from the matching case folder.



These four known cases form the training and validation distribution for the parametric generalization study.



\---



The `Testing` folder is divided into known and unknown testing cases.



Testing

├── Known Cases Testing Files

└── Unknown Cases Testing Files

```



Known Cases Testing Files



The `Known Cases Testing Files` folder contains testing files from the known parameter cases unseen during training. These cases belong to the same parameter distribution used during training.



Unknown Cases Testing Files



The `Unknown Cases Testing Files` folder contains testing files from unseen parameter cases. These cases are not used during model training and are used to evaluate parametric generalization.





How to Run Training



To train a model:



Step 1: Open the required training script from the Code folder.

Step 2: Run the training script.

Step 3: Select DBE and MCE files when prompted.

Step 4: For parametric generalization, select files according to the case folder names.

Step 5: Save the trained model checkpoint when prompted.

```

For parametric generalization, the training code asks for the four known cases separately. Select the files from the corresponding DBE and MCE case folders.



\---



How to Run Inference



To evaluate a trained model:



Step 1: Open the required inference script from the Code folder.

Step 2: Run the inference script.

Step 3: Select the trained model checkpoint obtained from training script.

Step 4: Select the required testing CSV files.

Step 5: Select the output folder.

Step 6: The script generates evaluation\_summary.csv.

```



The inference scripts generate only:

evaluation\_summary.csv

This file contains the evaluation results for each testing record used in manuscript.



Initial Transient Window Inference Script



The inference scripts use a fixed initial transient window:



INITIAL\_TRANSIENT\_SEC = 0.20



The following settings are used consistently:

IGNORE\_INITIAL\_TRANSIENT\_FOR\_METRICS = True

APPLY\_INITIAL\_TRANSIENT\_CLAMP = True

This means:

1\. The first 0.20 seconds are excluded from metric calculation.

2\. The predictions before the initial transient index are clamped.

3\. The same treatment is applied to all testing files.



This is used only to avoid boundary-related starting effects during sequence inference.

Important Note



The folder names are used only for organizing the dataset and helping the user select the correct files during training and inference. The model codes read the required numerical values from the CSV files themselves. All computations were performed in Python 3.11.12 using PyTorch 2.5.1 and CUDA 12.1 on a 64-bit Windows workstation with an Intel Core i9-14900HX CPU, 16 GB RAM, and an NVIDIA GeForce RTX 4060 GPU. For experimental reconstruction, each model was trained using the response data from all three floors independently, whereas parametric generalization was performed using only the first-floor damper response data. The experimental reconstruction files are arranged according to the selected best test set, consistently applied across all models and structural configurations.



\---



Contact



For questions about the dataset, code, model training, please contact the corresponding author of the manuscript.

