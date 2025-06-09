import os

#Step 1: Run the data processing
print("Running data extraction...")
os.system("python data.py")

#Train and test the model
print(" Starting model training...")
os.system("python train.py")