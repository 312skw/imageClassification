from datasets import load_dataset

#for local dataset, check before using to determine need for data preprocessing
dataset = load_dataset("imagefolder", data_dir="structuraldamage")
print(dataset)

#image data needs to be in separate folders for each class in both the train and test (and validation) directories,
#as huggingface auto-labels binary data (with structural damage, no structural damage)
#do not manually label the image data with any metadata, as it will not be used
#e.g. all images in the damaged test and train set will be labeled 1, and all images
#in the undamaged test and train set will be labeled 0, the model will predict if an image is 1 or 0
#keep the folder names in all the directories consistent, otherwise the auto-labels will be mismatched
#do not put any other files in the dataset folder, the auto-labeling will not work correctly 
#structure should look like this:
#datasetfolder/
#   train/
#       damaged/
#       undamaged/
#   test/
#       damaged/
#       undamaged/
#   validation/
#       damaged/
#       undamaged/