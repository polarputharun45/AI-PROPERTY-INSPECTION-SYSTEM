import kagglehub

# Download houses-in-london dataset
path = kagglehub.dataset_download("oktayrdeki/houses-in-london")
print("Dataset path:", path)

# Also download house room dataset
path2 = kagglehub.dataset_download("gpiosenka/house-room-20-class")
print("House room path:", path2)
