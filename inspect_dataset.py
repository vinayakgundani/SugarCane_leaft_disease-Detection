import os

dataset_path = "Dataset"

classes = os.listdir(dataset_path)

print("Classes:")
print(classes)

print("\nNumber of images in each class:")

for class_name in classes:
    class_path = os.path.join(dataset_path, class_name)

    images = os.listdir(class_path)

    print(class_name, ":", len(images))