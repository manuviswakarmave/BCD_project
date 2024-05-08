import os
import matplotlib.pyplot as plt
from PIL import Image
import segmentation
from prediction import predict


# Load the trained model
model = segmentation.load_segmentation_model("segmentation_model.pth")


# Function to load and preprocess image
def load_image(file_path):
    img = Image.open(file_path)
    img = segmentation.segment_image(img, model)
    segmented_image = Image.fromarray(img)
    return segmented_image


# Path to the directory containing the subfolders
main_dir = "grapj"
# List all files in the folder
files = os.listdir(main_dir)
# Calculate the total number of files
total_files = len(files)

intervals = [int(total_files * (i / 10)) for i in range(0, 11)]

# Lists to store true labels and predicted labels
true_labels = []
predicted_labels = []

accuracies =[0]
recalls=[0]
precisions=[0]


# Initialize counts
tp_count = 0
fp_count = 0
tn_count = 0
fn_count = 0

index = 0

# Iterate through subfolders
for file in os.listdir(main_dir):
        # Iterate through images in the subfolder
            if file.endswith(".png"):
                index+=1
                # Load image
                segmented_img = load_image(os.path.join(main_dir, file))
                # Make prediction
                prediction = predict(segmented_img.convert('RGB'))
                # Update true labels
                true_labels.append(0 if 'benign' in file.lower()  else 1)
                # Update predicted labels
                predicted_labels.append(prediction)

                # Update counts based on true label and prediction
                if true_labels[-1] == 0 and predicted_labels[-1] == 0:
                    tp_count += 1
                elif true_labels[-1] == 1 and predicted_labels[-1] == 0:
                    fp_count += 1
                elif true_labels[-1] == 0 and predicted_labels[-1] == 1:
                    fn_count += 1
                else:  # true_labels[-1] == 0 and predicted_labels[-1] == 0
                    tn_count += 1

                if(index in intervals):
                    accuracies.append(( tp_count + tn_count ) / (tp_count + tn_count + fn_count + fp_count))
                    recalls.append(tp_count  / (tp_count + fn_count))
                    precisions.append(tp_count/ (tp_count + fp_count))



print(tp_count, tn_count, fn_count, fp_count)
print(total_files)
print(intervals)
print("accuracies : ", accuracies )
print("recalls : ", recalls)
print("precisions" , precisions)

# Plotting
plt.plot(intervals, accuracies, label='Accuracy')
plt.plot(intervals, recalls, label='Recall')
plt.plot(intervals, precisions, label='Precision')

# Add a point at the origin
plt.plot(0, 0, marker='o', markersize=5, color="black")

# Set labels and title
plt.xlabel('Percentage of Input')
plt.ylabel('Metrics')
plt.title('Metrics vs Percentage of Input')

# Set x-axis ticks
plt.xticks(intervals)

# Set y-axis range and ticks
plt.ylim(0, 1.1)
plt.yticks([i/10 for i in range(11)])

# Show legend
plt.legend()

# Show plot
plt.grid(True)
plt.show()