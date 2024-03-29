import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def load_data(abFilePath):
    """Load data from an Excel file."""
    try:
        abData = pd.read_excel(abFilePath)
        return abData
    except FileNotFoundError:
        print("File not found. Please provide a valid file path.")
        return None
    except Exception as abError:
        print(f"An error occurred: {abError}")
        return None

def analyze_data(abData):
    """Perform basic data analysis."""
    if abData is not None:
        # Display summary statistics
        print("Summary Statistics:")
        print(abData.describe())

        # Plot histograms for numeric columns
        print("Histograms:")
        for abCol in abData.select_dtypes(include=['int64', 'float64']):
            abData[abCol].plot(kind='hist', bins=10)
            plt.title(abCol)
            plt.xlabel(abCol)
            plt.ylabel('Frequency')
            plt.show()
        
        # Encode categorical class labels
        if 'Class' in abData.columns:
            abLe = LabelEncoder()
            abData['Class'] = abLe.fit_transform(abData['Class'])
            
            # Plot bar plot for the class label (numeric type)
            abClassLabelCounts = abData['Class'].value_counts()
            abClassLabelCounts.plot(kind='bar')
            plt.title('Class Label Distribution')
            plt.xlabel('Class Label')
            plt.ylabel('Count')
            plt.show()
        else:
            print("Class column not found.")

def main():
    abFilePath = "DryBeanDataset/Dry_Bean_Dataset.xlsx"
    abData = load_data(abFilePath)
    analyze_data(abData)

if __name__ == "__main__":
    main()
