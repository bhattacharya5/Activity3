import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
        
        # Compute and visualize correlation matrix for numeric columns
        numeric_cols = abData.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            print("Correlation Matrix:")
            corr_matrix = abData[numeric_cols].corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title('Correlation Matrix')
            plt.show()
        else:
            print("No numeric columns found for correlation analysis.")
        
        # Calculate and visualize percentage of missing values in each column
        missing_percentage = (abData.isnull().mean() * 100).sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        missing_percentage.plot(kind='bar', color='skyblue')
        plt.title('Percentage of Missing Values in Each Column')
        plt.xlabel('Columns')
        plt.ylabel('Percentage of Missing Values')
        plt.show()

def main():
    abFilePath = "DryBeanDataset/Dry_Bean_Dataset.xlsx"
    abData = load_data(abFilePath)
    analyze_data(abData)

if __name__ == "__main__":
    main()
