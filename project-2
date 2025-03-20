#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <string>

using namespace std;

// Function Declarations
pair<int, int> readTableFromFile(const string& filename, vector<vector<double>>& data, vector<int>& labels);
void normalizeData(vector<vector<double>>& data);
double euclideanDistance(const vector<double>& a, const vector<double>& b);
double LOOCV(const vector<vector<double>>& data, const vector<int>& labels);
vector<int> forwardSelection(const vector<vector<double>>& data, const vector<int>& labels, int numFeatures);
vector<vector<double>> selectFeatures(const vector<vector<double>>& data, const vector<int>& selectedFeatures);
vector<int> backwardElimination(const vector<vector<double>>& data, const vector<int>& labels, int numFeatures);

// Main Function
int main() {
    // VARIABLES
    string filename = "";

    int algorithm_choice = 0;
  
    vector<vector<double>> data; // 2D vector to hold the dataset
    vector<int> labels; // 1D vector to hold the class labels

    cout << "Welcome to Feature Selection Algorithm" << endl;
    cout << "Type in the name of the file to test: ";
    cin >> filename; // Read the filename from user input

    cout << " Type the number of the algorithm you want to run." << endl << endl;
    cout << "1) Forward Selection" << endl;
    cout << "2) Backward Elimination" << endl;
    cin >> algorithm_choice;
    while (algorithm_choice != 1 && algorithm_choice != 2) {
        cout << "Please select a valid algorithm:" << endl;
        cin.clear();
        cin.ignore(numeric_limits<streamsize>::max(), '\n'); // Clear the input buffer
        cin >> algorithm_choice;        
    }

    pair<int, int> result = readTableFromFile(filename, data, labels);
    int numInstances = result.first; // Number of instances in the dataset
    int numFeatures = result.second; // Number of features in the dataset (not including the class attribute)
    
    // Check if data is valid
    if (numInstances > 0 && numFeatures > 0) {
        normalizeData(data); // Normalize the data
        // Output the dataset information
        cout << endl << "This dataset has " << numFeatures << " features (not including the class attribute), with " << numInstances << " instances." << endl << endl;
        // Perform Leave-One-Out Cross-Validation (LOOCV) to get accuracy with all features
        double accuracy = LOOCV(data, labels);
        cout << "Running nearest neighbor with all " << numFeatures << " features, using \"leave-one-out\" evaluation, I get an accuracy of " << accuracy << "%" << endl << endl;
    }

    // Choose and run the feature selection algorithm based on user input
    if (algorithm_choice == 1) {
        cout << "You have selected Forward Selection!" << endl;
        forwardSelection(data, labels, numFeatures); // Run forward selection
    } else if (algorithm_choice == 2) {
        cout << "You have selected Backward Elimination!" << endl;
        backwardElimination(data, labels, numFeatures); // Run backward elimination
    }
    return 0;
}

/*
    * Function: readTableFromFile
    * -----------------------------
    * Reads a table of data from a file and returns it as a 2D vector of doubles.
    *
    * filename: the name of the file to read from
    *
    * returns: a 2D vector containing the data from the file, or an empty vector if the file could not be opened
*/
pair<int, int> readTableFromFile(const string& filename, vector<vector<double>>& data, vector<int>& labels) {
    ifstream inFS;
    inFS.open(filename); // Open the file for reading
    if (!inFS.is_open()) {
        cout << "Error opening file: " << filename << endl;
        return {}; // Return an empty pair if the file could not be opened
    } else {
        int numInstances = 0;
        int numFeatures = 0;
        string line;

        // Read the file line by line
        while (getline(inFS, line)) {
            stringstream ss(line); // Create a stringstream to parse the line
            vector<double> features;
            double value;
            int label;
            
            ss >> label; // Read the class label (first value in the line)
            labels.push_back(label); // Store the label

            // Read the feature values of the instance
            while (ss >> value) {
                features.push_back(value);
            }

            if (numInstances == 0) {
                numFeatures = features.size(); // Set the number of features based on the first line
            }
            data.push_back(features); // Add the instance's features to the data
            numInstances++;
        }
        inFS.close(); // Close the file
        return {numInstances, numFeatures}; // Return the number of instances and features
    }
}

/*
    * Function: normalizeData
    * -----------------------------
    * Reads a  2D vector of doubles and normalizes the data.
    *
    * data: a 2D vector containing the data to be normalized
    *
    * returns: NA
*/
void normalizeData(vector<vector<double>>& data) {
    int numRows = data.size();
    int numCols = data[0].size();
    double value = 0.0;
    double mean = 0.0;
    double stdDev = 0.0;

    // Loop through each feature (excluding the first column of class labels)
    for (unsigned int col = 1; col < numCols; col++) {
        double minVal = data[0][col];
        double maxVal = data[0][col];

        // Calculate mean and standard deviation for each feature
        for (unsigned int row = 0; row < numRows; row++) {
            value += data[row][col];
        }

        mean = value / numRows;

        for (unsigned int row = 0; row < numRows; row++) {
            stdDev += pow(data[row][col] - mean, 2); // Variance
        }
        stdDev = sqrt(stdDev / numRows); // Standard deviation
        // Normalize the feature using the mean and standard deviation
        for (unsigned int row = 0; row < numRows; row++) {
            data[row][col] = (data[row][col] - mean) / stdDev; // Standardized data
        }
    }
}

/*
* Function: euclideanDistance
* -----------------------------
* Computes the Euclidean distance between two vectors of doubles.
*
* a: the first vector
* b: the second vector
*
* returns: the Euclidean distance between the two vectors
*/
double euclideanDistance(const vector<double>& a, const vector<double>& b) {
    double distance = 0.0;
    if (a.size() != b.size()) {
        cout << "Error: Vectors must be of the same size to compute distance." << endl;
        return -1; // Return -1 to indicate an error
    }
    // Calculate the squared differences and sum them up
    for (unsigned int i = 0; i < a.size(); i++) {
        distance += ((a[i] - b[i]) * (a[i] - b[i])); // Calculate squared difference
    }
    return sqrt(distance); // Return the square root of the sum of squared differences (Euclidean distance)
}

/*
* Function: LOOCV
* -----------------------------
* Performs Leave-One-Out Cross-Validation (LOOCV) on the dataset.
*
* data: a 2D vector containing the dataset, where each row is an instance and the first column is the class label
* 
* returns: a double representing the classification accuracy (the proportion of correctly classified instances)
*/
double LOOCV(const vector<vector<double>>& data, const vector<int>& labels) {
    int correctCount = 0;

    if (data.empty()) {
        cout << "Error: No data available for LOOCV." << endl;
        return 0.0;
    }

    // For each instance, leave it out and use the rest of the data to classify it
    for (unsigned int i = 0; i < data.size(); i++) {
        double minDistance = numeric_limits<double>::max();
        int predictedLabel = -1;

        // Loop through eac other instance to find the nearest neighbor
        for (unsigned int j = 0; j < data.size(); j++) {
            if (i == j) continue;

            double distance = euclideanDistance(data[i], data[j]); // Compute distance
            
            if (distance < minDistance) {
                minDistance = distance;
                predictedLabel = labels[j]; // Get the class label of the nearest neighbor
            }
        }
        if (predictedLabel == labels[i]) {
            correctCount++; // Increment correct count if the label matches the predicted label
        }
    }

    double answer = (static_cast<double>(correctCount) / static_cast<double>(data.size())) * 100.0; // Calculate accuracy as a percentage
    return answer; // Return the classification accuracy as percentage
}

/*
* Function: selectFeatures
* -----------------------------
* Selects the features from the dataset based on the indices provided in selectedFeatures.
*
* data: a 2D vector containing the dataset, where each row is an instance and the first column is the class label
* selectedFeatures: a vector of integers representing the indices of the features to select
*
* returns: a 2D vector containing the selected features from the dataset
*/
vector<vector<double>> selectFeatures(const vector<vector<double>>& data, const vector<int>& selectedFeatures) {
    vector<vector<double>> selectedData;
    for (const auto& instance : data) {
        vector<double> selectedInstance;
        for (int feature : selectedFeatures) {
            if (feature >= 0 && feature < instance.size()) {
                selectedInstance.push_back(instance[feature]); // Select the feature based on the index
            } else {
                cout << "Error: Feature index " << feature << " is out of bounds for the instance." << endl;
                return {}; // Return empty vector if feature index is out of bounds
            }
        } 
        selectedData.push_back(selectedInstance);
    }
    return selectedData;
}

/* 
* Function: forwardSelection
* -----------------------------
* Performs forward selection to find the best subset of features for classification.
*
* data: a 2D vector containing the dataset, where each row is an instance and the first column is the class label
* labels: a vector of integers representing the class labels for each instance
* numFeatures: the total number of features in the dataset
*
* returns: a vector of integers representing the indices of the selected features
*/
vector<int> forwardSelection(const vector<vector<double>>& data, const vector<int>& labels, int numFeatures) {
    vector<int> bestFeatureSet;
    vector<int> currentFeatureSet;
    double bestOverallAccuracy = 0.0;

    cout << "Beginning search." << endl << endl;

    for (unsigned int i = 0; i < numFeatures; i++) {
        int bestFeature = -1;
        double bestAccuracy = -1;

        // Iterate over all features to find the best one to add
        for (int feature = 0; feature < numFeatures; feature++) { 
            if (find(currentFeatureSet.begin(), currentFeatureSet.end(), feature) == currentFeatureSet.end()) {
                vector<int> tempFeatureSet = currentFeatureSet;
                tempFeatureSet.push_back(feature);
                vector<vector<double>> reducedData = selectFeatures(data, tempFeatureSet); // Select features for the current iteration
                double accuracy = LOOCV(reducedData, labels); // Calculate accuracy with the current feature set
                
                // Display current feature set and its accuracy
                cout << "Using feature(s) {";
                for (unsigned int j = 0; j < tempFeatureSet.size(); j++) {
                    cout << tempFeatureSet[j] + 1;
                    if (j < tempFeatureSet.size() - 1) {
                        cout << ", ";
                    }
                }

                // Update best feature if accuracy is improved
                cout << "}, accuracy = " << accuracy << "%" << endl; // EDITED
                if (accuracy > bestAccuracy) {
                    bestAccuracy = accuracy;
                    bestFeature = feature;
                }
            }
        }

        // Add the best feature found in this iteration
        if (bestFeature != -1) {
            currentFeatureSet.push_back(bestFeature);
            cout << "\nFeature set {";
            for (unsigned int j = 0; j < currentFeatureSet.size(); j++) {
                cout << currentFeatureSet[j] + 1;
                if (j < currentFeatureSet.size() - 1) {
                    cout << ", ";
                }
            }
            cout << "} was best, accuracy is " << bestAccuracy << "%" << endl << endl;
            
            // Update best overall feature set
            if (bestAccuracy > bestOverallAccuracy) {
                bestOverallAccuracy = bestAccuracy;
                bestFeatureSet = currentFeatureSet; // Update the best feature set
            } else {
                cout << "(Warning, Accuracy has decreased! Continuing search in case of local maxima)" << endl;
            }
        }
    }

    // Output the best feature set found
    cout << "\nFinished search!! The best feature subset is {";
    for (unsigned int j = 0; j < bestFeatureSet.size(); j++) {
        cout << bestFeatureSet[j] + 1;
        if (j < bestFeatureSet.size() - 1) {
            cout << ", ";
        }
    }
    cout << "} with an accuracy of " << bestOverallAccuracy << "%" << endl;
    return bestFeatureSet; // Return the best feature set found
}

/*
* Function: backwardElimination
* -----------------------------
* Performs backward elimination to find the best subset of features for classification.
*
* data: a 2D vector containing the dataset, where each row is an instance and the first column is the class label
* labels: a vector of integers representing the class labels for each instance
* numFeatures: the total number of features in the dataset 
*
* returns: a vector of integers representing the indices of the selected features
*/
vector<int> backwardElimination(const vector<vector<double>>& data, const vector<int>& labels, int numFeatures) {
    vector<int> bestFeatureSet;
    vector<int> currentFeatureSet;
    for (unsigned int i = 0; i < numFeatures; i++) {
        currentFeatureSet.push_back(i);
    }

    double bestOverallAccuracy = LOOCV(selectFeatures(data, currentFeatureSet), labels); // Calculate accuracy with all features
    cout << "Beginning search." << endl << endl;

    // Start with all features and iteratively remove the worst one
    while (currentFeatureSet.size() > 1) {
        int worstFeature = -1;
        double bestAccuracy = 0.0;

        // Iterate over all features to find the worst one to remove
        for (unsigned int feature = 0; feature < currentFeatureSet.size(); feature++) {
            vector<int> tempFeatureSet = currentFeatureSet;
            tempFeatureSet.erase(tempFeatureSet.begin() + feature); // Remove the feature at index 'feature'
            double accuracy = LOOCV(selectFeatures(data, tempFeatureSet), labels); // Calculate accuracy with the current feature set
            cout << "Using feature(s) {";
            for (unsigned int j = 0; j < tempFeatureSet.size(); j++) {
                cout << tempFeatureSet[j] + 1;
                if (j < tempFeatureSet.size() - 1) {
                    cout << ", ";
                }
            }
            cout << "}, accuracy = " << accuracy << "%" << endl;

            if (accuracy > bestAccuracy) {
                bestAccuracy = accuracy;
                worstFeature = feature; // Store the index of the worst feature
            }
        }

        // Remove the worst feature found in this iteration
        if (worstFeature != -1) {
            currentFeatureSet.erase(currentFeatureSet.begin() + worstFeature); // Remove the worst feature
            cout << "\nFeature set {";
            for (unsigned int j = 0; j < currentFeatureSet.size(); j++) {
                cout << currentFeatureSet[j] + 1;
                if (j < currentFeatureSet.size() - 1) {
                    cout << ", ";
                }
            }
            cout << "} was best, accuracy is " << bestAccuracy << "%" << endl << endl;

            if (bestAccuracy > bestOverallAccuracy) {
                bestOverallAccuracy = bestAccuracy; // Update the best overall accuracy
                bestFeatureSet = currentFeatureSet; // Update the best feature set
            } else {
                cout << "(Warning, Accuracy has decreased! Continuing search in case of local maxima)" << endl;
            }
        }
    }

    cout << "\nFinished search!! The best feature subset is {";
    for (unsigned int j = 0; j < bestFeatureSet.size(); j++) {
        cout << bestFeatureSet[j] + 1;
        if (j < bestFeatureSet.size() - 1) {
            cout << ", ";
        }
    }
    cout << "} with an accuracy of " << bestOverallAccuracy << "%" << endl;
    return bestFeatureSet; // Return the best feature set found
}
