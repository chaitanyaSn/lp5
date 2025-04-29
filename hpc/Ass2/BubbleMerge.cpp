#include <iostream>
#include <vector>
#include <algorithm>
#include <omp.h>

using namespace std;

// Function to print array
void printArray(const vector<int>& arr) {
    for (int val : arr)
        cout << val << " ";
    cout << endl;
}

// Sequential Bubble Sort
void sequentialBubbleSort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n-1; ++i) {
        for (int j = 0; j < n-i-1; ++j) {
            if (arr[j] > arr[j+1])
                swap(arr[j], arr[j+1]);
        }
    }
}

// Parallel Bubble Sort (Odd-Even Transposition Sort)
void parallelBubbleSort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n; ++i) {
        #pragma omp parallel for
        for (int j = (i % 2); j < n-1; j += 2) {
            if (arr[j] > arr[j+1])
                swap(arr[j], arr[j+1]);
        }
    }
}

// Sequential Merge Sort
void sequentialMerge(vector<int>& arr, int l, int m, int r) {
    vector<int> left(arr.begin() + l, arr.begin() + m + 1);
    vector<int> right(arr.begin() + m + 1, arr.begin() + r + 1);

    int i = 0, j = 0, k = l;
    while (i < left.size() && j < right.size()) {
        if (left[i] <= right[j])
            arr[k++] = left[i++];
        else
            arr[k++] = right[j++];
    }

    while (i < left.size()) arr[k++] = left[i++];
    while (j < right.size()) arr[k++] = right[j++];
}

void sequentialMergeSort(vector<int>& arr, int l, int r) {
    if (l < r) {
        int m = (l + r) / 2;
        sequentialMergeSort(arr, l, m);
        sequentialMergeSort(arr, m+1, r);
        sequentialMerge(arr, l, m, r);
    }
}

// Parallel Merge Sort
void parallelMergeSort(vector<int>& arr, int l, int r) {
    if (l < r) {
        int m = (l + r) / 2;

        #pragma omp parallel sections
        {
            #pragma omp section
            parallelMergeSort(arr, l, m);

            #pragma omp section
            parallelMergeSort(arr, m+1, r);
        }

        sequentialMerge(arr, l, m, r);
    }
}

int main() {
    const int N = 10000; // Change N for larger arrays
    vector<int> arr(N);

    // Fill array with random numbers
    for (int i = 0; i < N; ++i)
        arr[i] = rand() % 10000;

    vector<int> arr1 = arr;
    vector<int> arr2 = arr;
    vector<int> arr3 = arr;
    vector<int> arr4 = arr;

    // Bubble Sort
    double start, end;

    cout << "---- Bubble Sort ----" << endl;
    start = omp_get_wtime();
    sequentialBubbleSort(arr1);
    end = omp_get_wtime();
    cout << "Sequential Bubble Sort Time: " << end - start << " seconds" << endl;

    start = omp_get_wtime();
    parallelBubbleSort(arr2);
    end = omp_get_wtime();
    cout << "Parallel Bubble Sort Time: " << end - start << " seconds" << endl;

    cout << "\n---- Merge Sort ----" << endl;
    // Merge Sort
    start = omp_get_wtime();
    sequentialMergeSort(arr3, 0, arr3.size()-1);
    end = omp_get_wtime();
    cout << "Sequential Merge Sort Time: " << end - start << " seconds" << endl;

    start = omp_get_wtime();
    parallelMergeSort(arr4, 0, arr4.size()-1);
    end = omp_get_wtime();
    cout << "Parallel Merge Sort Time: " << end - start << " seconds" << endl;

    return 0;
}
