#include <iostream>
#include <omp.h>
#include <vector>
#include <climits> // for INT_MIN, INT_MAX

using namespace std;

int main()
{
    int n;
    cout << "Enter number of elements: ";
    cin >> n;

    vector<int> arr(n);
    cout << "Enter elements:" << endl;
    for (int i = 0; i < n; i++)
    {
        cin >> arr[i];
    }

    int min_val = INT_MAX;
    int max_val = INT_MIN;
    long long sum = 0; // Use long long to prevent overflow

// Parallel Reduction
#pragma omp parallel for reduction(min : min_val) reduction(max : max_val) reduction(+ : sum)
    for (int i = 0; i < n; i++)
    {
        if (arr[i] < min_val)
            min_val = arr[i];
        if (arr[i] > max_val)
            max_val = arr[i];
        sum += arr[i];
    }

    double average = (double)sum / n;

    cout << "\nResults using Parallel Reduction:" << endl;
    cout << "Minimum Value: " << min_val << endl;
    cout << "Maximum Value: " << max_val << endl;
    cout << "Sum: " << sum << endl;
    cout << "Average: " << average << endl;

    return 0;
}
