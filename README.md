# MSCS 634 – Lab 2: Classification Using KNN and RNN Algorithms

**Name:** Anuj Shrestha

## Purpose
This lab compares two classification methods, K-Nearest Neighbors (KNN) and Radius Neighbors (RNN), on the Wine dataset from scikit-learn (178 samples, 13 chemical features, 3 wine classes). The goal is to see how the parameter choices (k for KNN, radius for RNN) affect accuracy, and to discuss when each method is the better fit.

## Repository Contents
- `Lab2_Assignment_Anuj_Shrestha.ipynb` – the full notebook
- `screenshots/` – screenshots of each notebook step
- `Lab2_Assignment_Anuj_Shrestha.docx` – screenshots with short descriptions
- `README.md` – this summary

## Method
The data was split 80% training (142 samples) and 20% testing (36 samples) with `random_state=42`. KNN was tested with k = 1, 5, 11, 15, and 21. RNN was tested with radius = 350, 400, 450, 500, 550, and 600. The features were left unscaled, as in the lab instructions.

## Results

| KNN (k) | Accuracy | RNN (radius) | Accuracy |
|---|---|---|---|
| 1 | 77.8% | 350 | 75.0% |
| 5 | 72.2% | 400 | 72.2% |
| 11 | 75.0% | 450 | 72.2% |
| 15 | 75.0% | 500 | 72.2% |
| 21 | 77.8% | 550 | 72.2% |
| | | 600 | 72.2% |

## Key Insights
- KNN accuracy stayed between 72% and 78% for every k. The best results (77.8%) came from k = 1 and k = 21, and the lowest from k = 5. There is no clear upward or downward trend.
- RNN performed best at the smallest radius (350, 75.0%) and then stayed flat at 72.2% for every larger radius. At these radii each test sample already has many neighbors (on average about 78 of the 142 training samples at radius 350), so growing the radius adds little new information.
- Overall accuracy is modest for both methods because the features are on very different scales. Proline ranges from about 278 to 1680, while most other features are below 15, so proline dominates the distance calculation. This is also why radius values in the hundreds are needed.
- As an extra check, standardizing the features first raised KNN accuracy to about 94% to 97%, which confirms that scaling matters much more than the choice of k here.

## KNN vs RNN: When to Use Each
KNN always uses a fixed number of neighbors, so every prediction is based on the same amount of evidence, and it is simple to tune. RNN uses every neighbor within a fixed radius, so it adapts to how dense the data is around a point: it uses many neighbors in crowded regions and few in sparse ones, and it can flag points with no neighbors at all as outliers. KNN is the safer default when the data density varies or the feature scales are unclear. RNN is a better fit when a meaningful distance threshold is known and the features are on a comparable scale.

## Challenges and Decisions
- Choosing the radius was the main difficulty. Without scaling, the distances are dominated by proline, so radii like 350 to 600 are needed to capture neighbors at all. A radius that is too small leaves a test point with no neighbors, so `outlier_label="most_frequent"` was set as a safeguard (every test point did have neighbors at the radii used here).
- The results come from a single train/test split of only 36 test samples, so one sample changes accuracy by about 2.8 percentage points. Small differences between k values or radii should therefore not be over-interpreted.
