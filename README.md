# 🌚 EmoRecognition

EmoRecognition is a facial emotion recognition project focused on distinguishing three distinct emotional states: mild disgust expression, slight smile, and neutral face state. The classification relies on differences between key facial points' positions across consecutive frames taken two seconds apart. Two powerful supervised learning techniques—XGBoost and logistic regression—are utilized for accurate classification.

## How It Works

The pipeline consists of several stages:

1. **Facial Landmark Extraction:** Key facial points are extracted from each frame using dlib’s shape predictor.

2. **Feature Engineering**: Positional differences between corresponding landmarks across two frames separated by two seconds are calculated.

3. **Classification**: These features are fed into either XGBoost or logistic regression classifier for prediction.