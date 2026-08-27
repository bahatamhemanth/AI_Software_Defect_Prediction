# Model Evaluation

Model selection used five-fold stratified cross-validation on the training set, ordered by defective-class recall, then mean average precision and F1. This reflects the cost of missing a defective module. The held-out test set was used once for final reporting.

Selected model: **Logistic Regression**

              model  cv_average_precision  cv_recall  cv_f1  accuracy  precision  recall     f1  roc_auc  pr_auc
Logistic Regression                0.3375     0.6924 0.3391    0.8010     0.2391  0.7857 0.3667   0.9088  0.4865
                SVM                0.2328     0.5879 0.2817    0.9267     0.0000  0.0000 0.0000   0.8085  0.2402
      Decision Tree                0.1830     0.4273 0.2604    0.8220     0.1667  0.3571 0.2273   0.6150  0.1424
      Random Forest                0.2704     0.2333 0.2669    0.9162     0.4375  0.5000 0.4667   0.8592  0.3933
  Gradient Boosting                0.3548     0.2121 0.2623    0.9058     0.3333  0.2857 0.3077   0.8156  0.3422

Class weights were used only on estimators that support them. No resampling was applied. Duplicate rows were removed before the stratified split, and `T` was excluded because EDA found it perfectly correlated with `E`.