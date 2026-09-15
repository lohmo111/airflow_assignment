import pandas as pd
from ydata_profiling import ProfileReport
from sweetviz import compare

train = pd.read_csv("data/raw/data.csv")
test = pd.read_csv("data/raw/test.csv")

p_train = ProfileReport(train, title="Train")
p_test  = ProfileReport(test, title="Test")
p_train.compare(p_test).to_file("reports/comparison.html")

report = compare(
    [train, "Train"],
    [test, "Test"],
    target_feat="target"   # если таргет есть в обоих
)
report.show_html("reports/train_vs_test.html")

