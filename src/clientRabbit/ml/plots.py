import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(confusion_matrix, title='', figsize=(12,8)):
    plt.figure(figsize=figsize)
    sns.heatmap(confusion_matrix, annot=True, cmap='Blues', cbar=True)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title)