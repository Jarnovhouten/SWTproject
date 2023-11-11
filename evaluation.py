import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

if __name__ == '__main__':
    df = pd.read_excel('results.xlsx')
    # Replace empty number cells with the default value 3
    df['true_number'].fillna(3.0, inplace=True)
    # Fill empty cells
    df['pred_number'].fillna(0.0, inplace=True)
    df.fillna('', inplace=True)

    precision, recall, fscore, _ = precision_recall_fscore_support(
        df['true_intent'],
        df['pred_intent'],
        average='weighted',
        zero_division=1)
    print('Scores for intent:')
    print(f"Precision: {round(precision, 2)}")
    print(f"Recall: {round(recall, 2)}")
    print(f"F1 Score: {round(fscore, 2)}")

    print('\n---------------------------\n')

    precision, recall, fscore, _ = precision_recall_fscore_support(
        df['true_number'],
        df['pred_number'],
        average='weighted',
        zero_division=1)
    print('Scores for number:')
    print(f"Precision: {round(precision, 2)}")
    print(f"Recall: {round(recall, 2)}")
    print(f"F1 Score: {round(fscore, 2)}")

    print('\n---------------------------\n')

    precision, recall, fscore, _ = precision_recall_fscore_support(
        df['true_entity'],
        df['pred_entity'],
        average='weighted',
        zero_division=1)
    print('Scores for entity:')
    print(f"Precision: {round(precision, 2)}")
    print(f"Recall: {round(recall, 2)}")
    print(f"F1 Score: {round(fscore, 2)}")

    print('\n---------------------------\n')

    precision, recall, fscore, _ = precision_recall_fscore_support(
        df['corrected_genre'],
        df['pred_genre'],
        average='weighted',
        zero_division=1)
    print('Scores for genre:')
    print(f"Precision: {round(precision, 2)}")
    print(f"Recall: {round(recall, 2)}")
    print(f"F1 Score: {round(fscore, 2)}")

    print('\n---------------------------\n')
    
    precision, recall, fscore, _ = precision_recall_fscore_support(
        df['true_location'],
        df['pred_location'],
        average='weighted',
        zero_division=1)
    print('Scores for location:')
    print(f"Precision: {round(precision, 2)}")
    print(f"Recall: {round(recall, 2)}")
    print(f"F1 Score: {round(fscore, 2)}")
    