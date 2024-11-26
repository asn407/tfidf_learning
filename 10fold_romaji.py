import os
import MeCab
import ipadic
import romkan
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import dill

def makeBigramData(file_path):
    # MeCabの設定
    wakati = MeCab.Tagger("ipadic.MECAB_ARGS")

    # ファイルを読み込み、1行ごとに形態素解析し、「読み」の部分のみをリストに格納する
    yomi_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            wakati_result = wakati.parse(line).strip().split('\n')
            yomi_line = []
            for word in wakati_result:
                if word != 'EOS':
                    parts = word.split('\t')
                    if len(parts) > 1:
                        features = parts[1].split(',')
                        if len(features) > 7:
                            yomi_line.append(features[8])
            yomi_list.append(yomi_line)

    # 各行において、各単語の始めの文字と終わりの文字を取得し、新しいリストに格納する
    char_list = []
    for line in yomi_list:
        char_line = []
        for i, word in enumerate(line):
            if len(word) > 0:
                if i == 0:
                    char_line.append(word[-1])  # 最初の単語は終わりの文字のみ
                elif i == len(line) - 1:
                    char_line.append(word[0])  # 最後の単語は始めの文字のみ
                else:
                    if word[0] in 'ャュョ' and i > 0:
                        char_line.append(line[i-1][-1] + word[0])  # 小さいヤユヨが含まれる場合は直前の文字もセット
                    elif word[0] == 'ー' and i > 0:
                        char_line.append(line[i-1][-1])  # 伸ばし棒の場合は直前の文字をセット
                    else:
                        char_line.append(word[0])
                    if word[-1] in 'ャュョ' and len(word) > 1:
                        char_line.append(word[-2] + word[-1])  # 小さいヤユヨが含まれる場合は直前の文字もセット
                    elif word[-1] == 'ー' and len(word) > 1:
                        char_line.append(word[-2])  # 伸ばし棒の場合は直前の文字をセット
                    else:
                        char_line.append(word[-1])
        char_list.append(char_line)

    # カタカナをローマ字に変換し、3文字に連なる場合は始めの2文字を取得してリストに格納する
    romaji_list = []
    for line in char_list:
        romaji_line = []
        for word in line:
            romaji_word = romkan.to_roma(word)
            if len(romaji_word) > 2:
                romaji_line.append(romaji_word[:2])
            else:
                if romaji_word in ['a', 'i', 'u', 'e', 'o']:
                    romaji_line.append(romaji_word * 2)
                elif romaji_word == 'n':
                    romaji_line.append('nn')
                else:
                    romaji_line.append(romaji_word)
        romaji_list.append(' '.join(romaji_line))  # スペースで結合して1つの文字列にする

    return romaji_list

def makeRomajiData(file_path):
    # MeCabの設定
    wakati = MeCab.Tagger("ipadic.MECAB_ARGS")

    # ファイルを読み込み、1行ごとに形態素解析し、「読み」の部分のみをリストに格納する
    yomi_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            wakati_result = wakati.parse(line).strip().split('\n')
            yomi_line = []
            for word in wakati_result:
                if word != 'EOS':
                    parts = word.split('\t')
                    if len(parts) > 1:
                        features = parts[1].split(',')
                        if len(features) > 7:
                            yomi_line.append(features[8])
            yomi_list.append(yomi_line)

    # カタカナをローマ字に変換し、リストに格納する
    romaji_list = []
    for line in yomi_list:
        romaji_line = []
        for word in line:
            romaji_word = romkan.to_roma(word)
            # 伸ばし棒（「-」ハイフン）とシングルクオーテーション（「'」）を取り除く
            romaji_word = romaji_word.replace('-', '').replace("'", '')
            romaji_line.append(romaji_word)
        romaji_list.append(romaji_line)

    sentences_list = []
    for line in romaji_list:
        sentences_list.append(''.join(line))
    
    return sentences_list

# モデルのリスト
models = {
    'SVM': svm.SVC(probability=True),
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier()
}

# データとラベルの準備
romajiData = makeRomajiData('./data/0.txt') + makeRomajiData('./data/1.txt')
bigramData = makeBigramData('./data/0.txt') + makeBigramData('./data/1.txt')
labels = [0 for _ in range(146)] + [1 for _ in range(146)]

# TfidfVectorizerを使用
romaji_vectorizer = TfidfVectorizer(analyzer='char')
X = romaji_vectorizer.fit_transform(romajiData)
# bigram_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 2))
# bigram_features = bigram_vectorizer.fit_transform(bigramData)

# 特徴量の結合
# X = np.hstack((romaji_features.toarray(), bigram_features.toarray()))

# 各モデルに対して評価を行う
for model_name, model in models.items():
    print(f"Evaluating model with {model_name}...")

    # 評価結果を保存するリスト
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    conf_matrices = []
    model_predictions = []

    # 10分割交差検証を使用
    kf = KFold(n_splits=10, shuffle=True, random_state=None)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = np.array(labels)[train_index], np.array(labels)[test_index]

        # モデルの訓練
        model.fit(X_train, y_train)

        # テストデータで予測
        y_pred = model.predict(X_test)
        model_predictions.append(model.predict_proba(X_test))
        
        # モデルの性能評価
        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, zero_division=1))
        recalls.append(recall_score(y_test, y_pred, zero_division=1))
        f1_scores.append(f1_score(y_test, y_pred, zero_division=1))
        conf_matrices.append(confusion_matrix(y_test, y_pred))
    
    # 平均値を計算
    avg_accuracy = np.mean(accuracies)
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1 = np.mean(f1_scores)
    avg_conf_matrix = np.mean(conf_matrices, axis=0)

    print(f"Average Accuracy: {avg_accuracy}")
    print(f"Average Precision: {avg_precision}")
    print(f"Average Recall: {avg_recall}")
    print(f"Average F1 Score: {avg_f1}")
    print("Average Confusion Matrix:")
    print(avg_conf_matrix)
    print("\n")

    # # アンサンブルモデルの作成
    # ensemble_predictions = np.mean(model_predictions, axis=0)
    # ensemble_model = lambda X: np.argmax(np.mean([model.predict_proba(X) for model in models.values()], axis=0), axis=1)
    
    # # アンサンブルモデルを保存
    # model_filename = f"{model_name.replace(' ', '_').lower()}_ensemble_model.dill"
    # with open(model_filename, 'wb') as f:
    #     dill.dump(ensemble_model, f)
    # print(f"Ensemble model saved as {model_filename}")
