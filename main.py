# main.py

# 📦 Gerekli Kütüphaneler
import pandas as pd
import re
import string
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# 1️⃣ Veri Setini Yükle
print("Veri seti yükleniyor...")
dosya_yolu = "training.1600000.processed.noemoticon.csv"  # Github'a göre dosya aynı klasörde olmalı
df = pd.read_csv(dosya_yolu, encoding="latin-1", header=None)
df.columns = ["target", "id", "date", "flag", "user", "text"]
df = df[["target", "text"]]
df["target"] = df["target"].replace({4: 1})  # 4 → 1


# 2️⃣ Tweet Temizleme Fonksiyonu
def temizle(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()


df["text"] = df["text"].apply(temizle)
df = df[df["text"].str.strip() != ""]

# 3️⃣ TF-IDF ile Sayısallaştırma
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df["text"])
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4️⃣ Modelleri Eğit ve Karşılaştır
modeller = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42),
    "SVM": LinearSVC()
}

print("\nModeller eğitiliyor ve değerlendiriliyor...")

dogruluklar = []
for ad, model in modeller.items():
    print(f"\n🔹 {ad}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Confusion matrix görselleştir
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title(f"{ad} - Confusion Matrix")
    plt.show()

    dogruluklar.append(accuracy_score(y_test, y_pred))

# 5️⃣ Özet Tablosu
print("\nModellerin başarı özetleri:")
for model_adi, acc in zip(modeller.keys(), dogruluklar):
    print(f"{model_adi}: {acc:.4f}")