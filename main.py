# 📦 GEREKLİ KÜTÜPHANELERİ YÜKLE
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1️⃣ VERİ SETİNİ YÜKLE
# Not: Dosya yolunu kendi bilgisayarındaki konuma göre değiştir.
dosya_yolu = r"C:\Users\hakan\Desktop\training.1600000.processed.noemoticon.csv"

# CSV'yi oku, encoding ve sütunlar belirleniyor
df = pd.read_csv(dosya_yolu, encoding="latin-1", header=None)
df.columns = ["target", "id", "date", "flag", "user", "text"]

# Yalnızca target (etiket) ve text (tweet) kolonlarını alıyoruz
df = df[["target", "text"]]

# 4 = pozitif → 1 yap, 0 zaten negatif
df["target"] = df["target"].replace({4: 1})

print("Veri seti başarıyla yüklendi. İlk 5 satır:")
print(df.head())

# 2️⃣ METİN TEMİZLEME FONKSİYONU
def temizle(text):
    text = text.lower()  # Küçük harf
    text = re.sub(r"http\S+", "", text)  # link sil
    text = re.sub(r"@\w+", "", text)     # @mention sil
    text = re.sub(r"#", "", text)        # hashtag sembolü sil
    text = re.sub(r"\d+", "", text)      # sayılar sil
    text = text.translate(str.maketrans("", "", string.punctuation))  # noktalama sil
    return text.strip()

# Temizleme fonksiyonunu tüm tweet'lere uygula
df["text"] = df["text"].apply(temizle)

print("\nTemizlenmiş ilk 5 tweet:")
print(df.head())

# 3️⃣ METNİ TF-IDF VEKÖRÜNE DÖNÜŞTÜR
tfidf = TfidfVectorizer(max_features=5000)  # En çok geçen 5000 kelime
X = tfidf.fit_transform(df["text"])         # X: Özellikler (vektör)
y = df["target"]                            # y: Etiket

# Eğitim ve test verisini ayır (80% eğitim, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4️⃣ MODELLERİ TANIMLA VE EĞİT
modeller = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42),
    "SVM": LinearSVC()
}

# Her model için eğit, tahmin yap ve değerlendir
for ad, model in modeller.items():
    print(f"\n🔹 {ad} modeli eğitiliyor...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Doğruluk Oranı:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

from sklearn.metrics import accuracy_score

dogruluklar = []
modeller = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "SVM": LinearSVC()
}

for ad, model in modeller.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    dogruluklar.append(acc)

ozet = {
    "Model": list(modeller.keys()),
    "Doğruluk": dogruluklar
}

df_ozet = pd.DataFrame(ozet)
df_ozet

