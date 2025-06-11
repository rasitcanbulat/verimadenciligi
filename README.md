# 🧠 Tweet Duygu Analizi - Sentiment Analysis Project

Bu proje, sosyal medya üzerindeki kullanıcı paylaşımlarını (tweet'leri) analiz ederek duygusal içeriklerini otomatik olarak sınıflandırmayı amaçlamaktadır. Makine öğrenmesi algoritmaları kullanılarak pozitif veya negatif olarak etiketleme yapılmıştır.

## 🎯 Proje Amacı

- Tweet metinlerini işleyerek olumlu/olumsuz duygu analizinin yapılması.
- Farklı makine öğrenmesi algoritmalarının (Logistic Regression, Naive Bayes, Random Forest, SVM) performanslarını karşılaştırmak.
- TF-IDF yöntemiyle metinlerin sayısallaştırılması ve sınıflandırma işlemleri gerçekleştirmek.

## 📁 Veri Seti Bilgisi

- **Kaynak:** [Kaggle - Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)
- **Satır Sayısı:** 1.600.000 tweet
- **Sütun Sayısı:** 6
- **Kullanılan Sütunlar:** `text` (tweet), `target` (etiket - 0: negatif, 4 → 1: pozitif)

## ⚙️ Kullanılan Adımlar

### 1. Veri Yükleme ve Hazırlama
```python
df = pd.read_csv("training.1600000.processed.noemoticon.csv", encoding="latin-1", header=None)
df.columns = ["target", "id", "date", "flag", "user", "text"]
df = df[["target", "text"]]
df["target"] = df["target"].replace({4: 1})
```

### 2. Metin Temizleme
Tweetler temizlenerek gereksiz karakterler, linkler, mention ve sayılar silinmiştir.

### 3. TF-IDF ile Özellik Çıkarımı
Maksimum 5000 kelime ile TF-IDF vektörleştirmesi uygulanmıştır.

### 4. Modellerin Eğitimi
Dört model kullanılarak eğitim/test işlemleri yapılmıştır:

- Logistic Regression
- Multinomial Naive Bayes
- Random Forest Classifier
- Linear SVM

### 5. Değerlendirme Metrikleri

- Accuracy (Doğruluk Oranı)
- Confusion Matrix (Karışıklık Matrisi)
- Precision, Recall, F1-Score (Sınıflandırma Raporu)

### 6. Karışıklık Matrisi Görselleştirme
Matplotlib ile her model için confusion matrix görselleri oluşturulmuştur.

## 📊 Performans Özeti

| Model               | Accuracy | Precision (1) | Recall (1) | F1-Score (1) | False Negative (1) | Yorum                                                                 |
|--------------------|----------|---------------|------------|--------------|---------------------|-----------------------------------------------------------------------|
| Logistic Regression| 0.7901   | 🟩 0.78        | 🟩 0.80     | ✅ 0.79       | ~31.479             | Dengeli ve güçlü performans. Hem precision hem recall yüksek.         |
| Naive Bayes        | 0.7691   | 🟨 0.78        | 🟨 0.76     | ✅ 0.77       | ~38.733             | Fena değil, doğruluk makul ama FN sayısı biraz fazla.                |
| Random Forest      | 0.7142   | 🟨 0.69        | 🟩 0.78     | 🟠 0.73       | ~35.460             | Recall yüksek ama sınıf 0 için başarısız.                            |
| SVM                | 0.7898   | 🟩 0.78        | 🟩 0.81     | ✅ 0.79       | ~31.126             | Logistic Regression kadar başarılı, sınıf 1 üzerinde güçlü.          |

## 🛠 Kurulum

1. Gerekli kütüphaneleri yükleyin:
```bash
pip install pandas scikit-learn matplotlib
```

2. `main.py` dosyasını çalıştırın.

## 📌 Notlar

- Bu proje Google Colab veya Jupyter Notebook ortamlarında da rahatlıkla çalıştırılabilir.
- Dosya yolu `main.py` içinde kendi sisteminize göre ayarlanmalıdır.
