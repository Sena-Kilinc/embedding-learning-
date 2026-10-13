import numpy as np
import matplotlib.pyplot as plt

# Amaç fonksiyonunu çiz: f(x) = x^2
x = np.arange(-5, 5.01, 0.01)
y = np.zeros(len(x))

# Her x değeri için fonksiyonun değerini hesapla
for i in range(len(x)):
    y[i] = x[i] ** 2

# Amaç fonksiyonunun grafiğini çiz
plt.figure()
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("f(x) = x^2")
plt.title("Objective Function")
plt.grid(True)

# Başlangıç değerleri
x_old = 0
x_new = -4.9       # Başlangıç noktası
eps = 0.05         # Adım boyu (learning rate)
precision = 0.00001  # Durdurma koşulu 1: hassasiyet
max_iter = 200     # Durdurma koşulu 2: maksimum iterasyon sayısı

# Her iterasyondaki x ve f(x) değerlerini saklamak için diziler
Xs = np.zeros(max_iter)
Ys = np.zeros(max_iter)

# İlk iterasyon numarası
i = 0

# x değerindeki değişim hassasiyet sınırından büyük olduğu
# ve maksimum iterasyon sayısına ulaşılmadığı sürece devam et
while abs(x_new - x_old) > precision and i < max_iter:

    # Mevcut x ve amaç fonksiyonu değerlerini kaydet
    Xs[i] = x_new
    Ys[i] = x_new ** 2

    # Mevcut noktayı grafik üzerinde kırmızı yıldızla göster
    plt.plot(Xs[i], Ys[i], "r*")

    # Noktanın yanına iterasyon numarasını yaz
    plt.text(Xs[i], Ys[i], str(i + 1))

    # Bir sonraki iterasyon için mevcut x değerini sakla
    x_old = x_new

    # Amaç fonksiyonunun türevini hesapla
    # f(x) = x^2  =>  f'(x) = 2x
    df = 2 * x_old

    # Türevin işaretine göre x değerini güncelle
    # sign(df) = +1 ise sola, -1 ise sağa doğru hareket edilir
    x_new = x_old - eps * np.sign(df)

    # İterasyon sayısını artır
    i += 1

# İlk grafiği göster
plt.show()

# İterasyonlar boyunca x değerlerinin değişimini çiz
plt.figure()
plt.plot(range(1, i + 1), Xs[:i], "b-o")

# Eksen isimlerini ve grafik başlığını ekle
plt.xlabel("Iteration")
plt.ylabel("x values")
plt.title("Change in x Values During Iterations")
plt.grid(True)

# İkinci grafiği göster
plt.show()