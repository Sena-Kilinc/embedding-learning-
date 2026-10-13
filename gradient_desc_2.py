import numpy as np
import matplotlib.pyplot as plt


# Amaç fonksiyonu: f(x) = x^2
x = np.arange(-5, 5.01, 0.01)
y = x ** 2

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
precision = 0.00001  # Durdurma hassasiyeti
max_iter = 200     # Maksimum iterasyon sayısı

# Her iterasyondaki x ve f(x) değerlerini sakla
Xs = np.zeros(max_iter)
Ys = np.zeros(max_iter)

# İterasyon sayacı
i = 0

# x değerindeki değişim hassasiyetten büyük olduğu
# ve maksimum iterasyon sayısına ulaşılmadığı sürece devam et
while abs(x_new - x_old) > precision and i < max_iter:

    # Mevcut noktayı kaydet
    Xs[i] = x_new
    Ys[i] = x_new ** 2

    # Mevcut noktayı grafikte kırmızı yıldızla göster
    plt.plot(Xs[i], Ys[i], "r*")

    # Noktanın yanına iterasyon numarasını yaz
    plt.text(Xs[i], Ys[i], str(i + 1))

    # Mevcut x değerini sakla
    x_old = x_new

    # Türevi hesapla: f'(x) = 2x
    df = 2 * x_old

    # Güncelleme:
    # Türevin büyüklüğü de adım hesabında kullanılır
    x_new = x_old - eps * df

    # İterasyon sayısını artır
    i += 1

# İlk grafiği göster
plt.show()

# İterasyonlar boyunca x değerlerinin değişimini çiz
plt.figure()
plt.plot(range(1, i + 1), Xs[:i], "b-o")

plt.xlabel("Iteration")
plt.ylabel("x values")
plt.title("Gradient Descent: Change in x Values")
plt.grid(True)

# İkinci grafiği göster
plt.show()

# Sonuçları ekrana yazdır
print("İterasyon sayısı:", i)
print("Son x değeri:", x_new)
print("Son f(x) değeri:", x_new ** 2)