import numpy as np
import matplotlib.pyplot as plt

# Amaç fonksiyonu: f(x) = x^2
# Türevi: f'(x) = 2x

def gradient_descent(eps, x_start=-4.9, precision=0.00001,
                     max_iter=200):

    # Başlangıç noktası
    x = x_start

    # Her iterasyondaki x değerlerini sakla
    Xs = [x]

    # Gradient Descent algoritması
    for i in range(max_iter):

        # Mevcut noktadaki türevi hesapla
        df = 2 * x

        # Yeni x değerini hesapla
        x_new = x - eps * df

        # Ardışık iki x değeri arasındaki fark
        # hassasiyetten küçükse algoritmayı durdur
        if abs(x_new - x) < precision:
            x = x_new
            Xs.append(x)
            break

        # Bir sonraki iterasyona geç
        x = x_new
        Xs.append(x)

    return np.array(Xs)


# Karşılaştırılacak adım boyları
step_sizes = [0.01, 0.05, 0.7, 0.99, 1.0]

# Amaç fonksiyonunu çiz
x = np.linspace(-5, 5, 500)
y = x ** 2

plt.figure(figsize=(9, 5))
plt.plot(x, y, label="f(x) = x^2")

# Farklı adım boylarının izlediği yolu göster
for eps in step_sizes:

    # Gradient Descent algoritmasını çalıştır
    Xs = gradient_descent(eps)

    # Her iterasyondaki amaç fonksiyonu değerleri
    Ys = Xs ** 2

    # İterasyon noktalarını grafiğe ekle
    plt.plot(Xs, Ys, "o-", markersize=3,
             label=f"eps = {eps}")

    # İterasyon sayısını ve son değeri yazdır
    print(f"Adım boyu = {eps}")
    print(f"İterasyon sayısı = {len(Xs) - 1}")
    print(f"Son x değeri = {Xs[-1]:.8f}")
    print(f"Son f(x) değeri = {Xs[-1] ** 2:.8f}")
    print("-" * 35)

plt.xlabel("x")
plt.ylabel("f(x) = x^2")
plt.title("Effect of Step Size on Gradient Descent")
plt.legend()
plt.grid(True)
plt.show()


# İterasyonlara göre x değerlerinin değişimini karşılaştır
plt.figure(figsize=(9, 5))

for eps in step_sizes:
    Xs = gradient_descent(eps)

    plt.plot(
        range(len(Xs)),
        Xs,
        "o-",
        markersize=3,
        label=f"eps = {eps}"
    )

plt.axhline(0, linestyle="--")
plt.xlabel("Iteration")
plt.ylabel("x value")
plt.title("Convergence Behavior for Different Step Sizes")
plt.legend()
plt.grid(True)
plt.show()