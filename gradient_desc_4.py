import numpy as np
import matplotlib.pyplot as plt

# Amaç fonksiyonu:
# f(x) = x^4 - 3x^3 + 2
def f(x):
    return x**4 - 3*x**3 + 2


# Birinci türev:
# f'(x) = 4x^3 - 9x^2
def df(x):
    return 4*x**3 - 9*x**2


def gradient_descent(x_start, eps, precision=0.00001,
                     max_iter=200):

    # Başlangıç noktası
    x = x_start

    # İterasyonlardaki x değerlerini sakla
    Xs = [x]

    for i in range(max_iter):

        # Sayısal taşmayı önlemek için kontrol
        if not np.isfinite(x) or abs(x) > 1e4:
            break

        # Mevcut noktadaki türevi hesapla
        gradient = df(x)

        # Gradient Descent güncellemesi:
        # Yeni x = Eski x - Adım boyu * Türev
        x_new = x - eps * gradient

        # Yeni değerin geçerli olup olmadığını kontrol et
        if not np.isfinite(x_new) or abs(x_new) > 1e4:
            break

        # Yeni noktayı kaydet
        Xs.append(x_new)

        # Ardışık iki x değeri arasındaki fark
        # hassasiyet sınırından küçükse dur
        if abs(x_new - x) < precision:
            x = x_new
            break

        # Bir sonraki iterasyona geç
        x = x_new

    return np.array(Xs)


# --------------------------------------------------
# DENEY 1: Farklı başlangıç noktalarının etkisi
# Adım boyu sabit: eps = 0.01
# --------------------------------------------------

eps = 0.01

# İki farklı başlangıç noktası için algoritmayı çalıştır
Xs1 = gradient_descent(x_start=3.9, eps=eps)
Xs2 = gradient_descent(x_start=-1.9, eps=eps)

# Amaç fonksiyonunu çiz
x = np.linspace(-2.5, 4.2, 600)

plt.figure(figsize=(10, 6))
plt.plot(x, f(x), label="f(x) = x^4 - 3x^3 + 2")

# İlk başlangıç noktasının izlediği yol
plt.plot(Xs1, f(Xs1), "ro-", markersize=3,
         label="Başlangıç = 3.9")

# İkinci başlangıç noktasının izlediği yol
plt.plot(Xs2, f(Xs2), "go-", markersize=3,
         label="Başlangıç = -1.9")

# Kritik noktaları göster
plt.scatter([0, 2.25], [f(0), f(2.25)],
            color="black", zorder=5)

plt.annotate("x = 0 (minimum değil)",
             (0, f(0)), xytext=(-1.7, 15),
             arrowprops=dict(arrowstyle="->"))

plt.annotate("x = 2.25 (yerel minimum)",
             (2.25, f(2.25)), xytext=(2.5, 10),
             arrowprops=dict(arrowstyle="->"))

plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Effect of Initial Point on Gradient Descent")
plt.legend()
plt.grid(True)
plt.show()


# --------------------------------------------------
# DENEY 2: Farklı adım boylarının etkisi
# Başlangıç noktası sabit: x = 3.9
# --------------------------------------------------

step_sizes = [0.01, 0.05, 0.1]

plt.figure(figsize=(10, 6))

for eps in step_sizes:

    # Seçilen adım boyuyla Gradient Descent çalıştır
    Xs = gradient_descent(x_start=3.9, eps=eps)

    # İterasyon boyunca x değerlerini çiz
    plt.plot(range(len(Xs)), Xs, "o-",
             markersize=3, label=f"eps = {eps}")

    # Sonuçları yazdır
    print(f"Adım boyu = {eps}")
    print(f"İterasyon sayısı = {len(Xs) - 1}")
    print(f"Son x değeri = {Xs[-1]:.8f}")
    print(f"Son f(x) değeri = {f(Xs[-1]):.8f}")
    print("-" * 40)

plt.axhline(0, linestyle="--")
plt.axhline(2.25, linestyle="--")
plt.xlabel("Iteration")
plt.ylabel("x value")
plt.title("Effect of Step Size on Gradient Descent")
plt.legend()
plt.grid(True)
plt.show()