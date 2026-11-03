import os, sys, time, math, itertools
import matplotlib
matplotlib.use("Agg")

_OZET = None
def rapor(x, baslik=""):
    s = (f"\n### {baslik}\n" if baslik else "\n") + (x.to_string() if hasattr(x, "to_string") else str(x)) + "\n"
    print(s); _OZET.write(s); _OZET.flush()
def display(x): rapor(x)


# RF-SGD: Regula-Falsi SGD
_BOLUM = 'RF-SGD: Regula-Falsi SGD'


# 1. Algoritma
_BOLUM = '1. Algoritma'


# 2. Kod
_BOLUM = '2. Kod'

import os, time, math, itertools
import numpy as np
import pandas as pd
import torch
from torch.optim import Optimizer
import matplotlib.pyplot as plt

DT = torch.float64               # 1e-6 altı hataları ölçtüğümüz için float64
HIZLI = os.environ.get("HIZLI") == "0"                    # True: her deney 10 adım (yalnızca smoke test)
def ADIM(n): return 10 if HIZLI else n

OUT = "results_final_v2"
os.makedirs(OUT, exist_ok=True)
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "axes.grid": True, "grid.color": "#e4e3df",
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
RENK = {"RF-SGD": "#2a78d6", "SGD": "#eb6834", "SGD+Momentum": "#1baf7a", "iRprop+": "#eda100", "RF-SGD-M": "#6250d6",
        "RF-SGD-M(g)": "#a99ce8", "Adam": "#c23b7a", "RMSprop": "#5d6b7a"}
print("torch", torch.__version__)
_OZET = open(f"{OUT}/ozet.txt", "w", encoding="utf-8")
_OZET.write(f"HIZLI={HIZLI}\n")


# 2.1 Optimizer'lar
_BOLUM = "2.1 Optimizer'lar"

class RFSGD(Optimizer):
    """Regula-Falsi SGD. Gradyan işareti değişen koordinatta sekant köküne gider, diğerlerinde SGD adımı."""
    def __init__(self, params, lr=1e-2):
        super().__init__(params, dict(lr=lr))
        self.diag = {}

    @torch.no_grad()
    def step(self, loss=None):
        n_tetik, n_toplam = 0, 0
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g, st = p.grad, self.state[p]
                sgd_yeni = p - lr * g
                if "prev_g" in st:
                    flip = (g * st["prev_g"]) < 0
                    payda = g.abs() + st["prev_g"].abs()
                    # flip olmayan koordinatlarda payda 0 olabilir; rho orada kullanılmıyor
                    rho = torch.where(flip, g.abs() / payda.clamp_min(1e-300), torch.zeros_like(g))
                    yeni = torch.where(flip, p - rho * (p - st["prev_p"]), sgd_yeni)
                    n_tetik += int(flip.sum())
                else:
                    yeni = sgd_yeni                     # ilk adım: önceki gradyan yok
                n_toplam += g.numel()
                st["prev_p"], st["prev_g"] = p.detach().clone(), g.detach().clone()
                p.copy_(yeni)
        self.diag = {"trigger_rate": n_tetik / max(n_toplam, 1)}   # tanı: RF kuralının devreye girdiği oran
        return loss


class RFSGDM(Optimizer):
    """Tur 2: işaret testi ve sekant, gradyanın EMA'sı m üzerinden. beta=0 -> RFSGD ile aynı.
    adim="m": normal adım p - lr*m (asıl yöntem); adim="g": normal adım p - lr*g (ablasyon)."""
    def __init__(self, params, lr=1e-2, beta=0.9, adim="m"):
        super().__init__(params, dict(lr=lr, beta=beta))
        self.adim = adim
        self.diag = {}

    @torch.no_grad()
    def step(self, loss=None):
        n_tetik, n_toplam = 0, 0
        for group in self.param_groups:
            lr, beta = group["lr"], group["beta"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g, st = p.grad, self.state[p]
                m = g.detach().clone() if "m" not in st else beta * st["m"] + (1 - beta) * g   # m_1 = g_1
                sgd_yeni = p - lr * (m if self.adim == "m" else g)
                if "m" in st:
                    pm = st["m"]
                    flip = (m * pm) < 0
                    rho = torch.where(flip, m.abs() / (m.abs() + pm.abs()).clamp_min(1e-300), torch.zeros_like(m))
                    yeni = torch.where(flip, p - rho * (p - st["prev_p"]), sgd_yeni)
                    n_tetik += int(flip.sum())
                else:
                    yeni = sgd_yeni
                n_toplam += g.numel()
                st["prev_p"], st["m"] = p.detach().clone(), m
                p.copy_(yeni)
        self.diag = {"trigger_rate": n_tetik / max(n_toplam, 1)}
        return loss


class IRpropPlus(Optimizer):
    """iRprop+ (Igel & Hüsken, 2003). Yalnızca gradyan işareti; lr = başlangıç Delta."""
    def __init__(self, params, lr=1e-2, eta_minus=0.5, eta_plus=1.2, step_min=1e-12, step_max=50.0):
        super().__init__(params, dict(lr=lr, eta_minus=eta_minus, eta_plus=eta_plus,
                                      step_min=step_min, step_max=step_max))
        self.prev_loss = None

    @torch.no_grad()
    def step(self, loss=None):
        cur = float(loss) if loss is not None else None
        kotulesti = self.prev_loss is not None and cur is not None and cur > self.prev_loss
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g, st = p.grad.clone(), self.state[p]
                if not st:
                    st["delta"] = torch.full_like(p, group["lr"])
                    st["prev_g"], st["prev_dw"] = torch.zeros_like(p), torch.zeros_like(p)
                c = g * st["prev_g"]
                ayni, degisti = c > 0, c < 0
                st["delta"] = torch.where(ayni, (st["delta"] * group["eta_plus"]).clamp(max=group["step_max"]), st["delta"])
                st["delta"] = torch.where(degisti, (st["delta"] * group["eta_minus"]).clamp(min=group["step_min"]), st["delta"])
                dw = -torch.sign(g) * st["delta"]
                # işaret değişen koordinat: loss arttıysa önceki adımı geri al, artmadıysa bu adımda hareket etme
                dw = torch.where(degisti, -st["prev_dw"] if kotulesti else torch.zeros_like(dw), dw)
                g = torch.where(degisti, torch.zeros_like(g), g)
                p.add_(dw)
                st["prev_dw"], st["prev_g"] = dw, g
        self.prev_loss = cur
        return loss


def make_optimizer(name, params, lr, **ek):
    if name == "SGD":          return torch.optim.SGD(params, lr=lr)
    if name == "SGD+Momentum": return torch.optim.SGD(params, lr=lr, momentum=ek.get("momentum", 0.9))
    if name == "RF-SGD":       return RFSGD(params, lr=lr)
    if name == "RF-SGD-M":     return RFSGDM(params, lr=lr, beta=ek.get("beta", 0.9))
    if name == "RF-SGD-M(g)":  return RFSGDM(params, lr=lr, beta=ek.get("beta", 0.9), adim="g")
    if name == "iRprop+":      return IRpropPlus(params, lr=lr)
    if name == "Adam":         return torch.optim.Adam(params, lr=lr)       # betas=(0.9, 0.999), eps=1e-8
    if name == "RMSprop":      return torch.optim.RMSprop(params, lr=lr)    # alpha=0.99, eps=1e-8
    raise ValueError(name)


# 2.2 Doğruluk testleri
_BOLUM = '2.2 Doğruluk testleri'

def P(v): return torch.nn.Parameter(torch.tensor(v, dtype=DT))

# 1) el hesabı
p = P([2.0]); opt = RFSGD([p], lr=0.75)
p.grad = 2 * p.detach(); opt.step(); assert abs(p.item() + 1) < 1e-12
p.grad = 2 * p.detach(); opt.step(); assert abs(p.item()) < 1e-12

# 2) işaret değişmezse SGD ile aynı
a, b = P([5.0, 3.0]), P([5.0, 3.0]); oa, ob = RFSGD([a], lr=0.1), torch.optim.SGD([b], lr=0.1)
for _ in range(30):
    a.grad, b.grad = 2 * a.detach(), 2 * b.detach(); oa.step(); ob.step()
assert torch.allclose(a, b, atol=1e-12)

# 3) karışık adım: işaret değişen koordinatlar [x0, x1] aralığında, değişmeyenler SGD adımı
torch.manual_seed(1)
p = P([0.0] * 100); opt = RFSGD([p], lr=0.5); x0 = p.detach().clone()
g1 = torch.randn(100, dtype=DT); p.grad = g1.clone(); opt.step(); x1 = p.detach().clone()
g2 = torch.randn(100, dtype=DT); p.grad = g2.clone(); opt.step(); x2 = p.detach()
flip = (g1 * g2) < 0
assert ((x2 >= torch.minimum(x0, x1) - 1e-12) & (x2 <= torch.maximum(x0, x1) + 1e-12))[flip].all()
assert torch.allclose(x2[~flip], (x1 - 0.5 * g2)[~flip], atol=1e-12)

# 4) iRprop+ dar vadide yakınsıyor
A = torch.diag(torch.tensor([1.0, 100.0], dtype=DT)); p = P([-3.0, 2.0]); opt = IRpropPlus([p], lr=0.1)
for _ in range(300):
    loss = float(0.5 * p.detach() @ A @ p.detach()); p.grad = A @ p.detach(); opt.step(loss=loss)
assert float(0.5 * p.detach() @ A @ p.detach()) < 1e-8

# 5) beta = 0 -> RF-SGD-M == RF-SGD
torch.manual_seed(3)
a, b = P([1.0] * 20), P([1.0] * 20); oa, ob = RFSGD([a], lr=0.3), RFSGDM([b], lr=0.3, beta=0.0)
for _ in range(40):
    g = torch.randn(20, dtype=DT) + a.detach(); a.grad, b.grad = g.clone(), g.clone(); oa.step(); ob.step()
assert torch.allclose(a, b, atol=1e-12)

# 6) RF-SGD-M el hesabı, f = x^2, beta = 0.5, lr = 0.75
#    m: 4 -> 1 -> -1.25 (işaret değişti, rho = 5/9) -> -1.958
p = P([2.0]); opt = RFSGDM([p], lr=0.75, beta=0.5)
for beklenen in (-1.0, -1.75, -4 / 3, 13 / 96):
    p.grad = 2 * p.detach(); opt.step(); assert abs(p.item() - beklenen) < 1e-12, (p.item(), beklenen)

print("6/6 test geçti")


# 3. Deney düzeni
_BOLUM = '3. Deney düzeni'

def gurultu(rng, n, sigma):
    return np.zeros(n) if sigma == 0 else sigma * rng.standard_normal(n)


class RotatedQuadratic:
    def __init__(self, kappa=100.0, angle_deg=0.0, sigma=0.0, x0=(-3.0, 2.0), lam_min=1.0):
        self.kappa, self.angle, self.sigma = kappa, angle_deg, sigma
        th = math.radians(angle_deg)
        R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        # özdeğerler lam_min ve lam_min*kappa; ana deneylerde lam_min = 1
        self.A = torch.tensor(R @ np.diag([lam_min, lam_min * kappa]) @ R.T, dtype=DT)
        self._x0 = torch.tensor(x0, dtype=DT)
        self.f0 = float(self.f(self._x0))
        self.name = f"Quad(κ={kappa:g}, açı={angle_deg:g}°, σ={sigma:g})"
    def f(self, x):     return 0.5 * x @ self.A @ x
    def x0(self):       return self._x0.clone()
    def error(self, x): return float(self.f(x)) / self.f0
    def loss_and_grad(self, x, rng):
        return float(self.f(x)), self.A @ x + torch.tensor(gurultu(rng, 2, self.sigma), dtype=DT)


class Rosenbrock:
    def __init__(self, sigma=0.0, x0=(-1.5, 2.0)):
        self.sigma = sigma
        self._x0 = torch.tensor(x0, dtype=DT)
        self.f0 = float(self.f(self._x0))
        self.name = f"Rosenbrock(σ={sigma:g})"
    def f(self, x):     return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
    def x0(self):       return self._x0.clone()
    def error(self, x): return float(self.f(x)) / self.f0
    def loss_and_grad(self, x, rng):
        a, b = x[0], x[1]
        g = torch.stack([-2 * (1 - a) - 400 * a * (b - a ** 2), 200 * (b - a ** 2)])
        return float(self.f(x)), g + torch.tensor(gurultu(rng, 2, self.sigma), dtype=DT)


class LinRegProblem:
    """y = X w* + eps. Özellik ölçekleri 1..sqrt(cond_scale), eşit korelasyon rho. Veri seed'i sabit (42)."""
    def __init__(self, n=1000, d=50, cond_scale=100.0, rho=0.0, sigma=0.1, batch=None, data_seed=42):
        self.batch, self.n, self.d = batch, n, d
        self.cfg = dict(cond_scale=cond_scale, rho=rho, batch="full" if batch is None else batch)
        rng = np.random.default_rng(data_seed)
        Sigma = np.full((d, d), rho) + (1 - rho) * np.eye(d)
        L = np.linalg.cholesky(Sigma)
        s = np.logspace(0, np.log10(np.sqrt(cond_scale)), d)
        X = (rng.standard_normal((n, d)) @ L.T) * s
        w_star = rng.standard_normal(d)
        eps = gurultu(rng, n, sigma)
        y = X @ w_star + eps
        self.X, self.y = torch.tensor(X, dtype=DT), torch.tensor(y, dtype=DT)
        self.w_ls = torch.linalg.lstsq(self.X, self.y.unsqueeze(1)).solution.squeeze(1)   # kesin minimum
        self.L_ls, self.L0 = self.full_loss(self.w_ls), self.full_loss(self.x0())
        ev = np.linalg.eigvalsh((self.X.T @ self.X / n).numpy())
        self._stats = dict(n=n, d=d, **self.cfg, sigma=sigma, hess_eig_min=ev.min(), hess_eig_max=ev.max(),
                           hess_cond=ev.max() / ev.min(), y_mean=float(y.mean()), y_std=float(y.std()),
                           noise_std=float(eps.std()), w_star_norm=float(np.linalg.norm(w_star)))
        self.name = f"LinReg(cond={cond_scale:g}, ρ={rho:g}, B={self.cfg['batch']})"
    def x0(self):        return torch.zeros(self.d, dtype=DT)
    def full_loss(self, w):
        r = self.X @ w - self.y
        return float(0.5 * (r @ r) / self.n)
    def error(self, w):  return max(self.full_loss(w) - self.L_ls, 0.0) / (self.L0 - self.L_ls)
    def loss_and_grad(self, w, rng):
        if self.batch is None:
            Xb, yb = self.X, self.y
        else:
            idx = torch.as_tensor(rng.choice(self.n, self.batch, replace=False))
            Xb, yb = self.X[idx], self.y[idx]
        r = Xb @ w - yb
        return float(0.5 * (r @ r) / len(yb)), Xb.T @ r / len(yb)
    def stats(self):     return dict(self._stats)

CONFIGS = [dict(cond_scale=100, rho=0.0, batch=None), dict(cond_scale=10000, rho=0.0, batch=None)] + \
          [dict(cond_scale=1000, rho=r, batch=b) for r in (0.0, 0.5, 0.9) for b in (None, 32)]
probs = [LinRegProblem(**c) for c in CONFIGS]
STATS = pd.DataFrame([p.stats() for p in probs])
STATS.to_csv(f"{OUT}/veri_istatistikleri.csv", index=False)
rapor(STATS[["cond_scale", "rho", "batch", "hess_cond", "hess_eig_min", "hess_eig_max", "y_mean", "y_std", "noise_std", "w_star_norm"]].round(3), _BOLUM)

TUNE_SEEDS = (100, 101)
EVAL_SEEDS = (0, 1, 2, 3, 4)
IRAKSAMA = 1e6
TABAN = 1e-16                    # ölçüm tabanı: regresyonda float64 farkı ~1e-21'in altında gürültü

def lr_grid(lo, hi, n): return list(np.logspace(np.log10(lo), np.log10(hi), n))
GRID_Q = lr_grid(1e-4, 1.0, 9)       # kuadratik
GRID_R = lr_grid(1e-5, 1e-1, 9)      # Rosenbrock
GRID_L = lr_grid(1e-5, 1e-1, 9)      # regresyon


def run_once(problem, method, lr, seed, steps, ek=None, record_traj=False, record_diag=False):
    rng = np.random.default_rng(seed)            # gürültü ve mini-batch bu seed'e bağlı
    p = torch.nn.Parameter(problem.x0())
    opt = make_optimizer(method, [p], lr, **(ek or {}))
    errors = np.full(steps + 1, np.inf); errors[0] = problem.error(p.detach())
    traj = [p.detach().clone().numpy()] if record_traj else None
    diags, sure, diverged = [], 0.0, False
    for t in range(1, steps + 1):
        loss, g = problem.loss_and_grad(p.detach(), rng)
        p.grad = g
        t0 = time.perf_counter()
        if hasattr(opt, "diag") or isinstance(opt, IRpropPlus): opt.step(loss=loss)
        else: opt.step()
        sure += time.perf_counter() - t0
        e = problem.error(p.detach())
        if not np.isfinite(e) or e > IRAKSAMA:
            diverged = True; break
        errors[t] = e
        if record_traj: traj.append(p.detach().clone().numpy())
        if record_diag and hasattr(opt, "diag"): diags.append(dict(opt.diag))
    return dict(errors=errors, traj=np.array(traj) if record_traj else None,
                diag=diags, time_per_step=sure / t, diverged=diverged)


def final_score(errors):
    if not np.all(np.isfinite(errors)): return np.inf
    k = max(1, len(errors) // 10)
    return float(np.mean(np.log10(np.maximum(errors[-k:], TABAN))))   # −16 = tam çözüm


def ortalama_hata(errors):
    """Tüm adımlar boyunca ortalama log10 hata: tuning'de eşitlik bozucu (daha hızlı olan önde)."""
    if not np.all(np.isfinite(errors)): return np.inf
    return float(np.mean(np.log10(np.maximum(errors, TABAN))))


def steps_to(errors, esik):
    idx = np.where(errors <= esik)[0]
    return float(idx[0]) if len(idx) else np.nan


def compare(problem, methods, grid, steps, ek_izgara=None, record_traj=False, record_diag=False):
    """Her yöntem: (lr x ek ayar) araması tuning seed'leriyle, sonra eval seed'leriyle değerlendirme."""
    ek_izgara = ek_izgara or {}
    rows, runs = [], {}
    for m in methods:
        izg = ek_izgara.get(m, {})
        keys = list(izg)
        combos = list(itertools.product(*[izg[k] for k in keys])) or [()]
        best = None
        for lr in grid:
            for combo in combos:
                ek = dict(zip(keys, combo))
                errs = [run_once(problem, m, lr, s, steps, ek)["errors"] for s in TUNE_SEEDS]
                # birincil ölçüt final hata; ikisi de tabandaysa (eşitlik) ortalama hata, yani hız
                sc = (float(np.mean([final_score(e) for e in errs])), float(np.mean([ortalama_hata(e) for e in errs])))
                if best is None or sc < best[0]:
                    best = (sc, lr, ek)
        _, best_lr, best_ek = best
        rs = [run_once(problem, m, best_lr, s, steps, best_ek, record_traj, record_diag) for s in EVAL_SEEDS]
        runs[m] = rs
        fs = np.array([final_score(r["errors"]) for r in rs])
        row = dict(problem=problem.name, method=m, best_lr=best_lr, **{f"best_{k}": v for k, v in best_ek.items()},
                   n_diverged=sum(r["diverged"] for r in rs), log10_final_mean=fs.mean(), log10_final_std=fs.std())
        for th in (1e-2, 1e-4, 1e-6):
            st = np.array([steps_to(r["errors"], th) for r in rs])
            row[f"steps_to_{th:g}"] = np.nanmean(st) if np.any(~np.isnan(st)) else np.nan
            row[f"reached_{th:g}"] = int(np.sum(~np.isnan(st)))
        row["us_per_step"] = 1e6 * np.mean([r["time_per_step"] for r in rs])
        if record_diag and m in ("RF-SGD", "RF-SGD-M", "RF-SGD-M(g)"):
            row["trigger_rate"] = np.mean([np.mean([d["trigger_rate"] for d in r["diag"][-max(1, len(r["diag"]) // 10):]])
                                           for r in rs])
        rows.append(row)
    return pd.DataFrame(rows), runs

def hata_egrisi(runs, ax, baslik, ylabel="göreli hata"):
    for m, rs in runs.items():
        E = np.array([r["errors"] for r in rs], dtype=float)
        E = np.where(np.isfinite(E), np.maximum(E, 1e-16), np.nan)
        x = np.arange(E.shape[1])
        ax.plot(x, np.nanmedian(E, 0), color=RENK[m], lw=2, label=m)                         # 5 seed medyanı
        ax.fill_between(x, np.nanmin(E, 0), np.nanmax(E, 0), color=RENK[m], alpha=0.15, lw=0)  # min-max
    ax.set_yscale("log"); ax.set_xlabel("adım"); ax.set_ylabel(ylabel)
    ax.set_title(baslik, loc="left", fontsize=10); ax.legend()


def yol_grafigi(problem, runs, ax, baslik, n_goster=40, lim=None):
    trajs = {m: rs[0]["traj"][:n_goster] for m, rs in runs.items()}
    if lim is None:
        allp = np.vstack(list(trajs.values())); pad = 0.15 * (allp.max(0) - allp.min(0) + 1e-9)
        lim = ((allp[:, 0].min() - pad[0], allp[:, 0].max() + pad[0]), (allp[:, 1].min() - pad[1], allp[:, 1].max() + pad[1]))
    xs, ys = np.linspace(*lim[0], 250), np.linspace(*lim[1], 250)
    Z = np.array([[float(problem.f(torch.tensor([a, b], dtype=DT))) for a in xs] for b in ys])
    ax.contour(xs, ys, np.log10(Z + 1e-12), levels=20, colors="#e4e3df", linewidths=0.8)
    for m, t in trajs.items():
        ax.plot(t[:, 0], t[:, 1], "-o", color=RENK[m], ms=3, lw=1.2, label=m)
    ax.set_xlim(*lim[0]); ax.set_ylim(*lim[1]); ax.grid(False)
    ax.set_title(baslik, loc="left", fontsize=10); ax.legend()


def isi_haritasi(tablo, satirlar, sutunlar, ax, baslik, etiket):
    v = np.nanmax(np.abs(tablo[np.isfinite(tablo)])) if np.any(np.isfinite(tablo)) else 1
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("d", ["#2a78d6", "#f1f0ed", "#eb6834"])
    im = ax.imshow(tablo, cmap=cmap, vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(len(sutunlar)), sutunlar); ax.set_yticks(range(len(satirlar)), satirlar)
    for i in range(tablo.shape[0]):
        for j in range(tablo.shape[1]):
            ax.text(j, i, f"{tablo[i, j]:+.1f}", ha="center", va="center", fontsize=9)
    ax.grid(False); ax.set_title(baslik, loc="left", fontsize=10)
    plt.colorbar(im, ax=ax).set_label(etiket)


ANGLES, SIGMAS = (0, 15, 30, 45), (0.0, 0.1, 1.0, 10.0)

def fark_tablosu(df, a, b):
    """(açı, σ) hücreleri: log10_final(a) - log10_final(b); negatif = a daha iyi."""
    t = np.full((len(ANGLES), len(SIGMAS)), np.nan)
    for i, an in enumerate(ANGLES):
        for j, s in enumerate(SIGMAS):
            sub = df[(df.angle == an) & (df.sigma == s)].set_index("method")
            t[i, j] = sub.loc[a, "log10_final_mean"] - sub.loc[b, "log10_final_mean"]
    return t


# 4. Tur 1: RF-SGD
_BOLUM = '4. Tur 1: RF-SGD'


# 4.2 Deneyler
_BOLUM = '4.2 Deneyler'

METHODS = ["RF-SGD", "SGD", "SGD+Momentum", "iRprop+"]
rows, RUNS_A = [], {}
for ang in (0, 45):
    for kappa in (10, 100, 1000):
        prob = RotatedQuadratic(kappa=kappa, angle_deg=ang)
        ozet, runs = compare(prob, METHODS, GRID_Q, ADIM(500), record_traj=True)
        ozet["kappa"], ozet["angle"], ozet["sigma"] = kappa, ang, 0.0
        rows.append(ozet); RUNS_A[(kappa, ang)] = (prob, runs)
T11a = pd.concat(rows, ignore_index=True)
T11a.to_csv(f"{OUT}/tur1_T1_1a.csv", index=False)
rapor(T11a.pivot_table(index=["kappa", "angle"], columns="method", values="steps_to_1e-06"), _BOLUM)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, ang in zip(axes, (0, 45)):
    prob, runs = RUNS_A[(100, ang)]
    yol_grafigi(prob, runs, ax, f"κ=100, açı={ang}°, ilk 40 adım (seed 0)")
fig.savefig(f"{OUT}/tur1_yollar.png", bbox_inches="tight"); plt.close("all")
rapor(T11a[["kappa", "angle", "method", "best_lr", "us_per_step"]].round(3), _BOLUM)

satir = []
for lam in (1.0, 0.5):
    for lr in (0.5, 0.9, 1.0, 2.0, 2.5, 5.0):
        r = run_once(RotatedQuadratic(kappa=100, angle_deg=0, lam_min=lam), "RF-SGD", lr, 0, ADIM(500))
        satir.append(dict(lam_min=lam, lr=lr, steps_to_1e6=steps_to(r["errors"], 1e-6), diverged=r["diverged"]))
LAM = pd.DataFrame(satir)
LAM.to_csv(f"{OUT}/tur1_lam_min_kontrol.csv", index=False)
rapor(LAM.pivot_table(index="lam_min", columns="lr", values="steps_to_1e6"), _BOLUM)


# T1.1b/c: Açı × gürültü ızgarası, κ = 100, 500 adım
_BOLUM = 'T1.1b/c: Açı × gürültü ızgarası, κ = 100, 500 adım'

rows, RUNS_B = [], {}
for ang in ANGLES:
    for sg in SIGMAS:
        prob = RotatedQuadratic(kappa=100, angle_deg=ang, sigma=sg)
        ozet, runs = compare(prob, METHODS, GRID_Q, ADIM(500), record_diag=True)
        ozet["angle"], ozet["sigma"] = ang, sg
        rows.append(ozet); RUNS_B[(ang, sg)] = runs
T11bc = pd.concat(rows, ignore_index=True)
T11bc.to_csv(f"{OUT}/tur1_T1_1bc.csv", index=False)
rapor("log10 final hata (ortalama, 5 seed):")
display(T11bc.pivot_table(index=["angle", "sigma"], columns="method", values="log10_final_mean").round(2))
rapor("RF-SGD trigger rate (son %10 adım):")
rapor(T11bc[T11bc.method == "RF-SGD"].pivot_table(index="angle", columns="sigma", values="trigger_rate").round(2), _BOLUM)

fig, axes = plt.subplots(1, 3, figsize=(17, 4))
for ax, rakip in zip(axes, ("SGD", "SGD+Momentum", "iRprop+")):
    isi_haritasi(np.clip(fark_tablosu(T11bc, "RF-SGD", rakip), -10, 10), [f"{a}°" for a in ANGLES],
                 [f"σ={s:g}" for s in SIGMAS], ax, f"RF-SGD − {rakip} (±10'da kırpıldı)", "Δ log10 final (negatif = RF-SGD iyi)")
fig.savefig(f"{OUT}/tur1_isi_haritalari.png", bbox_inches="tight"); plt.close("all")


# T1.2: Rosenbrock, σ ∈ {0, 1, 10}, 3000 adım
_BOLUM = 'T1.2: Rosenbrock, σ ∈ {0, 1, 10}, 3000 adım'

rows, RUNS_R = [], {}
for sg in (0.0, 1.0, 10.0):
    prob = Rosenbrock(sigma=sg)
    ozet, runs = compare(prob, METHODS, GRID_R, ADIM(3000), record_traj=True)
    ozet["sigma"] = sg; rows.append(ozet); RUNS_R[sg] = (prob, runs)
T12 = pd.concat(rows, ignore_index=True)
T12.to_csv(f"{OUT}/tur1_T1_2.csv", index=False)
rapor(T12[["sigma", "method", "best_lr", "log10_final_mean", "log10_final_std", "steps_to_0.01", "steps_to_0.0001", "reached_0.0001"]].round(2), _BOLUM)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
prob, runs = RUNS_R[0.0]
yol_grafigi(prob, runs, axes[0], "Rosenbrock σ=0, yollar (seed 0)", n_goster=3000, lim=((-2, 2), (-1, 3)))
hata_egrisi(runs, axes[1], "Rosenbrock σ=0")
fig.savefig(f"{OUT}/tur1_rosenbrock.png", bbox_inches="tight"); plt.close("all")


# T2.1: Sentetik linear regression, 8 veri kümesi, 2000 adım
_BOLUM = 'T2.1: Sentetik linear regression, 8 veri kümesi, 2000 adım'

rows, RUNS_L = [], {}
for prob in probs:
    ozet, runs = compare(prob, METHODS, GRID_L, ADIM(2000), record_diag=True)
    for k, v in prob.cfg.items(): ozet[k] = v
    rows.append(ozet); RUNS_L[prob.name] = runs
T21 = pd.concat(rows, ignore_index=True)
T21.to_csv(f"{OUT}/tur1_T2_1.csv", index=False)
rapor("log10 final hata:")
display(T21.pivot_table(index=["cond_scale", "rho", "batch"], columns="method", values="log10_final_mean").round(2))
rapor("1e-2 eşiğine ulaşma adımı:")
rapor(T21.pivot_table(index=["cond_scale", "rho", "batch"], columns="method", values="steps_to_0.01").round(1), _BOLUM)

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for ax, prob in zip(axes.flat, probs[2:]):          # cond_scale = 1000
    hata_egrisi(RUNS_L[prob.name], ax, prob.name, ylabel="göreli fazla loss")
fig.savefig(f"{OUT}/tur1_regresyon.png", bbox_inches="tight"); plt.close("all")


# 4.3 Karşılaştırma
_BOLUM = '4.3 Karşılaştırma'

def flip_dogrulugu(prob, lr, seed, steps):
    """Gözlenen işaret değişimlerinden gürültüsüz gradyanda da gerçekleşenlerin oranı (son %10 adım)."""
    rng = np.random.default_rng(seed)               # run_once ile aynı gürültü dizisi
    p = torch.nn.Parameter(prob.x0()); opt = RFSGD([p], lr=lr)
    onceki, gozlenen, gercek = None, 0, 0
    for t in range(steps):
        x = p.detach().clone()
        g_temiz = prob.A @ x
        _, g = prob.loss_and_grad(x, rng)
        p.grad = g; opt.step()
        if onceki is not None and t >= steps - max(1, steps // 10):
            fo = (g * onceki[0]) < 0
            gozlenen += int(fo.sum()); gercek += int((fo & ((g_temiz * onceki[1]) < 0)).sum())
        onceki = (g.clone(), g_temiz.clone())
    return gercek / max(gozlenen, 1)

satir = []
for ang in ANGLES:
    for sg in (0.1, 1.0, 10.0):
        lr = T11bc[(T11bc.method == "RF-SGD") & (T11bc.angle == ang) & (T11bc.sigma == sg)].best_lr.iloc[0]
        prob = RotatedQuadratic(kappa=100, angle_deg=ang, sigma=sg)
        satir.append(dict(angle=ang, sigma=sg, gercek_oran=np.mean([flip_dogrulugu(prob, lr, s, ADIM(500)) for s in EVAL_SEEDS])))
FLIP = pd.DataFrame(satir)
FLIP.to_csv(f"{OUT}/tur1_flip_dogrulugu.csv", index=False)
rapor("Gözlenen işaret değişimlerinin gerçek olan oranı (5 seed ortalaması):")
rapor(FLIP.pivot_table(index="angle", columns="sigma", values="gercek_oran").round(2), _BOLUM)


# 5. Tur 2: RF-SGD-M
_BOLUM = '5. Tur 2: RF-SGD-M'


# 5.3 Deneyler
_BOLUM = '5.3 Deneyler'

METHODS2 = ["RF-SGD-M", "SGD+Momentum", "Adam", "RMSprop"]
BETA_IZGARA = [0.0, 0.5, 0.9, 0.95, 0.99]          # iki yöntem için aynı ızgara
EK2 = {"RF-SGD-M": {"beta": BETA_IZGARA}, "SGD+Momentum": {"momentum": BETA_IZGARA},
       "RF-SGD-M(g)": {"beta": BETA_IZGARA}}
TUR1_SABIT = ["RF-SGD", "SGD", "iRprop+"]      # Tur 2'de değişmeyen yöntemler: Tur 1 sonuçları kullanılıyor

def birlestir(yeni, tur1):
    """Tur 2 tablosu = yeni çalıştırılan yöntemler + Tur 1'deki değişmeyen yöntemler."""
    return pd.concat([yeni, tur1[tur1.method.isin(TUR1_SABIT)]], ignore_index=True)


# T1.1a (Tur 2): Gürültüsüz kuadratik
_BOLUM = 'T1.1a (Tur 2): Gürültüsüz kuadratik'

rows = []
for ang in (0, 45):
    for kappa in (10, 100, 1000):
        ozet, _ = compare(RotatedQuadratic(kappa=kappa, angle_deg=ang), METHODS2, GRID_Q, ADIM(500), EK2)
        ozet["kappa"], ozet["angle"] = kappa, ang
        rows.append(ozet)
T2_11a = birlestir(pd.concat(rows, ignore_index=True), T11a)
T2_11a.to_csv(f"{OUT}/tur2_T1_1a.csv", index=False)
rapor(T2_11a[T2_11a.method.isin(["RF-SGD-M", "RF-SGD", "SGD+Momentum", "Adam", "RMSprop"])][
    ["kappa", "angle", "method", "best_lr", "best_beta", "best_momentum", "steps_to_1e-06", "log10_final_mean"]].sort_values(["kappa", "angle"]).round(3), _BOLUM)


# T1.1b/c (Tur 2): Açı × gürültü, κ = 100
_BOLUM = 'T1.1b/c (Tur 2): Açı × gürültü, κ = 100'

rows, RUNS2_B = [], {}
for ang in ANGLES:
    for sg in SIGMAS:
        ozet, runs = compare(RotatedQuadratic(kappa=100, angle_deg=ang, sigma=sg), METHODS2, GRID_Q, ADIM(500), EK2, record_diag=True)
        ozet["angle"], ozet["sigma"] = ang, sg
        rows.append(ozet); RUNS2_B[(ang, sg)] = runs
T2_11bc = birlestir(pd.concat(rows, ignore_index=True), T11bc)
T2_11bc.to_csv(f"{OUT}/tur2_T1_1bc.csv", index=False)
rapor("log10 final hata:")
display(T2_11bc.pivot_table(index=["angle", "sigma"], columns="method", values="log10_final_mean").round(2))
rapor("Seçilen β (RF-SGD-M) / momentum (SGD+Momentum):")
display(pd.concat([T2_11bc[T2_11bc.method == "RF-SGD-M"].pivot_table(index="angle", columns="sigma", values="best_beta"),
                   T2_11bc[T2_11bc.method == "SGD+Momentum"].pivot_table(index="angle", columns="sigma", values="best_momentum")],
                  axis=1, keys=["RF-SGD-M β", "Momentum β"]))
rapor("Trigger rate (son %10 adım):")
rapor(T2_11bc[T2_11bc.method.isin(["RF-SGD", "RF-SGD-M"])].pivot_table(index=["angle", "sigma"], columns="method", values="trigger_rate").round(2), _BOLUM)

fig, axes = plt.subplots(1, 3, figsize=(17, 4))
for ax, rakip in zip(axes, ("RF-SGD", "SGD+Momentum", "Adam")):
    isi_haritasi(np.clip(fark_tablosu(T2_11bc, "RF-SGD-M", rakip), -5, 5), [f"{a}°" for a in ANGLES],
                 [f"σ={s:g}" for s in SIGMAS], ax, f"RF-SGD-M − {rakip} (±5'te kırpıldı)", "Δ log10 final (negatif = RF-SGD-M iyi)")
fig.savefig(f"{OUT}/tur2_isi_haritalari.png", bbox_inches="tight"); plt.close("all")

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, sg in zip(axes, (0.1, 1.0, 10.0)):
    runs = {"RF-SGD-M": RUNS2_B[(0, sg)]["RF-SGD-M"], "RF-SGD": RUNS_B[(0, sg)]["RF-SGD"],
            "SGD+Momentum": RUNS2_B[(0, sg)]["SGD+Momentum"], "Adam": RUNS2_B[(0, sg)]["Adam"]}
    hata_egrisi(runs, ax, f"açı=0°, σ={sg}")
fig.savefig(f"{OUT}/tur2_egriler.png", bbox_inches="tight"); plt.close("all")


# T1.2 (Tur 2): Rosenbrock
_BOLUM = 'T1.2 (Tur 2): Rosenbrock'

rows = []
for sg in (0.0, 1.0, 10.0):
    ozet, _ = compare(Rosenbrock(sigma=sg), METHODS2, GRID_R, ADIM(3000), EK2)
    ozet["sigma"] = sg; rows.append(ozet)
T2_12 = birlestir(pd.concat(rows, ignore_index=True), T12)
T2_12.to_csv(f"{OUT}/tur2_T1_2.csv", index=False)
rapor(T2_12[["sigma", "method", "best_lr", "best_beta", "best_momentum", "log10_final_mean", "log10_final_std", "steps_to_0.01"]].sort_values(["sigma", "method"]).round(2), _BOLUM)


# T2.1 (Tur 2): Sentetik linear regression
_BOLUM = 'T2.1 (Tur 2): Sentetik linear regression'

rows, RUNS2_L = [], {}
for prob in probs:
    ozet, runs = compare(prob, METHODS2, GRID_L, ADIM(2000), EK2, record_diag=True)
    for k, v in prob.cfg.items(): ozet[k] = v
    rows.append(ozet); RUNS2_L[prob.name] = runs
T2_21 = birlestir(pd.concat(rows, ignore_index=True), T21)
T2_21.to_csv(f"{OUT}/tur2_T2_1.csv", index=False)
rapor("log10 final hata:")
display(T2_21.pivot_table(index=["cond_scale", "rho", "batch"], columns="method", values="log10_final_mean").round(2))
rapor(T2_21[T2_21.method.isin(METHODS2)][["cond_scale", "rho", "batch", "method", "best_lr", "best_beta", "best_momentum"]].round(4), _BOLUM)

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, prob in zip(axes, [p for p in probs if p.cfg["batch"] == 32]):
    runs = {"RF-SGD-M": RUNS2_L[prob.name]["RF-SGD-M"], "RF-SGD": RUNS_L[prob.name]["RF-SGD"],
            "SGD+Momentum": RUNS2_L[prob.name]["SGD+Momentum"], "Adam": RUNS2_L[prob.name]["Adam"]}
    hata_egrisi(runs, ax, prob.name, ylabel="göreli fazla loss")
fig.savefig(f"{OUT}/tur2_regresyon_minibatch.png", bbox_inches="tight"); plt.close("all")


# 5.4 Ablasyon: kazanç işaret testinden mi, yumuşatılmış adımdan mı?
_BOLUM = '5.4 Ablasyon: kazanç işaret testinden mi, yumuşatılmış adımdan mı?'

rows = []
for ang in ANGLES:
    for sg in (0.1, 1.0, 10.0):
        ozet, _ = compare(RotatedQuadratic(kappa=100, angle_deg=ang, sigma=sg), ["RF-SGD-M(g)"], GRID_Q, ADIM(500), EK2, record_diag=True)
        ozet["problem_tipi"], ozet["angle"], ozet["sigma"] = "kuadratik", ang, sg
        rows.append(ozet)
for prob in [p for p in probs if p.cfg["batch"] == 32]:
    ozet, _ = compare(prob, ["RF-SGD-M(g)"], GRID_L, ADIM(2000), EK2, record_diag=True)
    ozet["problem_tipi"] = "regresyon B=32"
    for k, v in prob.cfg.items(): ozet[k] = v
    rows.append(ozet)
ABL = pd.concat(rows, ignore_index=True)
ABL.to_csv(f"{OUT}/tur2_ablasyon.csv", index=False)

# aynı hücrelerdeki RF-SGD, RF-SGD-M ve momentum değerleriyle yan yana
ref = pd.concat([T2_11bc[T2_11bc.sigma > 0], T2_21[T2_21.batch == 32]], ignore_index=True)
tablo = pd.concat([ref[ref.method.isin(["RF-SGD", "RF-SGD-M", "SGD+Momentum"])], ABL], ignore_index=True)
tablo["hucre"] = tablo.problem.str.replace("Quad(κ=100, ", "Quad(", regex=False)
ABL_OZET = tablo.pivot_table(index="hucre", columns="method", values="log10_final_mean", sort=False)[
    ["RF-SGD", "RF-SGD-M(g)", "RF-SGD-M", "SGD+Momentum"]]
ABL_OZET["kazanç: test (RF-SGD → M(g))"] = ABL_OZET["RF-SGD"] - ABL_OZET["RF-SGD-M(g)"]
ABL_OZET["kazanç: adım (M(g) → M)"] = ABL_OZET["RF-SGD-M(g)"] - ABL_OZET["RF-SGD-M"]
rapor("log10 final hata; kazanç sütunları pozitif = iyileşme")
rapor(ABL_OZET.round(2), _BOLUM)


# 5.5 Karşılaştırma
_BOLUM = '5.5 Karşılaştırma'


# 6. Sonuç
_BOLUM = '6. Sonuç'

# Tur 2 tablolarındaki Tur 1 kopyaları (RF-SGD, SGD, iRprop+) çift sayılmasın diye çıkarılıyor
yeni = lambda df: df[~df.method.isin(TUR1_SABIT)]
pd.concat([T11a.assign(tur=1), T11bc.assign(tur=1), T12.assign(tur=1), T21.assign(tur=1),
           yeni(T2_11a).assign(tur=2), yeni(T2_11bc).assign(tur=2), yeni(T2_12).assign(tur=2), yeni(T2_21).assign(tur=2),
           ABL.assign(tur="2-ablasyon")],
          ignore_index=True).to_csv(f"{OUT}/tum_sonuclar.csv", index=False)
print(sorted(os.listdir(OUT)))

_OZET.close()
print("bitti:", OUT)
