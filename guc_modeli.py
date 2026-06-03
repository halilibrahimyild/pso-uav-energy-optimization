import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# GRAFİK AYARLARI
# ============================================================
plt.rcParams['figure.facecolor'] = '#0a0a0a'
plt.rcParams['axes.facecolor'] = '#111111'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.labelcolor'] = '#ffffff'
plt.rcParams['xtick.color'] = '#aaaaaa'
plt.rcParams['ytick.color'] = '#aaaaaa'
plt.rcParams['text.color'] = '#ffffff'
plt.rcParams['grid.color'] = '#222222'
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['font.size'] = 11

COLORS = {
    'cyan':   '#00d4ff',
    'green':  '#00ff88',
    'orange': '#ff6b35',
    'purple': '#b388ff',
    'yellow': '#ffd700',
    'red':    '#ff4444',
    'pink':   '#ff69b4',
}

custom_cmap = LinearSegmentedColormap.from_list(
    'pso_cmap',
    ['#001f4d', '#00d4ff', '#00ff88', '#ffd700', '#ff4444']
)

# ============================================================
# VERİYİ OKU VE TEMİZLE
# ============================================================
print("Veri yükleniyor...")
df = pd.read_csv('flights.csv', low_memory=False)
df = df[df['route'].str.startswith('R', na=False)].copy()

for col in ['speed', 'payload', 'altitude', 'battery_voltage', 'battery_current']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=['speed', 'payload', 'altitude',
                        'battery_voltage', 'battery_current'])
df['power'] = df['battery_voltage'] * df['battery_current'].abs()

# Her uçuş için cruise güç hesapla
cruise_data = []
for fid, group in df.groupby('flight'):
    n = len(group)
    if n < 50:
        continue
    cruise = group.iloc[int(n*0.2):int(n*0.8)]
    cruise_data.append({
        'flight':    fid,
        'speed':     group['speed'].iloc[0],
        'payload':   group['payload'].iloc[0],
        'altitude':  group['altitude'].iloc[0],
        'avg_power': cruise['power'].mean(),
        'wind_speed': cruise['wind_speed'].mean(),
        'duration':  group['time'].max(),
        'total_energy': np.trapezoid(group['power'], group['time'])
    })

cdf = pd.DataFrame(cruise_data).dropna()
print(f"Toplam uçuş: {len(cdf)}")
print("Veri hazır!")

# ============================================================
# FİZİK TABANLI GÜÇ MODELİ
# ============================================================
# Quadrotor güç formülü:
# P = a * v^3 + b * m * g + c * payload + d * altitude + e
# 
# Burada:
# v^3  → Hava direnci gücü (hız küpüyle artar)
# m*g  → Ağırlığı taşımak için gereken güç
# payload → Yük ağırlığının güce etkisi
# altitude → İrtifanın hava yoğunluğuna etkisi

print("Fizik tabanlı güç modeli kuruluyor...")

def guc_modeli(X, a, b, c, d, e):
    """
    X → [hız, yük, irtifa] matrisi
    a → Hız kübü katsayısı (aerodinamik direnç)
    b → Yük katsayısı (taşıma gücü)
    c → İrtifa katsayısı (hava yoğunluğu etkisi)
    d → Sabit güç (hover için gereken minimum güç)
    e → Karışım terimi
    """
    v = X[0]   # hız (m/s)
    p = X[1]   # yük (gram)
    h = X[2]   # irtifa (metre)
    
    return a * v**3 + b * p + c * h + d * v**2 + e

# Giriş ve çıkış verilerini hazırla
X_data = np.array([
    cdf['speed'].values,    # hız
    cdf['payload'].values,  # yük
    cdf['altitude'].values  # irtifa
])
y_data = cdf['avg_power'].values

# Modeli veriye fit et
popt, pcov = curve_fit(guc_modeli, X_data, y_data, maxfev=10000)
a, b, c, d, e = popt

print(f"\nModel Katsayıları:")
print(f"  Hız³ katsayısı (a): {a:.6f}")
print(f"  Yük katsayısı  (b): {b:.6f}")
print(f"  İrtifa katsayısı (c): {c:.6f}")
print(f"  Hız² katsayısı (d): {d:.6f}")
print(f"  Sabit güç (e): {e:.2f} W")

# Tahmin yap
y_pred_physics = guc_modeli(X_data, *popt)
r2_physics = r2_score(y_data, y_pred_physics)
mae_physics = mean_absolute_error(y_data, y_pred_physics)

print(f"\nFizik Modeli Performansı:")
print(f"  R² skoru: {r2_physics:.4f} (1.0 = mükemmel)")
print(f"  Ortalama Hata: {mae_physics:.2f} W")

# ============================================================
# POLİNOM REGRESYON MODELİ
# ============================================================
# Fizik modeline ek olarak makine öğrenmesi modeli de kuruyoruz
# İkisini karşılaştırıp hangisi daha iyi fit ediyor göreceğiz

print("\nPolinom regresyon modeli kuruluyor...")

# Giriş değişkenleri: hız, yük, irtifa
X_ml = cdf[['speed', 'payload', 'altitude']].values
y_ml = cdf['avg_power'].values

# Veriyi eğitim (%80) ve test (%20) olarak böl
X_train, X_test, y_train, y_test = train_test_split(
    X_ml, y_ml, test_size=0.2, random_state=42
)

# 2. derece polinom özellikler oluştur
# Örnek: [hız, yük, irtifa] → [1, hız, yük, irtifa, hız², hız×yük, ...]
poly = PolynomialFeatures(degree=2, include_bias=True)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Doğrusal regresyon modeli kur
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Test verisi üzerinde tahmin yap
y_pred_train = model.predict(X_train_poly)
y_pred_test = model.predict(X_test_poly)

r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
mae_train = mean_absolute_error(y_train, y_pred_train)
mae_test = mean_absolute_error(y_test, y_pred_test)

print(f"\nPolinom Regresyon Performansı:")
print(f"  Eğitim R²:  {r2_train:.4f}")
print(f"  Test R²:    {r2_test:.4f}")
print(f"  Eğitim MAE: {mae_train:.2f} W")
print(f"  Test MAE:   {mae_test:.2f} W")

# Tüm veri için tahmin
y_pred_all = model.predict(poly.transform(X_ml))

print(f"\nModel karşılaştırması:")
print(f"  Fizik Modeli R²:     {r2_physics:.4f}")
print(f"  Polinom Regresyon R²: {r2_test:.4f}")

# PSO için kullanılacak modeli seç
# Hangisi daha iyi R² veriyorsa onu kullan
if r2_test >= r2_physics:
    best_model = 'polinom'
    print(f"\n→ PSO için POLİNOM REGRESYON modeli kullanılacak")
else:
    best_model = 'fizik'
    print(f"\n→ PSO için FİZİK modeli kullanılacak")

# PSO'da çağrılacak güç tahmin fonksiyonu
def guc_tahmin(hiz, yuk, irtifa):
    """
    PSO her iterasyonda bu fonksiyonu çağıracak
    hiz    → m/s cinsinden uçuş hızı
    yuk    → gram cinsinden yük ağırlığı
    irtifa → metre cinsinden uçuş irtifası
    
    Döndürür: tahmini güç tüketimi (Watt)
    """
    if best_model == 'polinom':
        X_input = poly.transform([[hiz, yuk, irtifa]])
        return model.predict(X_input)[0]
    else:
        return guc_modeli(
            np.array([[hiz], [yuk], [irtifa]]),
            *popt
        )[0]

# Test edelim
test_guc = guc_tahmin(8.0, 250, 50)
print(f"\nTest tahmini (8m/s, 250g yük, 50m irtifa): {test_guc:.1f} W")

# ============================================================
# GRAFİK 1 — GERÇEK vs TAHMİN EDİLEN GÜÇ
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Güç Modeli Doğrulama — Gerçek vs Tahmin Edilen Güç',
             fontsize=15, fontweight='bold', color='white', y=0.98)

# --- GRAFIK 1A: FİZİK MODELİ ---
ax = axes[0]

ax.scatter(y_data, y_pred_physics,
           color=COLORS['cyan'], alpha=0.6, s=60,
           edgecolors='white', linewidth=0.3,
           label='Uçuş Noktaları', zorder=3)

# Mükemmel tahmin çizgisi (45 derece)
min_val = min(y_data.min(), y_pred_physics.min())
max_val = max(y_data.max(), y_pred_physics.max())
ax.plot([min_val, max_val], [min_val, max_val],
        color=COLORS['yellow'], linewidth=2,
        linestyle='--', label='Mükemmel Tahmin', zorder=4)

# Hata bantları
ax.fill_between([min_val, max_val],
                [min_val - 30, max_val - 30],
                [min_val + 30, max_val + 30],
                alpha=0.1, color=COLORS['yellow'],
                label='±30W Hata Bandı')

ax.set_xlabel('Gerçek Güç (W)', fontsize=12)
ax.set_ylabel('Tahmin Edilen Güç (W)', fontsize=12)
ax.set_title(f'Fizik Modeli\nR² = {r2_physics:.4f} | MAE = {mae_physics:.1f}W',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# R² değerini grafiğe yaz
ax.text(0.05, 0.95, f'R² = {r2_physics:.4f}',
        transform=ax.transAxes,
        fontsize=14, fontweight='bold',
        color=COLORS['yellow'],
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='#222222', alpha=0.8))

# --- GRAFIK 1B: POLİNOM REGRESYON ---
ax = axes[1]

ax.scatter(y_ml, y_pred_all,
           color=COLORS['green'], alpha=0.6, s=60,
           edgecolors='white', linewidth=0.3,
           label='Uçuş Noktaları', zorder=3)

min_val2 = min(y_ml.min(), y_pred_all.min())
max_val2 = max(y_ml.max(), y_pred_all.max())
ax.plot([min_val2, max_val2], [min_val2, max_val2],
        color=COLORS['yellow'], linewidth=2,
        linestyle='--', label='Mükemmel Tahmin', zorder=4)

ax.fill_between([min_val2, max_val2],
                [min_val2 - 30, max_val2 - 30],
                [min_val2 + 30, max_val2 + 30],
                alpha=0.1, color=COLORS['yellow'],
                label='±30W Hata Bandı')

r2_all = r2_score(y_ml, y_pred_all)
mae_all = mean_absolute_error(y_ml, y_pred_all)

ax.set_xlabel('Gerçek Güç (W)', fontsize=12)
ax.set_ylabel('Tahmin Edilen Güç (W)', fontsize=12)
ax.set_title(f'Polinom Regresyon\nR² = {r2_all:.4f} | MAE = {mae_all:.1f}W',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax.text(0.05, 0.95, f'R² = {r2_all:.4f}',
        transform=ax.transAxes,
        fontsize=14, fontweight='bold',
        color=COLORS['yellow'],
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='#222222', alpha=0.8))

plt.tight_layout()
plt.savefig('grafik_model1_gercek_tahmin.png', dpi=150,
            bbox_inches='tight', facecolor='#0a0a0a')
plt.show()
print("Grafik 1 kaydedildi!")

# ============================================================
# GRAFİK 2 — HIZ-GÜÇ EĞRİSİ VE HATA DAĞILIMI
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Güç Modeli — Hız-Güç Eğrisi ve Hata Analizi',
             fontsize=15, fontweight='bold', color='white', y=0.98)

# --- GRAFIK 2A: HIZ-GÜÇ EĞRİSİ ---
ax = axes[0]

# Gerçek veri noktaları — yüke göre renklendir
payload_colors = {
    0.0:   COLORS['cyan'],
    250.0: COLORS['green'],
    500.0: COLORS['orange'],
    750.0: COLORS['purple']
}

for payload_val, color in payload_colors.items():
    subset = cdf[cdf['payload'] == payload_val]
    ax.scatter(subset['speed'], subset['avg_power'],
               color=color, alpha=0.5, s=50,
               edgecolors='white', linewidth=0.3,
               label=f'Gerçek — {payload_val:.0f}g', zorder=3)

# Model tahmin eğrileri
hiz_aralik = np.linspace(4, 12, 100)
irtifa_sabit = 50  # sabit irtifa

for payload_val, color in payload_colors.items():
    guc_tahminler = [guc_tahmin(v, payload_val, irtifa_sabit)
                     for v in hiz_aralik]
    ax.plot(hiz_aralik, guc_tahminler,
            color=color, linewidth=2.5,
            linestyle='-', alpha=0.9,
            label=f'Model — {payload_val:.0f}g', zorder=4)

ax.set_xlabel('Uçuş Hızı (m/s)', fontsize=12)
ax.set_ylabel('Güç Tüketimi (W)', fontsize=12)
ax.set_title(f'Hız-Güç Eğrisi (İrtifa={irtifa_sabit}m sabit)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)
ax.set_xticks([4, 6, 8, 10, 12])

# Minimum güç noktasını işaretle
for payload_val, color in payload_colors.items():
    guc_tahminler = [guc_tahmin(v, payload_val, irtifa_sabit)
                     for v in hiz_aralik]
    min_idx = np.argmin(guc_tahminler)
    min_hiz = hiz_aralik[min_idx]
    min_guc = guc_tahminler[min_idx]
    ax.plot(min_hiz, min_guc, marker='*',
            markersize=15, color=color,
            markeredgecolor='white', markeredgewidth=1,
            zorder=5)
    ax.annotate(f'{min_guc:.0f}W\n@{min_hiz:.1f}m/s',
                xy=(min_hiz, min_guc),
                xytext=(5, -25),
                textcoords='offset points',
                fontsize=7, color=color,
                fontweight='bold')

# --- GRAFIK 2B: HATA DAĞILIMI ---
ax = axes[1]

# Hata hesapla
hatalar = y_ml - y_pred_all
hata_std = hatalar.std()
hata_ort = hatalar.mean()

# Histogram
n, bins, patches = ax.hist(hatalar, bins=25,
                            color=COLORS['cyan'],
                            edgecolor='white',
                            linewidth=0.5,
                            alpha=0.7,
                            density=True,
                            label='Hata Dağılımı')

# Normal dağılım eğrisi ekle
from scipy.stats import norm
x_norm = np.linspace(hatalar.min(), hatalar.max(), 100)
y_norm = norm.pdf(x_norm, hata_ort, hata_std)
ax.plot(x_norm, y_norm,
        color=COLORS['yellow'],
        linewidth=2.5,
        label=f'Normal Dağılım\nμ={hata_ort:.1f}W, σ={hata_std:.1f}W')

# Sıfır çizgisi
ax.axvline(0, color=COLORS['green'],
           linewidth=2, linestyle='--',
           label='Sıfır Hata', alpha=0.8)

# ±1 standart sapma
ax.axvline(hata_std, color=COLORS['orange'],
           linewidth=1.5, linestyle=':',
           label=f'±1σ = ±{hata_std:.1f}W', alpha=0.8)
ax.axvline(-hata_std, color=COLORS['orange'],
           linewidth=1.5, linestyle=':', alpha=0.8)

ax.set_xlabel('Tahmin Hatası (W)', fontsize=12)
ax.set_ylabel('Yoğunluk', fontsize=12)
ax.set_title('Model Hata Dağılımı', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# İstatistikleri grafiğe yaz
stats_text = (f'Ortalama Hata: {hata_ort:.2f}W\n'
              f'Std Sapma: {hata_std:.2f}W\n'
              f'Max Hata: {abs(hatalar).max():.1f}W\n'
              f'%95 Güven: ±{1.96*hata_std:.1f}W')
ax.text(0.97, 0.95, stats_text,
        transform=ax.transAxes,
        fontsize=9, color='white',
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round',
                  facecolor='#222222', alpha=0.8))

plt.tight_layout()
plt.savefig('grafik_model2_hiz_guc.png', dpi=150,
            bbox_inches='tight', facecolor='#0a0a0a')
plt.show()
print("Grafik 2 kaydedildi!")

# ============================================================
# GRAFİK 3 — 3D MODEL YÜZEYİ
# ============================================================
from scipy.interpolate import griddata

fig = plt.figure(figsize=(18, 7))
fig.patch.set_facecolor('#0a0a0a')
fig.suptitle('Güç Modeli — 3D Tahmin Yüzeyi',
             fontsize=15, fontweight='bold', color='white', y=0.98)

# --- GRAFIK 3A: HIZ × YÜK → GÜÇ 3D ---
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_facecolor('#111111')

# Model tahmin yüzeyi oluştur
hiz_grid = np.linspace(4, 12, 30)
yuk_grid = np.linspace(0, 750, 30)
Hiz, Yuk = np.meshgrid(hiz_grid, yuk_grid)

# Her grid noktası için güç tahmin et
Guc = np.zeros_like(Hiz)
for i in range(Hiz.shape[0]):
    for j in range(Hiz.shape[1]):
        Guc[i,j] = guc_tahmin(Hiz[i,j], Yuk[i,j], 50)

surf1 = ax1.plot_surface(Hiz, Yuk, Guc,
                          cmap=custom_cmap,
                          alpha=0.85,
                          edgecolor='none')

# Gerçek veri noktalarını üstüne ekle
ax1.scatter(cdf['speed'], cdf['payload'], cdf['avg_power'],
            color=COLORS['yellow'], s=30,
            zorder=5, depthshade=False,
            label='Gerçek Veri')

ax1.set_xlabel('Hız (m/s)', fontsize=10, labelpad=8)
ax1.set_ylabel('Yük (gram)', fontsize=10, labelpad=8)
ax1.set_zlabel('Güç (W)', fontsize=10, labelpad=8)
ax1.set_title('Hız × Yük → Güç\n(İrtifa=50m sabit)',
              fontsize=11, fontweight='bold', pad=15)
ax1.tick_params(colors='#aaaaaa', labelsize=8)
ax1.xaxis.pane.fill = False
ax1.yaxis.pane.fill = False
ax1.zaxis.pane.fill = False
ax1.xaxis.pane.set_edgecolor('#333333')
ax1.yaxis.pane.set_edgecolor('#333333')
ax1.zaxis.pane.set_edgecolor('#333333')
fig.colorbar(surf1, ax=ax1, shrink=0.5, pad=0.1,
             label='Güç (W)').ax.yaxis.label.set_color('white')

# --- GRAFIK 3B: HIZ × İRTİFA → GÜÇ 3D ---
ax2 = fig.add_subplot(122, projection='3d')
ax2.set_facecolor('#111111')

hiz_grid2 = np.linspace(4, 12, 30)
irt_grid2 = np.linspace(25, 100, 30)
Hiz2, Irt2 = np.meshgrid(hiz_grid2, irt_grid2)

Guc2 = np.zeros_like(Hiz2)
for i in range(Hiz2.shape[0]):
    for j in range(Hiz2.shape[1]):
        Guc2[i,j] = guc_tahmin(Hiz2[i,j], 250, Irt2[i,j])

surf2 = ax2.plot_surface(Hiz2, Irt2, Guc2,
                          cmap=custom_cmap,
                          alpha=0.85,
                          edgecolor='none')

ax2.scatter(cdf['speed'], cdf['altitude'], cdf['avg_power'],
            color=COLORS['yellow'], s=30,
            zorder=5, depthshade=False,
            label='Gerçek Veri')

ax2.set_xlabel('Hız (m/s)', fontsize=10, labelpad=8)
ax2.set_ylabel('İrtifa (m)', fontsize=10, labelpad=8)
ax2.set_zlabel('Güç (W)', fontsize=10, labelpad=8)
ax2.set_title('Hız × İrtifa → Güç\n(Yük=250g sabit)',
              fontsize=11, fontweight='bold', pad=15)
ax2.tick_params(colors='#aaaaaa', labelsize=8)
ax2.xaxis.pane.fill = False
ax2.yaxis.pane.fill = False
ax2.zaxis.pane.fill = False
ax2.xaxis.pane.set_edgecolor('#333333')
ax2.yaxis.pane.set_edgecolor('#333333')
ax2.zaxis.pane.set_edgecolor('#333333')
fig.colorbar(surf2, ax=ax2, shrink=0.5, pad=0.1,
             label='Güç (W)').ax.yaxis.label.set_color('white')

plt.tight_layout()
plt.savefig('grafik_model3_3d.png', dpi=150,
            bbox_inches='tight', facecolor='#0a0a0a')
plt.show()
print("Grafik 3 kaydedildi!")

# ============================================================
# GRAFİK 4 — ÖZELLİK ÖNEMİ VE MODEL ÖZETI
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Güç Modeli — Özellik Önemi ve Model Özeti',
             fontsize=15, fontweight='bold', color='white', y=0.98)

# --- GRAFIK 4A: ÖZELLİK ÖNEMİ ---
ax = axes[0]

# Her değişkenin güce etkisini ölç
# Bir değişkeni sabit tutup diğerini değiştirince güç ne kadar değişiyor?
hiz_etkisi = guc_tahmin(12, 250, 50) - guc_tahmin(4, 250, 50)
yuk_etkisi = guc_tahmin(8, 750, 50) - guc_tahmin(8, 0, 50)
irtifa_etkisi = guc_tahmin(8, 250, 100) - guc_tahmin(8, 250, 25)

etkiler = {
    'Hız\n(4→12 m/s)': abs(hiz_etkisi),
    'Yük\n(0→750 g)': abs(yuk_etkisi),
    'İrtifa\n(25→100 m)': abs(irtifa_etkisi),
}

renkler = [COLORS['cyan'], COLORS['orange'], COLORS['green']]
bars = ax.barh(list(etkiler.keys()),
               list(etkiler.values()),
               color=renkler,
               edgecolor='white',
               linewidth=0.8,
               alpha=0.85)

# Değerleri çubukların üstüne yaz
for bar, val in zip(bars, etkiler.values()):
    ax.text(bar.get_width() + 0.5,
            bar.get_y() + bar.get_height()/2,
            f'+{val:.1f}W',
            va='center', fontsize=11,
            color=COLORS['yellow'],
            fontweight='bold')

ax.set_xlabel('Güç Tüketimine Etkisi (W)', fontsize=12)
ax.set_title('Parametrelerin Güce Etkisi\n(Diğerleri Sabitken)',
             fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.set_xlim(0, max(etkiler.values()) * 1.25)

# --- GRAFIK 4B: MODEL ÖZET TABLOSU ---
ax = axes[1]
ax.axis('off')

# Özet tablo verisi
tablo_data = [
    ['Model Türü', 'Polinom Regresyon (2. Derece)'],
    ['Eğitim Verisi', f'%80 ({int(len(cdf)*0.8)} uçuş)'],
    ['Test Verisi', f'%20 ({int(len(cdf)*0.2)} uçuş)'],
    ['Eğitim R²', f'{r2_train:.4f}'],
    ['Test R²', f'{r2_test:.4f}'],
    ['Ortalama Hata', f'{mae_test:.2f} W'],
    ['Fizik Modeli R²', f'{r2_physics:.4f}'],
    ['Fizik Modeli MAE', f'{mae_physics:.2f} W'],
    ['Seçilen Model', best_model.upper()],
    ['Giriş Değişkenleri', 'Hız, Yük, İrtifa'],
    ['Çıkış', 'Güç Tüketimi (W)'],
    ['Veri Seti', 'Rodrigues 2021'],
    ['Toplam Uçuş', f'{len(cdf)} uçuş'],
]

tablo = ax.table(
    cellText=tablo_data,
    colLabels=['Parametre', 'Değer'],
    cellLoc='left',
    loc='center',
    colWidths=[0.45, 0.55]
)

tablo.auto_set_font_size(False)
tablo.set_fontsize(10)
tablo.scale(1, 1.8)

# Tablo renklerini ayarla
for (row, col), cell in tablo.get_celld().items():
    cell.set_facecolor('#1a1a1a')
    cell.set_edgecolor('#333333')
    cell.set_text_props(color='white')
    if row == 0:
        cell.set_facecolor('#00d4ff')
        cell.set_text_props(color='#111111', fontweight='bold')
    elif row % 2 == 0:
        cell.set_facecolor('#111111')
    if row == len(tablo_data) and col == 1:
        cell.set_facecolor('#00ff88')
        cell.set_text_props(color='#111111', fontweight='bold')

ax.set_title('Model Performans Özeti',
             fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('grafik_model4_ozet.png', dpi=150,
            bbox_inches='tight', facecolor='#0a0a0a')
plt.show()
print("Grafik 4 kaydedildi!")

# ============================================================
# MODELİ KAYDET — PSO'DA KULLANILACAK
# ============================================================
import pickle

model_data = {
    'model': model,
    'poly': poly,
    'best_model': best_model,
    'popt': popt,
    'r2_test': r2_test,
    'r2_physics': r2_physics,
    'mae_test': mae_test,
}

with open('guc_modeli.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("\n" + "="*50)
print("GÜÇ MODELİ BAŞARIYLA OLUŞTURULDU!")
print("="*50)
print(f"  Seçilen model: {best_model.upper()}")
print(f"  Test R²: {r2_test:.4f}")
print(f"  Test MAE: {mae_test:.2f} W")
print(f"  Model kaydedildi: guc_modeli.pkl")
print("="*50)
print("\nPSO optimizasyonuna geçmeye hazır!")