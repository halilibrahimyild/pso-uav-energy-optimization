import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import pickle
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
    'white':  '#ffffff',
}

custom_cmap = LinearSegmentedColormap.from_list(
    'pso_cmap',
    ['#001f4d', '#00d4ff', '#00ff88', '#ffd700', '#ff4444']
)

# ============================================================
# GÜÇ MODELİNİ YÜKLE
# ============================================================
print("Güç modeli yükleniyor...")

with open('guc_modeli.pkl', 'rb') as f:
    model_data = pickle.load(f)

model    = model_data['model']
poly     = model_data['poly']
best_model = model_data['best_model']
popt     = model_data['popt']

print(f"Model yüklendi: {best_model.upper()}")
print(f"Test R²: {model_data['r2_test']:.4f}")

# ============================================================
# GÜÇ TAHMİN FONKSİYONU
# ============================================================
def guc_modeli_fizik(X, a, b, c, d, e):
    v = X[0]
    p = X[1]
    h = X[2]
    return a * v**3 + b * p + c * h + d * v**2 + e

def guc_tahmin(hiz, yuk, irtifa):
    """
    PSO her iterasyonda bu fonksiyonu çağırır.
    Girdi: hız (m/s), yük (gram), irtifa (metre)
    Çıktı: tahmini güç (Watt)
    """
    if best_model == 'polinom':
        X_input = poly.transform([[hiz, yuk, irtifa]])
        return model.predict(X_input)[0]
    else:
        return guc_modeli_fizik(
            np.array([[hiz], [yuk], [irtifa]]),
            *popt
        )[0]

print("Güç tahmin fonksiyonu hazır!")
print(f"\nTest: 8m/s, 250g, 50m → {guc_tahmin(8, 250, 50):.1f} W")

# ============================================================
# PSO PARAMETRELERİ
# ============================================================
print("\nPSO parametreleri ayarlanıyor...")

# --- PROBLEM TANIMI ---
# Ne optimize ediyoruz?
# Bir İHA belirli bir mesafeyi uçacak
# Hangi hız ve irtifada en az enerji harcıyor?

MESAFE = 1000        # metre — uçulacak toplam mesafe
YUK = 250            # gram — sabit yük ağırlığı

# --- PSO HİPERPARAMETRELERİ ---
N_PARCACIK = 50      # Kaç parçacık olacak (sürü büyüklüğü)
N_ITERASYON = 200    # Kaç iterasyon çalışacak
W = 0.7              # Atalet ağırlığı — parçacığın mevcut hızına bağlılığı
                     # Büyük W → daha fazla keşif, küçük W → daha fazla sömürü
C1 = 1.5             # Bilişsel katsayı — parçacığın kendi en iyisine çekimi
                     # "Kendi geçmişinden ne kadar öğreniyor?"
C2 = 1.5             # Sosyal katsayı — sürünün en iyisine çekimi
                     # "Diğer parçacıklardan ne kadar öğreniyor?"

# --- ARAMA UZAYI SINIRLARI ---
# PSO bu aralıklar içinde arama yapacak
HIZ_MIN, HIZ_MAX       = 4.0, 12.0    # m/s
IRTIFA_MIN, IRTIFA_MAX = 25.0, 100.0  # metre

print(f"Sürü büyüklüğü: {N_PARCACIK} parçacık")
print(f"İterasyon sayısı: {N_ITERASYON}")
print(f"Atalet ağırlığı (w): {W}")
print(f"Bilişsel katsayı (c1): {C1}")
print(f"Sosyal katsayı (c2): {C2}")
print(f"Hız aralığı: {HIZ_MIN} - {HIZ_MAX} m/s")
print(f"İrtifa aralığı: {IRTIFA_MIN} - {IRTIFA_MAX} m")

# ============================================================
# FITNESS (UYGUNLUK) FONKSİYONU
# ============================================================
def fitness(hiz, irtifa):
    """
    PSO her parçacık için bunu hesaplıyor.
    
    Düşük fitness = iyi çözüm
    Yüksek fitness = kötü çözüm
    
    PSO bunu minimize etmeye çalışıyor.
    """
    # Güç tüketimini tahmin et
    guc = guc_tahmin(hiz, YUK, irtifa)
    
    # Bu hızda mesafeyi uçmak ne kadar sürer?
    sure = MESAFE / hiz  # saniye
    
    # Toplam enerji = Güç × Süre
    toplam_enerji = guc * sure  # Joule
    
    return toplam_enerji

# Test edelim
print(f"\nFitness testi:")
print(f"  4 m/s, 25m  → {fitness(4, 25):.0f} J")
print(f"  8 m/s, 50m  → {fitness(8, 50):.0f} J")
print(f"  12 m/s, 100m → {fitness(12, 100):.0f} J")

# ============================================================
# PSO ALGORİTMASI
# ============================================================
print("\nPSO başlatılıyor...")
np.random.seed(42)  # Tekrarlanabilir sonuçlar için

# --- BAŞLANGIÇ KONUMLARI ---
# 50 parçacığın başlangıç hız ve irtifaları rastgele belirlenir
pozisyon = np.zeros((N_PARCACIK, 2))
pozisyon[:, 0] = np.random.uniform(HIZ_MIN, HIZ_MAX, N_PARCACIK)       # hız
pozisyon[:, 1] = np.random.uniform(IRTIFA_MIN, IRTIFA_MAX, N_PARCACIK) # irtifa

# --- BAŞLANGIÇ HIZLARI ---
# Parçacıkların hareket hızları (pozisyon değişim miktarı)
# Bu uçuş hızıyla karıştırılmamalı — PSO'nun iç parametresi
hiz = np.zeros((N_PARCACIK, 2))
hiz[:, 0] = np.random.uniform(-2, 2, N_PARCACIK)   # hız boyutunda hareket
hiz[:, 1] = np.random.uniform(-10, 10, N_PARCACIK) # irtifa boyutunda hareket

# --- KİŞİSEL EN İYİ (pBest) ---
# Her parçacığın şimdiye kadar bulduğu en iyi konum
pBest_pozisyon = pozisyon.copy()
pBest_fitness  = np.array([fitness(p[0], p[1]) for p in pozisyon])

# --- GLOBAL EN İYİ (gBest) ---
# Tüm sürünün şimdiye kadar bulduğu en iyi konum
gBest_idx      = np.argmin(pBest_fitness)
gBest_pozisyon = pBest_pozisyon[gBest_idx].copy()
gBest_fitness  = pBest_fitness[gBest_idx]

# --- TAKİP DEĞİŞKENLERİ ---
# Her iterasyondaki en iyi fitness değerini kaydet
gecmis_gBest    = []  # global en iyi geçmişi
gecmis_ort      = []  # ortalama fitness geçmişi
tum_pozisyonlar = []  # animasyon için tüm konumlar

print(f"Başlangıç gBest fitness: {gBest_fitness:.0f} J")
print(f"Başlangıç gBest hız: {gBest_pozisyon[0]:.2f} m/s")
print(f"Başlangıç gBest irtifa: {gBest_pozisyon[1]:.2f} m")

# ============================================================
# ANA PSO DÖNGÜSÜ
# ============================================================
print("\nOptimizasyon başlıyor...")

for iterasyon in range(N_ITERASYON):

    # Her parçacık için güncelle
    for i in range(N_PARCACIK):

        # Rastgele sayılar üret (her parçacık için farklı)
        r1 = np.random.random(2)  # bilişsel için rastgele
        r2 = np.random.random(2)  # sosyal için rastgele

        # -----------------------------------------------
        # HIZ GÜNCELLEME DENKLEMİ
        # v(t+1) = w*v(t) + c1*r1*(pBest-x) + c2*r2*(gBest-x)
        # -----------------------------------------------
        atalet   = W * hiz[i]
        # Parçacığı kendi en iyisine çek
        bilissel = C1 * r1 * (pBest_pozisyon[i] - pozisyon[i])
        # Parçacığı sürünün en iyisine çek
        sosyal   = C2 * r2 * (gBest_pozisyon - pozisyon[i])

        # Yeni hız = atalet + bilişsel çekim + sosyal çekim
        hiz[i] = atalet + bilissel + sosyal

        # Hız sınırlandır (çok hızlı hareket etmesin)
        hiz[i, 0] = np.clip(hiz[i, 0], -3, 3)
        hiz[i, 1] = np.clip(hiz[i, 1], -15, 15)

        # -----------------------------------------------
        # POZİSYON GÜNCELLEME
        # x(t+1) = x(t) + v(t+1)
        # -----------------------------------------------
        pozisyon[i] += hiz[i]

        # Sınır dışına çıkarsa geri getir
        pozisyon[i, 0] = np.clip(pozisyon[i, 0], HIZ_MIN, HIZ_MAX)
        pozisyon[i, 1] = np.clip(pozisyon[i, 1], IRTIFA_MIN, IRTIFA_MAX)

        # -----------------------------------------------
        # FITNESS HESAPLA VE EN İYİLERİ GÜNCELLE
        # -----------------------------------------------
        yeni_fitness = fitness(pozisyon[i, 0], pozisyon[i, 1])

        # Kişisel en iyi güncelle
        if yeni_fitness < pBest_fitness[i]:
            pBest_fitness[i]     = yeni_fitness
            pBest_pozisyon[i]    = pozisyon[i].copy()

        # Global en iyi güncelle
        if yeni_fitness < gBest_fitness:
            gBest_fitness        = yeni_fitness
            gBest_pozisyon       = pozisyon[i].copy()

    # İterasyon sonuçlarını kaydet
    gecmis_gBest.append(gBest_fitness)
    gecmis_ort.append(pBest_fitness.mean())
    tum_pozisyonlar.append(pozisyon.copy())

    # Her 50 iterasyonda bir ekrana yaz
    if (iterasyon + 1) % 50 == 0:
        print(f"İterasyon {iterasyon+1:3d} | "
              f"En İyi: {gBest_fitness:.0f} J | "
              f"Hız: {gBest_pozisyon[0]:.2f} m/s | "
              f"İrtifa: {gBest_pozisyon[1]:.2f} m")

print(f"\nOptimizasyon tamamlandı!")
print(f"Optimum Hız:    {gBest_pozisyon[0]:.2f} m/s")
print(f"Optimum İrtifa: {gBest_pozisyon[1]:.2f} m")
print(f"Minimum Enerji: {gBest_fitness:.0f} J")
print(f"Minimum Güç:    {guc_tahmin(gBest_pozisyon[0], YUK, gBest_pozisyon[1]):.1f} W")

# ============================================================
# GRAFİK 1 — YAKINSAMA GRAFİĞİ
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('PSO Optimizasyonu — Yakınsama Analizi',
             fontsize=15, fontweight='bold', color='white', y=0.98)

iterasyonlar = np.arange(1, N_ITERASYON + 1)

# --- GRAFIK 1A: ENERJİ YAKINSAMA ---
ax = axes[0]

ax.plot(iterasyonlar, gecmis_gBest,
        color=COLORS['cyan'], linewidth=2.5,
        label='Global En İyi (gBest)', zorder=3)

ax.plot(iterasyonlar, gecmis_ort,
        color=COLORS['orange'], linewidth=1.5,
        linestyle='--', alpha=0.8,
        label='Sürü Ortalaması', zorder=2)

# Optimum noktayı işaretle
opt_iter = np.argmin(gecmis_gBest)
ax.scatter(opt_iter + 1, gecmis_gBest[opt_iter],
           color=COLORS['yellow'], s=200,
           zorder=5, marker='*',
           label=f'Optimum: {gecmis_gBest[opt_iter]:.0f} J')

# Optimum çizgisi
ax.axhline(gBest_fitness, color=COLORS['green'],
           linewidth=1.5, linestyle=':',
           alpha=0.7, label=f'Min Enerji: {gBest_fitness:.0f} J')

# Gölge alan — iyileşme bölgesi
ax.fill_between(iterasyonlar,
                gecmis_gBest, gecmis_ort,
                alpha=0.1, color=COLORS['cyan'])

ax.set_xlabel('İterasyon Sayısı', fontsize=12)
ax.set_ylabel('Toplam Enerji Tüketimi (J)', fontsize=12)
ax.set_title('PSO Yakınsama Eğrisi\n(Enerji Minimizasyonu)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Sonuç kutucuğu
sonuc_text = (f'Optimum Sonuç:\n'
              f'Hız: {gBest_pozisyon[0]:.2f} m/s\n'
              f'İrtifa: {gBest_pozisyon[1]:.2f} m\n'
              f'Min Enerji: {gBest_fitness:.0f} J\n'
              f'Min Güç: {guc_tahmin(gBest_pozisyon[0], YUK, gBest_pozisyon[1]):.1f} W')
ax.text(0.97, 0.95, sonuc_text,
        transform=ax.transAxes,
        fontsize=9, color='white',
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round',
                  facecolor='#222222', alpha=0.9))

# --- GRAFIK 1B: İYİLEŞME ORANI ---
ax = axes[1]

# Her iterasyondaki iyileşme yüzdesi
iyilesme = []
for i in range(1, len(gecmis_gBest)):
    yuzde = ((gecmis_gBest[i-1] - gecmis_gBest[i]) /
              gecmis_gBest[i-1]) * 100
    iyilesme.append(yuzde)

ax.bar(iterasyonlar[1:], iyilesme,
       color=COLORS['green'], alpha=0.6,
       edgecolor='none', width=1.0)

# Kümülatif iyileşme
kumulatif = ((gecmis_gBest[0] - np.array(gecmis_gBest[1:])) /
              gecmis_gBest[0]) * 100
ax2 = ax.twinx()
ax2.plot(iterasyonlar[1:], kumulatif,
         color=COLORS['yellow'], linewidth=2.5,
         label='Kümülatif İyileşme (%)')
ax2.set_ylabel('Kümülatif İyileşme (%)',
               fontsize=11, color=COLORS['yellow'])
ax2.tick_params(colors=COLORS['yellow'])
ax2.yaxis.label.set_color(COLORS['yellow'])

ax.set_xlabel('İterasyon Sayısı', fontsize=12)
ax.set_ylabel('Anlık İyileşme (%)', fontsize=12)
ax.set_title('PSO İyileşme Oranı\n(Her İterasyondaki Kazanım)',
             fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Toplam iyileşme
toplam_iyilesme = ((gecmis_gBest[0] - gBest_fitness) /
                    gecmis_gBest[0]) * 100
ax.text(0.97, 0.95,
        f'Toplam İyileşme:\n%{toplam_iyilesme:.1f}',
        transform=ax.transAxes,
        fontsize=11, color=COLORS['yellow'],
        verticalalignment='top',
        horizontalalignment='right',
        fontweight='bold',
        bbox=dict(boxstyle='round',
                  facecolor='#222222', alpha=0.9))

plt.tight_layout()
plt.savefig('grafik_pso1_yakinsama.png', dpi=150,
            bbox_inches='tight', facecolor='#0a0a0a')
plt.show()
print("Grafik 1 kaydedildi!")

# ============================================================
# GRAFİK 2 — PARCACIK HAREKETİ VE ARAMA UZAYI
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('PSO — Parçacık Hareketi ve Arama Uzayı',
             fontsize=15, fontweight='bold', color='white', y=0.98)

# --- GRAFIK 2A: ARAMA UZAYI ISI HARİTASI ---
ax = axes[0]

# Tüm arama uzayında fitness değerlerini hesapla
hiz_aralik = np.linspace(HIZ_MIN, HIZ_MAX, 50)
irt_aralik = np.linspace(IRTIFA_MIN, IRTIFA_MAX, 50)
Hiz_grid, Irt_grid = np.meshgrid(hiz_aralik, irt_aralik)

Fitness_grid = np.zeros_like(Hiz_grid)
for i in range(Hiz_grid.shape[0]):
    for j in range(Hiz_grid.shape[1]):
        Fitness_grid[i,j] = fitness(Hiz_grid[i,j], Irt_grid[i,j])

# Isı haritası olarak çiz
im = ax.contourf(Hiz_grid, Irt_grid, Fitness_grid,
                 levels=30, cmap=custom_cmap, alpha=0.9)
ax.contour(Hiz_grid, Irt_grid, Fitness_grid,
           levels=10, colors='white', alpha=0.2,
           linewidths=0.5)

cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label('Toplam Enerji (J)', color='white', fontsize=11)
cbar.ax.tick_params(colors='white')

# Başlangıç parçacıklarını çiz
baslangic = tum_pozisyonlar[0]
ax.scatter(baslangic[:, 0], baslangic[:, 1],
           color=COLORS['white'], alpha=0.5, s=40,
           marker='o', label='Başlangıç', zorder=3)

# Son parçacıkları çiz
son = tum_pozisyonlar[-1]
ax.scatter(son[:, 0], son[:, 1],
           color=COLORS['cyan'], alpha=0.8, s=60,
           marker='o', label='Son Konum', zorder=4)

# Optimum noktayı işaretle
ax.scatter(gBest_pozisyon[0], gBest_pozisyon[1],
           color=COLORS['yellow'], s=300,
           marker='*', zorder=5,
           label=f'Optimum\n({gBest_pozisyon[0]:.1f}m/s, {gBest_pozisyon[1]:.1f}m)',
           edgecolors='white', linewidth=1.5)

ax.set_xlabel('Uçuş Hızı (m/s)', fontsize=12)
ax.set_ylabel('İrtifa (metre)', fontsize=12)
ax.set_title('Fitness Arama Uzayı\n(Koyu = Düşük Enerji = İyi)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.2)

# --- GRAFIK 2B: PARCACIK YOLCULUKLARI ---
ax = axes[1]

# Arka plan fitness haritası
ax.contourf(Hiz_grid, Irt_grid, Fitness_grid,
            levels=30, cmap=custom_cmap, alpha=0.7)

# 10 rastgele parçacığın yolculuğunu çiz
secilen_parcaciklar = np.random.choice(N_PARCACIK, 10, replace=False)

for pid in secilen_parcaciklar:
    yolculuk_hiz = [tum_pozisyonlar[it][pid, 0]
                    for it in range(N_ITERASYON)]
    yolculuk_irt = [tum_pozisyonlar[it][pid, 1]
                    for it in range(N_ITERASYON)]

    # Yolu çiz
    ax.plot(yolculuk_hiz, yolculuk_irt,
            alpha=0.4, linewidth=1,
            color=COLORS['white'])

    # Başlangıç noktası
    ax.scatter(yolculuk_hiz[0], yolculuk_irt[0],
               color=COLORS['white'], s=30,
               alpha=0.6, zorder=3)

    # Bitiş noktası
    ax.scatter(yolculuk_hiz[-1], yolculuk_irt[-1],
               color=COLORS['cyan'], s=50,
               alpha=0.8, zorder=4)

# Optimum noktayı işaretle
ax.scatter(gBest_pozisyon[0], gBest_pozisyon[1],
           color=COLORS['yellow'], s=400,
           marker='*', zorder=5,
           edgecolors='white', linewidth=2,
           label=f'Optimum Nokta\n({gBest_pozisyon[0]:.1f}m/s, {gBest_pozisyon[1]:.1f}m)')

ax.set_xlabel('Uçuş Hızı (m/s)', fontsize=12)
ax.set_ylabel('İrtifa (metre)', fontsize=12)
ax.set_title('Parçacık Yolculukları\n(10 Örnek Parçacık)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('grafik_pso2_parcaciklar.png', dpi=150,
            bbox_inches='tight', facecolor='#0a0a0a')
plt.show()
print("Grafik 2 kaydedildi!")

# ============================================================
# GRAFİK 3 — OPTİMİZASYON ÖNCESİ vs SONRASI KARŞILAŞTIRMA
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle('PSO Optimizasyonu — Öncesi vs Sonrası Karşılaştırma',
             fontsize=15, fontweight='bold', color='white', y=0.98)

# --- SENARYOLAR ---
# Baseline: sabit hız ve irtifada uçuş (PSO öncesi)
baseline_hiz    = 8.0   # m/s — tipik sabit hız
baseline_irtifa = 50.0  # m   — tipik sabit irtifa

# Optimum: PSO'nun bulduğu en iyi değerler
opt_hiz    = gBest_pozisyon[0]
opt_irtifa = gBest_pozisyon[1]

# Güç ve enerji hesapla
baseline_guc    = guc_tahmin(baseline_hiz, YUK, baseline_irtifa)
opt_guc         = guc_tahmin(opt_hiz, YUK, opt_irtifa)

baseline_sure   = MESAFE / baseline_hiz
opt_sure        = MESAFE / opt_hiz

baseline_enerji = baseline_guc * baseline_sure
opt_enerji      = opt_guc * opt_sure

# İyileşme yüzdeleri
guc_iyilesme    = ((baseline_guc - opt_guc) / baseline_guc) * 100
enerji_iyilesme = ((baseline_enerji - opt_enerji) / baseline_enerji) * 100
sure_fark       = baseline_sure - opt_sure

print(f"\nKarşılaştırma:")
print(f"  Baseline → Güç: {baseline_guc:.1f}W | "
      f"Süre: {baseline_sure:.1f}s | Enerji: {baseline_enerji:.0f}J")
print(f"  Optimum  → Güç: {opt_guc:.1f}W | "
      f"Süre: {opt_sure:.1f}s | Enerji: {opt_enerji:.0f}J")
print(f"  Enerji tasarrufu: %{enerji_iyilesme:.1f}")

# --- GRAFIK 3A: GÜÇ KARŞILAŞTIRMASI ---
ax = axes[0]

kategoriler = ['Baseline\n(Sabit 8m/s)', 'PSO\nOptimum']
guc_deger   = [baseline_guc, opt_guc]
renkler     = [COLORS['orange'], COLORS['green']]

bars = ax.bar(kategoriler, guc_deger,
              color=renkler, edgecolor='white',
              linewidth=1, alpha=0.85, width=0.5)

# Değerleri yaz
for bar, val in zip(bars, guc_deger):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 2,
            f'{val:.1f}W',
            ha='center', va='bottom',
            fontsize=13, fontweight='bold',
            color='white')

# İyileşme oku
ax.annotate('',
            xy=(1, opt_guc + 10),
            xytext=(0, baseline_guc + 10),
            arrowprops=dict(arrowstyle='<->',
                           color=COLORS['yellow'],
                           lw=2))
ax.text(0.5, (baseline_guc + opt_guc)/2 + 15,
        f'%{guc_iyilesme:.1f}\nİyileşme',
        ha='center', fontsize=11,
        color=COLORS['yellow'], fontweight='bold')

ax.set_ylabel('Güç Tüketimi (W)', fontsize=12)
ax.set_title('Güç Tüketimi\nKarşılaştırması', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, max(guc_deger) * 1.3)

# --- GRAFIK 3B: ENERJİ KARŞILAŞTIRMASI ---
ax = axes[1]

enerji_deger = [baseline_enerji/1000, opt_enerji/1000]  # kJ

bars2 = ax.bar(kategoriler, enerji_deger,
               color=renkler, edgecolor='white',
               linewidth=1, alpha=0.85, width=0.5)

for bar, val in zip(bars2, enerji_deger):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.5,
            f'{val:.1f}kJ',
            ha='center', va='bottom',
            fontsize=13, fontweight='bold',
            color='white')

ax.annotate('',
            xy=(1, enerji_deger[1] + 2),
            xytext=(0, enerji_deger[0] + 2),
            arrowprops=dict(arrowstyle='<->',
                           color=COLORS['yellow'],
                           lw=2))
ax.text(0.5, (enerji_deger[0] + enerji_deger[1])/2 + 3,
        f'%{enerji_iyilesme:.1f}\nTasarruf',
        ha='center', fontsize=11,
        color=COLORS['yellow'], fontweight='bold')

ax.set_ylabel('Toplam Enerji (kJ)', fontsize=12)
ax.set_title('Toplam Enerji Tüketimi\nKarşılaştırması',
             fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, max(enerji_deger) * 1.3)

# --- GRAFIK 3C: ÖZET TABLO ---
ax = axes[2]
ax.axis('off')

tablo_data = [
    ['Parametre', 'Baseline', 'PSO Optimum', 'Fark'],
    ['Hız (m/s)',
     f'{baseline_hiz:.1f}',
     f'{opt_hiz:.2f}',
     f'{opt_hiz - baseline_hiz:+.2f}'],
    ['İrtifa (m)',
     f'{baseline_irtifa:.0f}',
     f'{opt_irtifa:.2f}',
     f'{opt_irtifa - baseline_irtifa:+.2f}'],
    ['Güç (W)',
     f'{baseline_guc:.1f}',
     f'{opt_guc:.1f}',
     f'{opt_guc - baseline_guc:+.1f}'],
    ['Süre (s)',
     f'{baseline_sure:.1f}',
     f'{opt_sure:.1f}',
     f'{opt_sure - baseline_sure:+.1f}'],
    ['Enerji (kJ)',
     f'{baseline_enerji/1000:.2f}',
     f'{opt_enerji/1000:.2f}',
     f'{(opt_enerji-baseline_enerji)/1000:+.2f}'],
    ['Tasarruf (%)',
     '—',
     '—',
     f'%{enerji_iyilesme:.1f}'],
]

tablo = ax.table(
    cellText=tablo_data[1:],
    colLabels=tablo_data[0],
    cellLoc='center',
    loc='center',
    colWidths=[0.3, 0.2, 0.25, 0.25]
)

tablo.auto_set_font_size(False)
tablo.set_fontsize(10)
tablo.scale(1, 2.0)

for (row, col), cell in tablo.get_celld().items():
    cell.set_facecolor('#1a1a1a')
    cell.set_edgecolor('#333333')
    cell.set_text_props(color='white')
    if row == 0:
        cell.set_facecolor('#00d4ff')
        cell.set_text_props(color='#111111', fontweight='bold')
    elif row % 2 == 0:
        cell.set_facecolor('#111111')
    # Son satır — tasarruf yüzdesi — yeşil yap
    if row == len(tablo_data) - 1:
        cell.set_facecolor('#003322')
        cell.set_text_props(color=COLORS['green'],
                           fontweight='bold')

ax.set_title('Detaylı Karşılaştırma Tablosu',
             fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('grafik_pso3_karsilastirma.png', dpi=150,
            bbox_inches='tight', facecolor='#0a0a0a')
plt.show()
print("Grafik 3 kaydedildi!")

# ============================================================
# GRAFİK 4 — PSO ÖZET PANELİ
# ============================================================
fig = plt.figure(figsize=(20, 12))
fig.patch.set_facecolor('#0a0a0a')
fig.suptitle('PSO İHA Enerji Optimizasyonu — Tam Analiz Paneli',
             fontsize=18, fontweight='bold', color='white', y=0.98)

gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)

# --- PANEL 1: YAKINSAMA ---
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(iterasyonlar, gecmis_gBest,
         color=COLORS['cyan'], linewidth=2.5,
         label='gBest')
ax1.plot(iterasyonlar, gecmis_ort,
         color=COLORS['orange'], linewidth=1.5,
         linestyle='--', alpha=0.8, label='Ortalama')
ax1.fill_between(iterasyonlar, gecmis_gBest, gecmis_ort,
                 alpha=0.1, color=COLORS['cyan'])
ax1.set_xlabel('İterasyon', fontsize=10)
ax1.set_ylabel('Enerji (J)', fontsize=10)
ax1.set_title('Yakınsama Eğrisi', fontsize=11, fontweight='bold')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# --- PANEL 2: ARAMA UZAYI ---
ax2 = fig.add_subplot(gs[0, 1])
im = ax2.contourf(Hiz_grid, Irt_grid, Fitness_grid,
                  levels=30, cmap=custom_cmap, alpha=0.9)
ax2.contour(Hiz_grid, Irt_grid, Fitness_grid,
            levels=10, colors='white', alpha=0.2, linewidths=0.5)
son_pozisyon = tum_pozisyonlar[-1]
ax2.scatter(son_pozisyon[:, 0], son_pozisyon[:, 1],
            color=COLORS['cyan'], s=30, alpha=0.6, zorder=3)
ax2.scatter(gBest_pozisyon[0], gBest_pozisyon[1],
            color=COLORS['yellow'], s=250, marker='*',
            zorder=5, edgecolors='white', linewidth=1.5)
ax2.set_xlabel('Hız (m/s)', fontsize=10)
ax2.set_ylabel('İrtifa (m)', fontsize=10)
ax2.set_title('Arama Uzayı', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.2)

# --- PANEL 3: GÜÇ EĞRİSİ ---
ax3 = fig.add_subplot(gs[0, 2])
hiz_test = np.linspace(HIZ_MIN, HIZ_MAX, 100)
guc_test  = [guc_tahmin(v, YUK, opt_irtifa) for v in hiz_test]
ax3.plot(hiz_test, guc_test,
         color=COLORS['cyan'], linewidth=2.5)
ax3.axvline(opt_hiz, color=COLORS['yellow'],
            linewidth=2, linestyle='--',
            label=f'Optimum: {opt_hiz:.2f} m/s')
ax3.scatter(opt_hiz, opt_guc,
            color=COLORS['yellow'], s=200,
            marker='*', zorder=5,
            edgecolors='white', linewidth=1.5)
ax3.set_xlabel('Hız (m/s)', fontsize=10)
ax3.set_ylabel('Güç (W)', fontsize=10)
ax3.set_title('Hız-Güç Eğrisi\n(Optimum İrtifada)',
              fontsize=11, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# --- PANEL 4: ENERJİ KARŞILAŞTIRMASI ---
ax4 = fig.add_subplot(gs[1, 0])
kategoriler = ['Baseline', 'PSO Optimum']
enerji_deger = [baseline_enerji/1000, opt_enerji/1000]
renkler_bar  = [COLORS['orange'], COLORS['green']]
bars = ax4.bar(kategoriler, enerji_deger,
               color=renkler_bar, edgecolor='white',
               linewidth=1, alpha=0.85, width=0.4)
for bar, val in zip(bars, enerji_deger):
    ax4.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.3,
             f'{val:.1f}kJ',
             ha='center', fontsize=11,
             fontweight='bold', color='white')
ax4.set_ylabel('Toplam Enerji (kJ)', fontsize=10)
ax4.set_title('Enerji Karşılaştırması', fontsize=11, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')
ax4.set_ylim(0, max(enerji_deger) * 1.3)
ax4.text(0.5, 0.85,
         f'%{enerji_iyilesme:.1f} Tasarruf',
         transform=ax4.transAxes,
         ha='center', fontsize=12,
         color=COLORS['green'], fontweight='bold',
         bbox=dict(boxstyle='round',
                   facecolor='#003322', alpha=0.8))

# --- PANEL 5: PARCACIK YOLCULUKLARI ---
ax5 = fig.add_subplot(gs[1, 1])
ax5.contourf(Hiz_grid, Irt_grid, Fitness_grid,
             levels=30, cmap=custom_cmap, alpha=0.7)
for pid in secilen_parcaciklar[:8]:
    yolculuk_hiz = [tum_pozisyonlar[it][pid, 0]
                    for it in range(N_ITERASYON)]
    yolculuk_irt = [tum_pozisyonlar[it][pid, 1]
                    for it in range(N_ITERASYON)]
    ax5.plot(yolculuk_hiz, yolculuk_irt,
             alpha=0.4, linewidth=1,
             color=COLORS['white'])
    ax5.scatter(yolculuk_hiz[-1], yolculuk_irt[-1],
                color=COLORS['cyan'], s=40,
                alpha=0.8, zorder=4)
ax5.scatter(gBest_pozisyon[0], gBest_pozisyon[1],
            color=COLORS['yellow'], s=300,
            marker='*', zorder=5,
            edgecolors='white', linewidth=1.5)
ax5.set_xlabel('Hız (m/s)', fontsize=10)
ax5.set_ylabel('İrtifa (m)', fontsize=10)
ax5.set_title('Parçacık Yolculukları', fontsize=11, fontweight='bold')
ax5.grid(True, alpha=0.2)

# --- PANEL 6: SONUÇ KARTI ---
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')

sonuc_bilgi = [
    ('🎯 OPTİMUM SONUÇLAR', '', True),
    ('', '', False),
    ('Optimum Hız', f'{opt_hiz:.2f} m/s', False),
    ('Optimum İrtifa', f'{opt_irtifa:.2f} m', False),
    ('Min. Güç', f'{opt_guc:.1f} W', False),
    ('Min. Enerji', f'{opt_enerji/1000:.2f} kJ', False),
    ('', '', False),
    ('📊 PSO PARAMETRELERİ', '', True),
    ('', '', False),
    ('Parçacık Sayısı', f'{N_PARCACIK}', False),
    ('İterasyon', f'{N_ITERASYON}', False),
    ('Atalet (w)', f'{W}', False),
    ('Bilişsel (c1)', f'{C1}', False),
    ('Sosyal (c2)', f'{C2}', False),
    ('', '', False),
    ('⚡ TASARRUF', '', True),
    ('', '', False),
    ('Güç Tasarrufu', f'%{guc_iyilesme:.1f}', False),
    ('Enerji Tasarrufu', f'%{enerji_iyilesme:.1f}', False),
]

y_pos = 0.98
for label, deger, baslik in sonuc_bilgi:
    if baslik:
        ax6.text(0.5, y_pos, label,
                 transform=ax6.transAxes,
                 fontsize=11, fontweight='bold',
                 color=COLORS['yellow'],
                 ha='center', va='top')
    elif label == '':
        pass
    else:
        ax6.text(0.05, y_pos, label,
                 transform=ax6.transAxes,
                 fontsize=10, color='#aaaaaa',
                 ha='left', va='top')
        ax6.text(0.95, y_pos, deger,
                 transform=ax6.transAxes,
                 fontsize=10, color='white',
                 fontweight='bold',
                 ha='right', va='top')
    y_pos -= 0.05

# Çerçeve ekle
for spine in ax6.spines.values():
    spine.set_edgecolor('#333333')

plt.savefig('grafik_pso4_ozet_panel.png', dpi=150,
            bbox_inches='tight', facecolor='#0a0a0a')
plt.show()
print("Grafik 4 kaydedildi!")

# ============================================================
# TAMAMLANDI
# ============================================================
print("\n" + "="*55)
print("PSO OPTİMİZASYONU BAŞARIYLA TAMAMLANDI!")
print("="*55)
print(f"  Optimum Hız:      {opt_hiz:.2f} m/s")
print(f"  Optimum İrtifa:   {opt_irtifa:.2f} m")
print(f"  Minimum Güç:      {opt_guc:.1f} W")
print(f"  Minimum Enerji:   {opt_enerji/1000:.2f} kJ")
print(f"  Enerji Tasarrufu: %{enerji_iyilesme:.1f}")
print("="*55)
print("\nKaydedilen grafikler:")
print("  grafik_pso1_yakinsama.png")
print("  grafik_pso2_parcaciklar.png")
print("  grafik_pso3_karsilastirma.png")
print("  grafik_pso4_ozet_panel.png")
print("="*55)