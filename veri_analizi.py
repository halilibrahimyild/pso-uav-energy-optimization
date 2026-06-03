import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

# Veriyi oku
print("Veri yükleniyor...")
df = pd.read_csv('flights.csv', low_memory=False)

# Sadece tam uçuşları al (R1-R7), yer testleri hariç
df = df[df['route'].str.startswith('R', na=False)].copy()

# Sütunları sayısal formata çevir
for col in ['speed', 'payload', 'altitude', 'battery_voltage', 'battery_current']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Eksik değerleri temizle
df = df.dropna(subset=['speed', 'payload', 'altitude', 'battery_voltage', 'battery_current'])

# GÜÇÜ HESAPLA: P = V × I
df['power'] = df['battery_voltage'] * df['battery_current'].abs()

print(f"Toplam satır: {len(df)}")
print(f"Toplam uçuş: {df['flight'].nunique()}")
print("Veri hazır!")

# Her uçuş için ortalama cruise (seyir) gücünü hesapla
print("Cruise güçleri hesaplanıyor...")

cruise_data = []

for fid, group in df.groupby('flight'):
    n = len(group)
    if n < 50:  # Çok kısa uçuşları atla
        continue
    
    # Uçuşun ortasındaki %60'lık kısmı al (kalkış ve iniş hariç)
    # Kalkış ve iniş fazları güç tüketimini çarpıtır
    cruise = group.iloc[int(n*0.2):int(n*0.8)]
    
    cruise_data.append({
        'flight':     fid,
        'speed':      group['speed'].iloc[0],        # Komut hızı (m/s)
        'payload':    group['payload'].iloc[0],      # Yük ağırlığı (gram)
        'altitude':   group['altitude'].iloc[0],     # İrtifa (metre)
        'avg_power':  cruise['power'].mean(),        # Ortalama güç (Watt)
        'wind_speed': cruise['wind_speed'].mean(),   # Ortalama rüzgar (m/s)
        'duration':   group['time'].max(),           # Uçuş süresi (saniye)
        'total_energy': np.trapezoid(               # Toplam enerji (Joule)
            group['power'], group['time']
        )
    })

# Listeyi tabloya çevir
cdf = pd.DataFrame(cruise_data)
cdf = cdf.dropna()

print(f"Analiz edilen uçuş sayısı: {len(cdf)}")
print(cdf[['speed','payload','altitude','avg_power']].describe().round(1))

# Grafik genel ayarları
plt.rcParams['figure.facecolor'] = '#0a0a0a'      # Arka plan siyah
plt.rcParams['axes.facecolor'] = '#111111'         # Grafik alanı koyu
plt.rcParams['axes.edgecolor'] = '#333333'         # Eksen çerçevesi
plt.rcParams['axes.labelcolor'] = '#ffffff'        # Eksen yazıları beyaz
plt.rcParams['xtick.color'] = '#aaaaaa'            # X ekseni rakamları
plt.rcParams['ytick.color'] = '#aaaaaa'            # Y ekseni rakamları
plt.rcParams['text.color'] = '#ffffff'             # Tüm yazılar beyaz
plt.rcParams['grid.color'] = '#222222'             # Izgara rengi
plt.rcParams['grid.linewidth'] = 0.5              # Izgara kalınlığı
plt.rcParams['font.family'] = 'DejaVu Sans'        # Font
plt.rcParams['font.size'] = 11                     # Font boyutu

# Renk paleti — gradyan
COLORS = {
    'cyan':    '#00d4ff',
    'green':   '#00ff88',
    'orange':  '#ff6b35',
    'purple':  '#b388ff',
    'yellow':  '#ffd700',
    'red':     '#ff4444',
    'pink':    '#ff69b4',
}

# Özel ısı haritası rengi (mavi → yeşil → sarı → kırmızı)
custom_cmap = LinearSegmentedColormap.from_list(
    'pso_cmap',
    ['#001f4d', '#00d4ff', '#00ff88', '#ffd700', '#ff4444']
)

print("Grafik ayarları hazır!")

print("Grafikler çiziliyor...")

# ============================================================
# GRAFİK 1 — Hız, Yük, İrtifa → Ortalama Güç
# ============================================================
fig1, axes = plt.subplots(1, 3, figsize=(18, 6))
fig1.suptitle('DJI Matrice 100 — Uçuş Parametrelerinin Güç Tüketimine Etkisi',
              fontsize=16, fontweight='bold', color='white', y=0.98)

# --- GRAFIK 1A: HIZ → GÜÇ ---
ax = axes[0]
speed_power = cdf.groupby('speed')['avg_power'].agg(['mean', 'std']).reset_index()

ax.plot(speed_power['speed'], speed_power['mean'],
        color=COLORS['cyan'], linewidth=2.5, marker='o',
        markersize=10, markerfacecolor=COLORS['yellow'],
        markeredgecolor='white', markeredgewidth=1.5,
        label='Ortalama Güç', zorder=3)

ax.fill_between(speed_power['speed'],
                speed_power['mean'] - speed_power['std'],
                speed_power['mean'] + speed_power['std'],
                alpha=0.2, color=COLORS['cyan'], label='±1 Std Sapma')

ax.set_xlabel('Uçuş Hızı (m/s)', fontsize=12)
ax.set_ylabel('Ortalama Cruise Gücü (W)', fontsize=12)
ax.set_title('Hız → Güç İlişkisi', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xticks([4, 6, 8, 10, 12])

# Her noktanın üstüne değer yaz
for _, row in speed_power.iterrows():
    ax.annotate(f"{row['mean']:.0f}W",
                xy=(row['speed'], row['mean']),
                xytext=(0, 12), textcoords='offset points',
                ha='center', fontsize=9,
                color=COLORS['yellow'], fontweight='bold')

# --- GRAFIK 1B: YÜK → GÜÇ ---
ax = axes[1]
payload_power = cdf.groupby('payload')['avg_power'].agg(['mean', 'std']).reset_index()

bars = ax.bar(payload_power['payload'], payload_power['mean'],
              width=60, color=[COLORS['green'], COLORS['cyan'],
                               COLORS['orange'], COLORS['purple']],
              edgecolor='white', linewidth=0.8, zorder=3)

ax.errorbar(payload_power['payload'], payload_power['mean'],
            yerr=payload_power['std'],
            fmt='none', color='white', capsize=6,
            capthick=1.5, linewidth=1.5, zorder=4)

ax.set_xlabel('Yük Ağırlığı (gram)', fontsize=12)
ax.set_ylabel('Ortalama Cruise Gücü (W)', fontsize=12)
ax.set_title('Yük → Güç İlişkisi', fontsize=13, fontweight='bold')
ax.set_xticks([0, 250, 500, 750])
ax.grid(True, alpha=0.3, axis='y')

for bar, (_, row) in zip(bars, payload_power.iterrows()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            f"{row['mean']:.0f}W",
            ha='center', va='bottom', fontsize=9,
            color='white', fontweight='bold')

# --- GRAFIK 1C: İRTİFA → GÜÇ ---
ax = axes[2]
alt_power = cdf.groupby('altitude')['avg_power'].agg(['mean', 'std']).reset_index()

ax.plot(alt_power['mean'], alt_power['altitude'],
        color=COLORS['orange'], linewidth=2.5,
        marker='D', markersize=10,
        markerfacecolor=COLORS['pink'],
        markeredgecolor='white', markeredgewidth=1.5,
        zorder=3)

ax.fill_betweenx(alt_power['altitude'],
                 alt_power['mean'] - alt_power['std'],
                 alt_power['mean'] + alt_power['std'],
                 alpha=0.2, color=COLORS['orange'])

ax.set_xlabel('Ortalama Cruise Gücü (W)', fontsize=12)
ax.set_ylabel('İrtifa (metre)', fontsize=12)
ax.set_title('İrtifa → Güç İlişkisi', fontsize=13, fontweight='bold')
ax.set_yticks([25, 50, 75, 100])
ax.grid(True, alpha=0.3)

for _, row in alt_power.iterrows():
    ax.annotate(f"{row['mean']:.0f}W",
                xy=(row['mean'], row['altitude']),
                xytext=(8, 0), textcoords='offset points',
                va='center', fontsize=9,
                color=COLORS['yellow'], fontweight='bold')

plt.tight_layout()
plt.savefig('grafik1_parametre_guc.png', dpi=150,
            bbox_inches='tight', facecolor='#0a0a0a')
plt.show()
print("Grafik 1 kaydedildi!")

# ============================================================
# GRAFİK 2 — Güç Dağılımı ve Uçuş Süresi
# ============================================================
fig2, axes = plt.subplots(1, 2, figsize=(16, 6))
fig2.suptitle('DJI Matrice 100 — Güç ve Süre Dağılım Analizi',
              fontsize=16, fontweight='bold', color='white', y=0.98)

# --- GRAFIK 2A: GÜÇ DAĞILIMI HİSTOGRAMI ---
ax = axes[0]

# Her hız için ayrı histogram
speed_colors = {
    4.0:  COLORS['cyan'],
    6.0:  COLORS['green'],
    8.0:  COLORS['yellow'],
    10.0: COLORS['orange'],
    12.0: COLORS['red']
}

for speed_val, color in speed_colors.items():
    subset = cdf[cdf['speed'] == speed_val]['avg_power']
    ax.hist(subset, bins=15, alpha=0.5, color=color,
            edgecolor='white', linewidth=0.5,
            label=f'{speed_val} m/s')
    # Ortalama çizgisi
    ax.axvline(subset.mean(), color=color,
               linewidth=2, linestyle='--', alpha=0.9)

ax.set_xlabel('Ortalama Cruise Gücü (W)', fontsize=12)
ax.set_ylabel('Uçuş Sayısı', fontsize=12)
ax.set_title('Hıza Göre Güç Dağılımı', fontsize=13, fontweight='bold')
ax.legend(title='Uçuş Hızı', fontsize=9, title_fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# --- GRAFIK 2B: UÇUŞ SÜRESİ DAĞILIMI ---
ax = axes[1]

# Kutu grafik (boxplot) — her hız için
speed_groups = [cdf[cdf['speed'] == s]['duration'].values
                for s in sorted(cdf['speed'].unique())]

bp = ax.boxplot(speed_groups,
                patch_artist=True,
                notch=True,
                medianprops=dict(color='white', linewidth=2.5),
                whiskerprops=dict(color='#aaaaaa', linewidth=1.5),
                capprops=dict(color='#aaaaaa', linewidth=1.5),
                flierprops=dict(marker='o', markerfacecolor=COLORS['red'],
                                markersize=5, alpha=0.6))

# Kutuları renklendir
box_colors = list(speed_colors.values())
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax.set_xticklabels([f'{s} m/s' for s in sorted(cdf['speed'].unique())])
ax.set_xlabel('Uçuş Hızı (m/s)', fontsize=12)
ax.set_ylabel('Uçuş Süresi (saniye)', fontsize=12)
ax.set_title('Hıza Göre Uçuş Süresi Dağılımı', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Medyan değerlerini yaz
for i, group in enumerate(speed_groups):
    median = np.median(group)
    ax.text(i+1, median + 3, f'{median:.0f}s',
            ha='center', va='bottom', fontsize=9,
            color='white', fontweight='bold')

plt.tight_layout()
plt.savefig('grafik2_dagılım.png', dpi=150,
            bbox_inches='tight', facecolor='#0a0a0a')
plt.show()
print("Grafik 2 kaydedildi!")

# ============================================================
# GRAFİK 3 — HIZ × YÜK × GÜÇ ISI HARİTASI
# ============================================================
fig3, axes = plt.subplots(1, 2, figsize=(26, 8))
fig3.suptitle('DJI Matrice 100 — Güç Tüketimi Isı Haritası Analizi',
              fontsize=16, fontweight='bold', color='white', y=0.98)

# --- GRAFIK 3A: HIZ × YÜK → GÜÇ ---
ax = axes[0]

pivot1 = cdf.groupby(['payload', 'speed'])['avg_power'].mean().unstack()

im1 = ax.imshow(pivot1.values, cmap=custom_cmap,
                aspect='auto', interpolation='bilinear')

ax.set_xticks(range(len(pivot1.columns)))
ax.set_xticklabels([f'{s:.0f} m/s' for s in pivot1.columns], fontsize=11)
ax.set_yticks(range(len(pivot1.index)))
ax.set_yticklabels([f'{p:.0f} g' for p in pivot1.index], fontsize=11)
ax.set_xlabel('Uçuş Hızı (m/s)', fontsize=12)
ax.set_ylabel('Yük Ağırlığı (gram)', fontsize=12)
ax.set_title('Hız × Yük → Güç Tüketimi (W)', fontsize=13, fontweight='bold')

for i in range(len(pivot1.index)):
    for j in range(len(pivot1.columns)):
        val = pivot1.values[i, j]
        if not np.isnan(val):
            ax.text(j, i, f'{val:.0f}W',
                    ha='center', va='center',
                    fontsize=10, fontweight='bold',
                    color='white' if val > pivot1.values.mean() else '#111111')

cbar1 = fig3.colorbar(im1, ax=axes[0],
                      shrink=0.8, pad=0.15)
cbar1.set_label('Ortalama Güç (W)', color='white', fontsize=11)
cbar1.ax.tick_params(colors='white')
cbar1.ax.yaxis.label.set_color('white')
cbar1.ax.tick_params(colors='white')

# --- GRAFIK 3B: HIZ × İRTİFA → GÜÇ ---
ax = axes[1]

pivot2 = cdf.groupby(['altitude', 'speed'])['avg_power'].mean().unstack()

im2 = ax.imshow(pivot2.values, cmap=custom_cmap,
                aspect='auto', interpolation='bilinear')

ax.set_xticks(range(len(pivot2.columns)))
ax.set_xticklabels([f'{s:.0f} m/s' for s in pivot2.columns], fontsize=11)
ax.set_yticks(range(len(pivot2.index)))
ax.set_yticklabels([f'{a:.0f} m' for a in pivot2.index], fontsize=11)
ax.set_xlabel('Uçuş Hızı (m/s)', fontsize=12)
ax.set_ylabel('İrtifa (metre)', fontsize=12)
ax.set_title('Hız × İrtifa → Güç Tüketimi (W)', fontsize=13, fontweight='bold')

for i in range(len(pivot2.index)):
    for j in range(len(pivot2.columns)):
        val = pivot2.values[i, j]
        if not np.isnan(val):
            ax.text(j, i, f'{val:.0f}W',
                    ha='center', va='center',
                    fontsize=10, fontweight='bold',
                    color='white' if val > pivot2.values.mean() else '#111111')

cbar2 = fig3.colorbar(im2, ax=axes[1],
                      shrink=0.8, pad=0.15)
cbar2.set_label('Ortalama Güç (W)', color='white', fontsize=11)
cbar2.ax.tick_params(colors='white')
cbar2.ax.yaxis.label.set_color('white')
cbar2.ax.tick_params(colors='white')

plt.tight_layout(pad=3.0)
plt.savefig('grafik3_heatmap.png', dpi=150,
            bbox_inches='tight', facecolor='#0a0a0a')
plt.show()
print("Grafik 3 kaydedildi!")

# ============================================================
# GRAFİK 4 — TOPLAM ENERJİ VE RÜZGAR ETKİSİ
# ============================================================
fig4, axes = plt.subplots(1, 2, figsize=(16, 6))
fig4.suptitle('DJI Matrice 100 — Enerji Tüketimi ve Rüzgar Etkisi Analizi',
              fontsize=16, fontweight='bold', color='white', y=0.98)

# --- GRAFIK 4A: HIZ × YÜK → TOPLAM ENERJİ ---
ax = axes[0]

speed_vals = sorted(cdf['speed'].unique())
payload_vals = sorted(cdf['payload'].unique())
bar_width = 0.18
x = np.arange(len(speed_vals))

payload_colors = [COLORS['cyan'], COLORS['green'],
                  COLORS['orange'], COLORS['purple']]

for i, (payload_val, color) in enumerate(zip(payload_vals, payload_colors)):
    subset = cdf[cdf['payload'] == payload_val]
    energy_by_speed = subset.groupby('speed')['total_energy'].mean() / 1000  # kJ

    offset = (i - len(payload_vals)/2 + 0.5) * bar_width
    bars = ax.bar(x + offset,
                  [energy_by_speed.get(s, 0) for s in speed_vals],
                  width=bar_width, color=color, alpha=0.85,
                  edgecolor='white', linewidth=0.5,
                  label=f'{payload_val:.0f} g')

ax.set_xlabel('Uçuş Hızı (m/s)', fontsize=12)
ax.set_ylabel('Toplam Enerji (kJ)', fontsize=12)
ax.set_title('Hız × Yük → Toplam Enerji Tüketimi', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'{s:.0f} m/s' for s in speed_vals])
ax.legend(title='Yük', fontsize=9, title_fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# --- GRAFIK 4B: RÜZGAR HIZI → GÜÇ SAÇILIM GRAFİĞİ ---
ax = axes[1]

for speed_val, color in speed_colors.items():
    subset = cdf[cdf['speed'] == speed_val]
    ax.scatter(subset['wind_speed'], subset['avg_power'],
               color=color, alpha=0.6, s=60,
               edgecolors='white', linewidth=0.3,
               label=f'{speed_val} m/s', zorder=3)

    # Trend çizgisi ekle
    if len(subset) > 2:
        z = np.polyfit(subset['wind_speed'], subset['avg_power'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(subset['wind_speed'].min(),
                             subset['wind_speed'].max(), 100)
        ax.plot(x_line, p(x_line), color=color,
                linewidth=1.5, linestyle='--', alpha=0.8)

ax.set_xlabel('Rüzgar Hızı (m/s)', fontsize=12)
ax.set_ylabel('Ortalama Cruise Gücü (W)', fontsize=12)
ax.set_title('Rüzgar Hızı → Güç İlişkisi', fontsize=13, fontweight='bold')
ax.legend(title='Uçuş Hızı', fontsize=9, title_fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('grafik4_enerji_ruzgar.png', dpi=150,
            bbox_inches='tight', facecolor='#0a0a0a')
plt.show()
print("Grafik 4 kaydedildi!")

# ============================================================
# GRAFİK 5 — 3D GÜÇ YÜZEYİ
# ============================================================
fig5 = plt.figure(figsize=(16, 7))
fig5.patch.set_facecolor('#0a0a0a')

# --- GRAFIK 5A: HIZ × YÜK → GÜÇ 3D ---
ax1 = fig5.add_subplot(121, projection='3d')
ax1.set_facecolor('#111111')

pivot_3d = cdf.groupby(['speed', 'payload'])['avg_power'].mean().reset_index()

X = pivot_3d['speed'].values
Y = pivot_3d['payload'].values
Z = pivot_3d['avg_power'].values

# 3D yüzey için grid oluştur
from scipy.interpolate import griddata
xi = np.linspace(X.min(), X.max(), 30)
yi = np.linspace(Y.min(), Y.max(), 30)
Xi, Yi = np.meshgrid(xi, yi)
Zi = griddata((X, Y), Z, (Xi, Yi), method='cubic')

surf1 = ax1.plot_surface(Xi, Yi, Zi, cmap=custom_cmap,
                          alpha=0.85, edgecolor='none')
ax1.scatter(X, Y, Z, color=COLORS['yellow'],
            s=50, zorder=5, depthshade=False)

ax1.set_xlabel('Hız (m/s)', fontsize=10, labelpad=8)
ax1.set_ylabel('Yük (gram)', fontsize=10, labelpad=8)
ax1.set_zlabel('Güç (W)', fontsize=10, labelpad=8)
ax1.set_title('Hız × Yük → Güç Yüzeyi', fontsize=12,
              fontweight='bold', pad=15)
ax1.tick_params(colors='#aaaaaa', labelsize=8)
ax1.xaxis.pane.fill = False
ax1.yaxis.pane.fill = False
ax1.zaxis.pane.fill = False
ax1.xaxis.pane.set_edgecolor('#333333')
ax1.yaxis.pane.set_edgecolor('#333333')
ax1.zaxis.pane.set_edgecolor('#333333')
fig5.colorbar(surf1, ax=ax1, shrink=0.5, label='Güç (W)')

# --- GRAFIK 5B: HIZ × İRTİFA → GÜÇ 3D ---
ax2 = fig5.add_subplot(122, projection='3d')
ax2.set_facecolor('#111111')

pivot_3d2 = cdf.groupby(['speed', 'altitude'])['avg_power'].mean().reset_index()

X2 = pivot_3d2['speed'].values
Y2 = pivot_3d2['altitude'].values
Z2 = pivot_3d2['avg_power'].values

xi2 = np.linspace(X2.min(), X2.max(), 30)
yi2 = np.linspace(Y2.min(), Y2.max(), 30)
Xi2, Yi2 = np.meshgrid(xi2, yi2)
Zi2 = griddata((X2, Y2), Z2, (Xi2, Yi2), method='cubic')

surf2 = ax2.plot_surface(Xi2, Yi2, Zi2, cmap=custom_cmap,
                          alpha=0.85, edgecolor='none')
ax2.scatter(X2, Y2, Z2, color=COLORS['yellow'],
            s=50, zorder=5, depthshade=False)

ax2.set_xlabel('Hız (m/s)', fontsize=10, labelpad=8)
ax2.set_ylabel('İrtifa (m)', fontsize=10, labelpad=8)
ax2.set_zlabel('Güç (W)', fontsize=10, labelpad=8)
ax2.set_title('Hız × İrtifa → Güç Yüzeyi', fontsize=12,
              fontweight='bold', pad=15)
ax2.tick_params(colors='#aaaaaa', labelsize=8)
ax2.xaxis.pane.fill = False
ax2.yaxis.pane.fill = False
ax2.zaxis.pane.fill = False
ax2.xaxis.pane.set_edgecolor('#333333')
ax2.yaxis.pane.set_edgecolor('#333333')
ax2.zaxis.pane.set_edgecolor('#333333')
fig5.colorbar(surf2, ax=ax2, shrink=0.5, label='Güç (W)')

fig5.suptitle('DJI Matrice 100 — 3D Güç Yüzeyi Analizi',
              fontsize=16, fontweight='bold', color='white', y=1.01)

plt.tight_layout()
plt.savefig('grafik5_3d_yuzey.png', dpi=150,
            bbox_inches='tight', facecolor='#0a0a0a')
plt.show()
print("Grafik 5 kaydedildi!")

# ============================================================
# TAMAMLANDI
# ============================================================
print("\n" + "="*50)
print("TÜM GRAFİKLER BAŞARIYLA OLUŞTURULDU!")
print("="*50)
print("Kaydedilen dosyalar:")
print("  grafik1_parametre_guc.png")
print("  grafik2_dagılım.png")
print("  grafik3_heatmap.png")
print("  grafik4_enerji_ruzgar.png")
print("  grafik5_3d_yuzey.png")
print("="*50)

