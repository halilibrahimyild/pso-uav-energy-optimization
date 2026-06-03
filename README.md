# pso-uav-energy-optimization
PSO ile İHA enerji optimizasyonu ( UAV Energy Optimization with PSO )— DJI Matrice 100 veri seti

# PSO ile İHA Enerji Optimizasyonu

DJI Matrice 100 quadrotor İHA'nın enerji tüketimini minimize etmek için
Parçacık Sürü Optimizasyonu (PSO) algoritması uygulanmıştır.

## Veri Seti

Rodrigues vd. (2021) tarafından yayımlanan açık erişim veri seti kullanılmıştır.

- 209 gerçek uçuş verisi
- DJI Matrice 100 quadrotor
- Değişkenler: hız, irtifa, yük, batarya voltajı, akım, güç

İndirme linki:
https://kilthub.cmu.edu/articles/dataset/Data_Collected_with_Package_Delivery_Quadcopter_Drone/12683453

## Dosyalar

| Dosya | Açıklama |
|---|---|
| `veri_analizi.py` | Veri seti analizi ve görselleştirme |
| `guc_modeli.py` | Güç tahmin modeli (Fizik + Polinom Regresyon) |
| `pso_optimizasyon.py` | PSO algoritması ve optimizasyon |

## Kurulum
python -m pip install pandas numpy matplotlib scipy scikit-learn

## Kullanım
Sırayla çalıştır:
python veri_analizi.py
python guc_modeli.py
python pso_optimizasyon.py

## Sonuçlar

PSO algoritması 200 iterasyonda optimum uçuş parametrelerini bularak
baseline senaryoya kıyasla enerji tüketimini minimize etmiştir.

## Referans

Rodrigues, T.A. vd. (2021). In-flight positional and energy use data set
of a DJI Matrice 100 quadrotor for small package delivery.
Scientific Data, 8(1), 155.

