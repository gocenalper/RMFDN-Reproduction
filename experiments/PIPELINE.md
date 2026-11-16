# Research pipeline — nasıl çalışır, neden böyle

Bu depo, LLM destekli deney yönetiminde üç kronik hatayı yapısal olarak
engellemek için kurulmuş bir pipeline taşır: (1) bir deneyde birden fazla
şeyin değişmesi, (2) yıkıcı değişikliklerin onaysız geçmesi, (3) raporlara
elle yazılmış / halüsine sayı girmesi.

## Yaşam döngüsü

```
baseline kaydı (results.json::run — config gömülü)
        │
        ▼
plan dosyası  experiments/plans/E<NNN>_<slug>.json     ← skill: new-experiment
        │  python3 scripts/exp_plan_check.py check <plan>   (tek faktör kapısı)
        ▼
launch (Colab; rclone ile veri, cell içinde de check koşar)
        │  ... training ... → results.json  (+ run_test_eval.py → test-eval JSON)
        ▼
python3 scripts/exp_plan_check.py verify <plan>        (drift kapısı)
        │
        ▼
uv run --with matplotlib scripts/make_run_report.py --plan <plan>
        │        → reports/E<NNN>_<slug>/report.pdf + report_data.json
        │        → reports/INDEX.csv (exp_id başına tek satır)
        ▼                                               ← skill: experiment-report
prose → python3 scripts/check_claims.py <draft> <report_data.json> ...
RESEARCH_LOG.md'ye tek satır
```

## Neden bu tasarım

- **Config dict tek sözleşmedir.** Trainer her run'ın efektif config'ini
  results.json'a gömüyor; validator baseline→plan→gerçekleşen üçlüsünü aynı
  dict-diff ile karşılaştırır. Flag'leri ayrıca modellemeye gerek kalmaz,
  box üzerinde elle yapılan ad-hoc edit'ler `verify`'da yakalanır.
- **Anahtar sınıfları:** `num_workers`, `log_interval`, `skip_test_eval`
  runtime'dır (yok sayılır); `bert_model`/`ast_checkpoint` environment'tır
  (offline path farkı — yalnız uyarı); geri kalan her şey + spec eksenleri
  deneyseldir. `batch_size × grad_accum_steps` çarpımı sabitse bu bir GPU
  repack'idir, faktör değildir.
- **Sayılar yalnız dosyadan akar.** `make_run_report.py` eksik metriği `—`
  basar, asla doldurmaz; PDF altbilgisi kaynak dosyaların sha256'sını taşır.
  `check_claims.py` yazılan her metnin sayılarını bu kaynaklara karşı denetler.
- **Hook'lar guardrail'dir, sandbox değildir.** `.claude/hooks/` Bash ve
  Write/Edit çağrılarını keser: history-rewrite ve AI-trailer'lı commit'ler
  DENY; `git commit/push`, korunan yollara `rm`, canonical dosya edit'leri
  (matrix, config.py, pyproject) ASK; `output/**` metrikleri ve `reports/**`
  DENY (üretici script'i düzelt, yeniden üret). Kararlı bir saldırganı değil,
  tek hamlelik kazaları durdurur.

## Sözleşmeler

- Deney kimliği `E<NNN>_<slug>`; numara tekrarı validator'da FAIL.
- `status: done` bir planı düzenlemek yasak (hook ASK'e düşürür) — kayıt
  yeniden yazılmaz, yeni plan açılır.
- Seed de bir faktördür: tekrar koşusu kendi planını alır.
- Tek seed = nokta tahmini; rapor bu ibareyi otomatik basar, varyans iddiası
  ancak tekrar planlarıyla yazılır.
- Tarihî wandb koşuları `output/analysis/wandb_*.csv` envanterindedir; yeni
  sistem E001'den başlar, eskiye dokunmaz.

## Bilinen sınırlar (dürüst mod)

- `check_claims.py` küçük tamsayıları ve yıl benzeri sayıları atlar
  (`--strict` ile açılır); PASS "her float kaynaklı" demektir, "metin doğru"
  demek değildir.
- Bash guard'ı bileşik/heredoc komutların içine gömülü yıkıcılığı
  yakalayamayabilir; niyetli bypass mümkündür ve görünürdür (transcript).
- `exp_plan_check`'in anahtar sınıfları statiktir; runner'a yeni bir runtime
  anahtarı eklersen `RUNTIME_KEYS`'e de ekle, yoksa yanlış FAIL görürsün
  (bu, sessiz kaçırmaya tercih edilen yön).
