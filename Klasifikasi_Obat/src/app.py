import gradio as gr
import joblib
import pandas as pd
import numpy as np

# ==============================================================================
# 1. LOGIKA AI (Backend)
# ==============================================================================
try:
    # Memuat semua komponen dari folder src
    model = joblib.load('src/catboost_model.pkl')
    scaler = joblib.load('src/scaler.pkl')
    le_dict = joblib.load('src/label_encoders.pkl')
    model_loaded = True
    print("✅ Semua model dan encoder berhasil dimuat dari folder src!")
except FileNotFoundError:
    try:
        model = joblib.load('catboost_model.pkl')
        scaler = joblib.load('scaler.pkl')
        le_dict = joblib.load('label_encoders.pkl')
        model_loaded = True
        print("✅ Model ditemukan di folder utama.")
    except FileNotFoundError:
        print("⚠️ File model (.pkl) tidak ditemukan. Pastikan file ada di folder 'src'.")
        model_loaded = False

DAFTAR_OBAT = ['Amlodipine', 'Aspirin', 'Atorvastatin', 'Hydrochlorothiazide', 'Lisinopril']

INFO_OBAT = {
    'Amlodipine': {
        'ikon'  : '💊',
        'kelas' : 'Calcium Channel Blocker',
        'fungsi': 'Mengontrol tekanan darah tinggi dan mencegah angina (nyeri dada) dengan merelaksasi pembuluh darah.',
        'rekomendasi': [
            ('🥗', 'Diet rendah garam',        'Batasi asupan natrium di bawah 1.500 mg/hari. Hindari makanan olahan, keripik, dan makanan kaleng.'),
            ('🏃', 'Olahraga rutin',            'Lakukan jalan kaki, bersepeda, atau renang minimal 30 menit, 5 hari seminggu.'),
            ('🚬', 'Berhenti merokok',          'Nikotin menyempitkan pembuluh darah dan memperburuk hipertensi secara signifikan.'),
            ('😴', 'Tidur cukup',               'Tidur 7–8 jam per malam. Kurang tidur meningkatkan tekanan darah jangka panjang.'),
            ('🧘', 'Kelola stres',              'Meditasi, pernapasan dalam, atau yoga dapat membantu menurunkan tekanan darah.'),
        ],
        'bahaya': [
            'Hindari konsumsi alkohol berlebihan karena dapat memperkuat efek penurun tekanan darah secara berbahaya.',
            'Jangan berhenti minum obat tiba-tiba tanpa konsultasi dokter meski merasa sudah sehat.',
            'Segera hubungi dokter jika mengalami pembengkakan kaki, pusing berat, atau jantung berdebar.',
        ],
    },
    'Aspirin': {
        'ikon'  : '🩸',
        'kelas' : 'Antiplatelet / Analgesik',
        'fungsi': 'Mencegah penggumpalan darah yang dapat menyebabkan serangan jantung dan stroke.',
        'rekomendasi': [
            ('🫐', 'Konsumsi antioksidan',      'Perbanyak buah beri, anggur, brokoli, dan bayam untuk menjaga kesehatan pembuluh darah.'),
            ('🐟', 'Omega-3 dari ikan',         'Konsumsi ikan salmon, tuna, atau sarden 2–3 kali per minggu untuk efek antiplatelet alami.'),
            ('🏃', 'Aktivitas fisik moderat',   'Hindari olahraga benturan keras. Pilih jalan kaki, renang, atau yoga untuk sirkulasi darah.'),
            ('⚖️', 'Jaga berat badan ideal',    'Obesitas meningkatkan risiko penggumpalan darah. Target BMI antara 18,5–24,9.'),
            ('💧', 'Minum air yang cukup',      'Dehidrasi mengentalkan darah. Minum minimal 8 gelas (2 liter) air putih per hari.'),
        ],
        'bahaya': [
            'Aspirin dapat menyebabkan iritasi lambung. Selalu konsumsi bersama makanan atau segelas susu.',
            'Hindari konsumsi ibuprofen atau naproxen bersamaan karena meningkatkan risiko perdarahan internal.',
            'Segera ke dokter jika muncul tinja berwarna hitam, darah dalam urin, atau mimisan yang tidak berhenti.',
        ],
    },
    'Atorvastatin': {
        'ikon'  : '🧬',
        'kelas' : 'Statin (Penurun Kolesterol)',
        'fungsi': 'Menghambat produksi kolesterol LDL di hati dan mengurangi risiko penyakit jantung koroner.',
        'rekomendasi': [
            ('🥑', 'Kurangi lemak jenuh',       'Batasi daging merah, mentega, keju, dan santan. Ganti dengan lemak baik dari alpukat dan kacang-kacangan.'),
            ('🌾', 'Perbanyak serat larut',     'Oatmeal, kacang merah, apel, dan pir terbukti menurunkan kolesterol LDL secara alami.'),
            ('🍟', 'Hindari gorengan',          'Makanan digoreng mengandung lemak trans yang secara drastis meningkatkan kolesterol jahat.'),
            ('🏋️', 'Olahraga aerobik',          'Lakukan kardio minimal 150 menit per minggu untuk meningkatkan kolesterol HDL (baik).'),
            ('🧃', 'Hindari minuman manis',     'Gula berlebih dikonversi menjadi trigliserida. Ganti dengan air putih atau teh hijau tanpa gula.'),
        ],
        'bahaya': [
            'Konsumsi bersamaan dengan jus grapefruit dapat meningkatkan kadar obat dalam darah secara berbahaya.',
            'Segera lapor ke dokter jika merasakan nyeri otot yang tidak biasa, kelemahan, atau urin berwarna gelap.',
            'Jangan hentikan obat sebelum konsultasi dokter meski kadar kolesterol sudah tampak membaik.',
        ],
    },
    'Hydrochlorothiazide': {
        'ikon'  : '💧',
        'kelas' : 'Diuretik Tiazid',
        'fungsi': 'Mengurangi retensi cairan berlebih dalam tubuh untuk menurunkan tekanan darah.',
        'rekomendasi': [
            ('🍌', 'Penuhi kebutuhan kalium',   'Diuretik membuang kalium. Konsumsi pisang, kentang, bayam, dan alpukat untuk menggantinya.'),
            ('🧂', 'Diet sangat rendah garam',  'Target di bawah 1.200 mg natrium/hari. Garam menyebabkan tubuh menahan lebih banyak cairan.'),
            ('☀️', 'Lindungi kulit',            'Obat ini meningkatkan sensitivitas kulit terhadap sinar UV. Gunakan tabir surya SPF 30+ di luar ruangan.'),
            ('💧', 'Minum air secukupnya',      'Konsumsi 6–8 gelas air/hari. Jangan berlebihan karena dapat mengganggu keseimbangan elektrolit.'),
            ('🥦', 'Perbanyak sayuran segar',   'Sayuran kaya mineral membantu menyeimbangkan elektrolit yang terbuang akibat diuretik.'),
        ],
        'bahaya': [
            'Obat ini menyebabkan sering buang air kecil — hindari konsumsi malam hari agar tidak mengganggu kualitas tidur.',
            'Waspada tanda dehidrasi: mulut kering, pusing mendadak, dan kelemahan otot. Segera hubungi dokter jika terjadi.',
            'Beri tahu dokter jika mengonsumsi obat diabetes — diuretik dapat mempengaruhi kadar gula darah.',
        ],
    },
    'Lisinopril': {
        'ikon'  : '❤️',
        'kelas' : 'ACE Inhibitor',
        'fungsi': 'Melindungi jantung dan ginjal dengan menghambat hormon yang menyempitkan pembuluh darah.',
        'rekomendasi': [
            ('🍎', 'Ikuti pola makan DASH',     'Perbanyak buah, sayuran, biji-bijian utuh, dan batasi garam untuk hasil pengobatan optimal.'),
            ('🧂', 'Hindari suplemen kalium',   'ACE inhibitor meningkatkan kalium darah. Hindari suplemen kalium atau garam pengganti tanpa saran dokter.'),
            ('🏃', 'Olahraga teratur',          'Aktivitas aerobik ringan seperti jalan kaki 30 menit/hari membantu mengoptimalkan kerja obat.'),
            ('⚖️', 'Turunkan berat badan',      'Setiap 1 kg penurunan berat badan dapat menurunkan tekanan darah hingga 1 mmHg.'),
            ('🚫', 'Kurangi kafein & alkohol',  'Keduanya meningkatkan tekanan darah dan mengurangi efektivitas Lisinopril secara signifikan.'),
        ],
        'bahaya': [
            'Batuk kering yang persisten adalah efek samping umum — konsultasikan ke dokter untuk evaluasi atau penggantian obat.',
            'Jangan konsumsi saat hamil atau berencana hamil — obat ini berbahaya untuk perkembangan janin.',
            'Segera ke UGD jika terjadi pembengkakan tiba-tiba pada wajah, bibir, atau tenggorokan (angioedema).',
        ],
    },
}


def prediksi_obat(usia, jenis_kelamin, tekanan_darah, kolesterol,
                  rasio_na_k, detak_jantung, gula_darah, bmi,
                  fungsi_hati, fungsi_ginjal):

    if not model_loaded:
        return ("⚠️  Model tidak tersedia", "", "", "", _panel_error())

    gender_enc  = 0 if jenis_kelamin == 'Perempuan' else 1
    bp_enc      = {'Rendah': 0, 'Normal': 1, 'Tinggi': 2}[tekanan_darah]
    chol_enc    = 0 if kolesterol == 'Tinggi' else 1
    bs_enc      = 0 if gula_darah == 'Tinggi' else 1
    liver_enc   = 0 if fungsi_hati == 'Tidak Normal' else 1
    kidney_enc  = 0 if fungsi_ginjal == 'Tidak Normal' else 1

    df = pd.DataFrame([[
        usia, gender_enc, bp_enc, chol_enc, rasio_na_k,
        detak_jantung, bs_enc, bmi, liver_enc, kidney_enc
    ]], columns=['Age','Gender','Blood_Pressure','Cholesterol','Na_to_K_Ratio',
                 'Heart_Rate','Blood_Sugar','BMI','Liver_Function','Kidney_Function'])

# Buat tabel baru khusus untuk menampung hasil desimal (float)
    df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns)

    # Masukkan df_scaled ke dalam model prediksi
    nama_obat = DAFTAR_OBAT[int(model.predict(df_scaled)[0])]
    info      = INFO_OBAT[nama_obat]

    return (
        f"{info['ikon']}  {nama_obat}",
        info['kelas'],
        info['fungsi'],
        "✅  Analisis berhasil",
        _build_panel(info, usia, tekanan_darah, kolesterol, gula_darah, bmi)
    )


def _item(ikon, judul, teks, color_title, color_body, color_border, sep=True):
    sep_style = f"padding-bottom:10px;border-bottom:1px solid {color_border};" if sep else ""
    return f"""
      <div style="display:flex;gap:10px;margin-bottom:10px;align-items:flex-start;{sep_style}">
        <span style="font-size:14px;margin-top:2px;">{ikon}</span>
        <div>
          <div style="color:{color_title};font-size:0.8rem;font-weight:600;margin-bottom:2px;">{judul}</div>
          <div style="color:{color_body};font-size:0.76rem;line-height:1.55;">{teks}</div>
        </div>
      </div>"""


def _section(bg, border, label_color, title, items_html):
    return f"""
    <div style="background:{bg};border:1px solid {border};border-radius:12px;
                padding:14px 16px;margin-bottom:14px;">
      <div style="color:{label_color};font-size:0.7rem;font-weight:600;
                  letter-spacing:1.3px;text-transform:uppercase;margin-bottom:10px;">
        {title}
      </div>
      {items_html}
    </div>"""


def _build_panel(info, usia, tekanan_darah, kolesterol, gula_darah, bmi):
    # Faktor risiko dari input pasien
    risiko = []
    if tekanan_darah == 'Tinggi':
        risiko.append(('🩺', 'Tekanan Darah Tinggi',
                        'Hipertensi adalah faktor risiko utama stroke dan penyakit jantung koroner.'))
    if kolesterol == 'Tinggi':
        risiko.append(('🫀', 'Kolesterol Tinggi',
                        'Penumpukan plak di arteri dapat menyempitkan aliran darah ke jantung.'))
    if gula_darah == 'Tinggi':
        risiko.append(('🍭', 'Gula Darah Tinggi',
                        'Hiperglikemia merusak dinding pembuluh darah dan saraf secara bertahap.'))
    if float(bmi) > 25:
        risiko.append(('⚖️', f'BMI {float(bmi):.1f} — Kelebihan Berat Badan',
                        'Obesitas meningkatkan beban kerja jantung dan resistensi insulin.'))
    if int(usia) > 55:
        risiko.append(('📅', f'Usia {int(usia)} Tahun — Faktor Risiko Usia',
                        'Risiko kardiovaskular meningkat signifikan di atas usia 55 tahun.'))

    html = '<div style="font-family:\'DM Sans\',sans-serif;">'

    # Blok faktor risiko (hanya tampil jika ada)
    if risiko:
        items = ''.join(_item(i, j, d, '#f09595', '#6a3a3a', 'rgba(226,75,74,0.12)') for i, j, d in risiko)
        html += _section('#1a0d0d', 'rgba(226,75,74,0.28)', '#f09595',
                         '⚠️  Faktor Risiko Terdeteksi', items)

    # Blok rekomendasi gaya hidup
    items_rek = ''.join(
        _item(i, j, d, '#97c459', '#4a7028', 'rgba(99,153,34,0.12)',
              sep=(n < len(info['rekomendasi']) - 1))
        for n, (i, j, d) in enumerate(info['rekomendasi'])
    )
    html += _section('#0a1a10', 'rgba(99,153,34,0.22)', '#97c459',
                     '✅  Rekomendasi Gaya Hidup', items_rek)

    # Blok peringatan klinis
    items_warn = ''.join(
        f"""<div style="display:flex;gap:10px;margin-bottom:8px;align-items:flex-start;">
              <span style="color:#378add;font-size:12px;font-weight:700;min-width:16px;margin-top:1px;">{n+1}.</span>
              <div style="color:#4a6888;font-size:0.76rem;line-height:1.55;">{p}</div>
            </div>"""
        for n, p in enumerate(info['bahaya'])
    )
    html += _section('#0d1a28', 'rgba(55,138,221,0.22)', '#85b7eb',
                     '⚕️  Peringatan Klinis Penting', items_warn)

    # Blok disclaimer
    html += """
    <div style="background:rgba(186,117,23,0.08);border:1px solid rgba(186,117,23,0.2);
                border-radius:10px;padding:12px 14px;">
      <div style="color:#fac775;font-size:0.7rem;font-weight:600;
                  letter-spacing:1.2px;text-transform:uppercase;margin-bottom:6px;">
        🔔  Perhatian — Bukan Pengganti Dokter
      </div>
      <div style="color:#7a6030;font-size:0.76rem;line-height:1.65;">
        Rekomendasi ini dihasilkan oleh sistem AI berbasis dataset sintetis.
        <strong style="color:#fac775;">Selalu konsultasikan kondisi Anda kepada dokter atau
        apoteker berlisensi</strong> sebelum memulai, menghentikan, atau mengubah terapi
        obat apapun. Keputusan medis akhir sepenuhnya berada di tangan tenaga kesehatan profesional.
      </div>
    </div>"""

    html += '</div>'
    return html


def _panel_error():
    return ('<div style="background:#1a0d0d;border:1px solid rgba(226,75,74,0.3);'
            'border-radius:12px;padding:20px;text-align:center;color:#f09595;font-size:0.88rem;">'
            '⚠️  Model tidak dapat dimuat.<br>'
            '<span style="color:#6a3a3a;font-size:0.78rem;">'
            'Pastikan file model_obat.joblib dan scaler_obat.joblib tersedia.</span></div>')


# ==============================================================================
# 2. CSS
# ==============================================================================
CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&family=Space+Grotesk:wght@500;600;700&display=swap');
*, *::before, *::after { box-sizing: border-box; }
body, .gradio-container {
    font-family: 'DM Sans', sans-serif !important;
    background: #0b1120 !important;
}
.gradio-container::before {
    content: '';
    position: fixed; inset: 0;
    background:
        radial-gradient(ellipse 70% 50% at 15% 8%, rgba(0,188,188,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 55% 45% at 82% 92%, rgba(0,119,182,0.06) 0%, transparent 60%),
        #0b1120;
    z-index: -1; pointer-events: none;
}
#header-block {
    background: linear-gradient(135deg, #0d2137 0%, #0f2c3f 100%);
    border: 1px solid rgba(0,188,188,0.2);
    border-radius: 18px !important;
    padding: 30px 36px !important;
    margin-bottom: 6px;
    position: relative; overflow: hidden;
}
#header-block::after {
    content: '';
    position: absolute; top: -50px; right: -50px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(0,188,188,0.1) 0%, transparent 70%);
    pointer-events: none;
}
#header-block h1 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.85rem !important; font-weight: 700 !important;
    color: #e0f7f7 !important; letter-spacing: -0.5px;
    margin: 0 0 6px 0 !important;
}
#header-block p { color: #7ecfcf !important; font-size: 0.9rem !important; margin: 0 !important; }
.panel-kiri, .panel-kanan {
    background: #0f1e2e !important;
    border: 1px solid rgba(0,188,188,0.14) !important;
    border-radius: 16px !important;
}
.slabel > p, .slabel p {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.68rem !important; font-weight: 600 !important;
    letter-spacing: 1.5px !important; text-transform: uppercase !important;
    color: #00bcbc !important; margin: 0 0 6px 0 !important;
}
.gr-form > label, label.svelte-1nnrg2n, .block > label, fieldset > div > label {
    color: #6aafc0 !important; font-size: 0.8rem !important; font-weight: 500 !important;
}
input[type=number], input[type=text], select, textarea {
    background: #0d1b2a !important;
    border: 1px solid rgba(0,188,188,0.18) !important;
    border-radius: 9px !important;
    color: #cce8f4 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
}
input:focus, select:focus, textarea:focus {
    border-color: #00bcbc !important;
    box-shadow: 0 0 0 3px rgba(0,188,188,0.1) !important;
    outline: none !important;
}
input[type=range] { accent-color: #00bcbc !important; }
.gr-radio label, .gr-check-radio label { color: #8ecad8 !important; font-size: 0.83rem !important; }
#btn-analisis {
    background: linear-gradient(135deg, #00bcbc 0%, #0077b6 100%) !important;
    border: none !important; border-radius: 11px !important;
    color: #fff !important; font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.95rem !important; font-weight: 600 !important;
    padding: 14px 24px !important; cursor: pointer !important;
    transition: opacity 0.2s, transform 0.15s !important;
    box-shadow: 0 4px 20px rgba(0,188,188,0.22) !important;
    letter-spacing: 0.2px !important;
}
#btn-analisis:hover  { opacity: 0.88 !important; transform: translateY(-1px) !important; }
#btn-analisis:active { transform: translateY(0) !important; }
#out-judul textarea, #out-judul input {
    background: #071525 !important;
    border: 1px solid rgba(0,188,188,0.3) !important;
    border-radius: 12px !important; color: #00e5e5 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.35rem !important; font-weight: 700 !important;
    text-align: center !important; padding: 18px !important;
}
#out-kelas textarea, #out-kelas input {
    background: #0a1520 !important;
    border: 1px solid rgba(0,188,188,0.12) !important;
    border-radius: 9px !important; color: #00bcbc !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.8rem !important; font-weight: 600 !important;
    letter-spacing: 0.5px !important; text-transform: uppercase !important;
    padding: 10px 14px !important;
}
#out-fungsi textarea, #out-fungsi input {
    background: #0a1a28 !important;
    border: 1px solid rgba(0,188,188,0.12) !important;
    border-radius: 9px !important; color: #a8d8e8 !important;
    font-style: italic !important; font-size: 0.87rem !important;
    padding: 12px 16px !important; line-height: 1.6 !important;
}
#out-status textarea, #out-status input {
    background: rgba(0,188,188,0.06) !important;
    border: 1px solid rgba(0,188,188,0.2) !important;
    border-radius: 8px !important; color: #5ee8d6 !important;
    font-size: 0.8rem !important; font-weight: 500 !important;
    text-align: center !important; padding: 9px !important;
}
.gr-accordion {
    background: #0d1b2a !important;
    border: 1px solid rgba(0,188,188,0.12) !important;
    border-radius: 11px !important;
}
.gr-accordion summary {
    color: #6aafc0 !important; font-size: 0.84rem !important; font-weight: 500 !important;
}
hr { border-color: rgba(0,188,188,0.08) !important; }
"""

# ==============================================================================
# 3. ANTARMUKA WEB
# ==============================================================================
with gr.Blocks(title="ClinicalAI — Rekomendasi Obat Klinis") as demo:

    with gr.Group(elem_id="header-block"):
        gr.Markdown("""
<h1>🧬 ClinicalAI — Sistem Rekomendasi Obat Klinis</h1>
<p>Analisis Obat Yang Dibutuhkan Berdasarkan Profil klinis Pasien  Berbasis algoritma Machine Learning <strong> LightGBM</strong>
        """)

    with gr.Row(equal_height=False):

        # ── Kolom Kiri: Input ────────────────────────────────────────────────
        with gr.Column(scale=4, min_width=290, elem_classes="panel-kiri"):
            gr.Markdown("**DATA DEMOGRAFIS**", elem_classes="slabel")
            with gr.Row():
                usia          = gr.Slider(15, 85, value=45, step=1, label="🗓 Usia (Tahun)")
                jenis_kelamin = gr.Radio(["Laki-laki", "Perempuan"],
                                         label="⚧ Jenis Kelamin", value="Laki-laki")

            gr.Markdown("**PARAMETER VITAL**", elem_classes="slabel")
            with gr.Row():
                tekanan_darah = gr.Dropdown(["Rendah","Normal","Tinggi"],
                                            label="🩺 Tekanan Darah", value="Normal")
                kolesterol    = gr.Dropdown(["Normal","Tinggi"],
                                            label="🫀 Kolesterol", value="Normal")
            rasio_na_k = gr.Slider(6.0, 38.0, value=15.0, step=0.1,
                                   label="⚗ Rasio Na/K Darah")

            with gr.Accordion("🔬  Parameter Lanjutan (Opsional)", open=False):
                gr.Markdown(
                    "<p style='color:#3d6a7a;font-size:0.79rem;margin:0 0 10px 0;'>"
                    "Parameter tambahan meningkatkan presisi prediksi model.</p>"
                )
                with gr.Row():
                    detak_jantung = gr.Slider(55, 120, value=80, step=1,
                                              label="💓 Detak Jantung (BPM)")
                    gula_darah    = gr.Radio(["Normal","Tinggi"],
                                             label="🩸 Gula Darah", value="Normal")
                with gr.Row():
                    fungsi_hati   = gr.Radio(["Normal","Tidak Normal"],
                                             label="🫁 Fungsi Hati", value="Normal")
                    fungsi_ginjal = gr.Radio(["Normal","Tidak Normal"],
                                             label="🫘 Fungsi Ginjal", value="Normal")
                bmi = gr.Slider(15.0, 45.0, value=24.0, step=0.1,
                                label="⚖ BMI (Indeks Massa Tubuh)")

            btn = gr.Button("⚡  Analisis & Rekomendasikan Obat",
                            variant="primary", size="lg", elem_id="btn-analisis")

        # ── Kolom Tengah: Hasil Prediksi ─────────────────────────────────────
        with gr.Column(scale=4, min_width=270, elem_classes="panel-kiri"):
            gr.Markdown("**HASIL PREDIKSI AI**", elem_classes="slabel")
            out_judul  = gr.Textbox(label="Obat yang Direkomendasikan",
                                    placeholder="— menunggu analisis —",
                                    interactive=False, lines=2,
                                    elem_id="out-judul")
            out_kelas  = gr.Textbox(label="Kelas Farmakologi", placeholder="...",
                                    interactive=False, lines=1, elem_id="out-kelas")
            out_fungsi = gr.Textbox(label="Mekanisme & Fungsi",
                                    placeholder="Penjelasan fungsi obat akan muncul di sini...",
                                    interactive=False, lines=3, elem_id="out-fungsi")
            out_status = gr.Textbox(label="Status Analisis", placeholder="...",
                                    interactive=False, lines=1, elem_id="out-status")

            gr.HTML("""
<div style="background:rgba(0,119,182,0.07);border:1px solid rgba(0,119,182,0.18);
            border-radius:12px;padding:14px 18px;margin-top:8px;">
  <div style="color:#00bcbc;font-size:0.7rem;font-weight:600;
              letter-spacing:1.3px;text-transform:uppercase;margin-bottom:8px;">
    💊 Kandidat Obat dalam Model
  </div>
  <div style="display:flex;flex-wrap:wrap;gap:6px;">
    <span style="background:rgba(0,188,188,0.1);border:1px solid rgba(0,188,188,0.2);
                 border-radius:20px;color:#5ee8d6;font-size:0.74rem;padding:3px 12px;">Amlodipine</span>
    <span style="background:rgba(0,188,188,0.1);border:1px solid rgba(0,188,188,0.2);
                 border-radius:20px;color:#5ee8d6;font-size:0.74rem;padding:3px 12px;">Aspirin</span>
    <span style="background:rgba(0,188,188,0.1);border:1px solid rgba(0,188,188,0.2);
                 border-radius:20px;color:#5ee8d6;font-size:0.74rem;padding:3px 12px;">Atorvastatin</span>
    <span style="background:rgba(0,188,188,0.1);border:1px solid rgba(0,188,188,0.2);
                 border-radius:20px;color:#5ee8d6;font-size:0.74rem;padding:3px 12px;">Hydrochlorothiazide</span>
    <span style="background:rgba(0,188,188,0.1);border:1px solid rgba(0,188,188,0.2);
                 border-radius:20px;color:#5ee8d6;font-size:0.74rem;padding:3px 12px;">Lisinopril</span>
  </div>
</div>
            """)

        # ── Kolom Kanan: Panel Rekomendasi ───────────────────────────────────
        with gr.Column(scale=5, min_width=300, elem_classes="panel-kanan"):
            gr.Markdown("**PANDUAN KESEHATAN PERSONAL**", elem_classes="slabel")
            out_panel = gr.HTML(
                value="""
<div style="background:#0d1525;border:1px solid rgba(0,188,188,0.1);
            border-radius:12px;padding:28px;text-align:center;
            font-family:'DM Sans',sans-serif;">
  <div style="font-size:2.2rem;margin-bottom:12px;">🏥</div>
  <div style="font-size:0.9rem;font-weight:500;color:#2a6070;margin-bottom:8px;">
    Menunggu Hasil Analisis
  </div>
  <div style="font-size:0.78rem;color:#1a3a48;line-height:1.7;">
    Isi formulir pasien dan tekan tombol<br>
    <strong style="color:#2a7080;">Analisis &amp; Rekomendasikan Obat</strong><br>
    untuk mendapatkan panduan kesehatan personal<br>berdasarkan profil klinis pasien.
  </div>
</div>""",
                elem_id="panel-rekomendasi"
            )

    gr.HTML("""
<div style="text-align:center;margin-top:16px;color:#2a5060;font-size:0.74rem;
            border-top:1px solid rgba(0,188,188,0.08);padding-top:14px;line-height:1.8;
            font-family:'DM Sans',sans-serif;">
  ⚕ ClinicalAI — Proyek Machine Learning Klasifikasi Obat Klinis &nbsp;|&nbsp;
  Algoritma: LightGBM &nbsp;|&nbsp; Akurasi: 99% &nbsp;|&nbsp; Dataset: Sintetis<br>
  <span style="color:#1a3040;">
    Sistem ini adalah alat bantu keputusan klinis awal, bukan pengganti resep dokter.
    Selalu konsultasikan kepada tenaga medis berlisensi.
  </span>
</div>
    """)

    btn.click(
        fn=prediksi_obat,
        inputs=[usia, jenis_kelamin, tekanan_darah, kolesterol, rasio_na_k,
                detak_jantung, gula_darah, bmi, fungsi_hati, fungsi_ginjal],
        outputs=[out_judul, out_kelas, out_fungsi, out_status, out_panel]
    )

# ==============================================================================
# 4. JALANKAN SERVER
# ==============================================================================
if __name__ == "__main__":
    demo.launch(share=True, show_error=True)