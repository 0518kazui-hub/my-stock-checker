from concurrent.futures import ThreadPoolExecutor
import time
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import yfinance as yf

# ====================================================
# 1. ページ基本設定（スマホ画面に最適化）
# ====================================================
st.set_page_config(
    page_title="My株チェッカー",
    page_icon="📈",
    layout="wide",
)

# 簡易パスワード保護（不要な場合は下のブロックを削除してください）
APP_PASSWORD = "7777"  # ← お好きな4桁などのパスワードに変更できます
if "authenticated" not in st.session_state:
  st.session_state.authenticated = False

if not st.session_state.authenticated:
  st.title("📈 My株チェッカー（ログイン）")
  pwd = st.text_input("パスワードを入力してください", type="password")
  if st.button("ログイン"):
    if pwd == APP_PASSWORD:
      st.session_state.authenticated = True
      st.rerun()
    else:
      st.error("パスワードが違います")
  st.stop()

# ====================================================
# 2. デフォルトの保有株・国策ウォッチリスト（自由に追加・編集可能）
# ====================================================
DEFAULT_WATCHLIST = [
    # --- 保有・買付依頼中の主力ポートフォリオ（8社） ---
    ("9434.T", "ソフトバンク", "【保有】通信インフラ"),
    ("9432.T", "NTT", "【保有】通信インフラ"),
    ("1414.T", "ショーボンドHD", "【保有】国策インフラ補修"),
    ("6200.T", "インソース", "【保有】人材・DX研修"),
    ("5802.T", "住友電気工業", "【保有】電力・通信網"),
    ("6902.T", "デンソー", "【依頼中】自動車・モビリティ"),
    ("4828.T", "ビジネスエンジニアリング", "【依頼中】企業DX・ERP"),
    ("8593.T", "三菱HCキャピタル", "【指値待】連続増配リース"),
    # --- 国策①：レアアース・資源安保 ---
    ("5714.T", "DOWAホールディングス", "【レアアース】都市鉱山"),
    ("5471.T", "大同特殊鋼", "【レアアース】省レアアース磁石"),
    ("5711.T", "三菱マテリアル", "【レアアース】リサイクル・非鉄"),
    ("4063.T", "信越化学工業", "【レアアース】磁石・半導体素材"),
    ("6269.T", "三井海洋開発", "【レアアース】深海資源開発"),
    # --- 国策②：防衛・安全保障 ---
    ("7011.T", "三菱重工業", "【防衛】防衛装備・重工"),
    ("7012.T", "川崎重工業", "【防衛】潜水艦・航空機"),
    ("7013.T", "IHI", "【防衛】航空エンジン・宇宙"),
    ("6503.T", "三菱電機", "【防衛】防衛レーダー・電子"),
    ("6701.T", "NEC", "【防衛】防衛通信・サイバー"),
    # --- 監視候補：割安高配当インフラ ---
    ("1911.T", "住友林業", "【監視】住宅・資源インフラ"),
]


# ====================================================
# 3. 財務・利回り計算用ヘルパー関数
# ====================================================
def get_first_valid_value(df, col, candidate_keys):
  for key in candidate_keys:
    if key in df.index:
      val = df.loc[key, col]
      if isinstance(val, pd.Series):
        val = val.dropna().iloc if not val.dropna().empty else None
      if val is not None and pd.notna(val):
        return float(val)
  return None


def calc_dividend_streak(divs_series):
  if divs_series.empty:
    return 0, 0
  annual_divs = divs_series.groupby(divs_series.index.year).sum()
  current_year = pd.Timestamp.now().year
  completed_years = annual_divs[annual_divs.index < current_year]
  if len(completed_years) < 2:
    return 0, 0
  vals = completed_years.values
  inc_streak = 0
  non_dec_streak = 0
  for i in range(len(vals) - 1, 0, -1):
    if vals[i] > vals[i - 1] * 1.001:
      inc_streak += 1
    else:
      break
  for i in range(len(vals) - 1, 0, -1):
    if vals[i] >= vals[i - 1] * 0.999:
      non_dec_streak += 1
    else:
      break
  return inc_streak, non_dec_streak


def analyze_single_stock(item):
  code, stock_name, tag = item
  now_normalized = pd.Timestamp.now().normalize()
  three_years_ago = now_normalized - pd.DateOffset(years=3)
  one_year_ago = now_normalized - pd.DateOffset(years=1)
  six_months_ago = now_normalized - pd.DateOffset(months=6)
  tax_rate = 0.30

  try:
    time.sleep(0.25)
    ticker = yf.Ticker(code)

    current_price = ticker.fast_info.get("lastPrice")
    if not current_price or pd.isna(current_price):
      return None

    market_cap_raw = ticker.fast_info.get("marketCap")
    market_cap_yen = float(market_cap_raw) if market_cap_raw else 0.0

    divs = ticker.dividends
    if divs.empty:
      div_yield = 0.0
      non_dec_streak = 0
    else:
      divs.index = pd.to_datetime(divs.index).tz_localize(None).normalize()
      divs = divs.groupby(divs.index).sum()
      trailing_1y_div = divs[divs.index >= one_year_ago].sum()
      div_yield = (trailing_1y_div / current_price) * 100
      _, non_dec_streak = calc_dividend_streak(divs)

    # 過去3年間の配当利回り時系列計算
    hist = ticker.history(period="5y")
    if hist.empty:
      return None
    hist.index = pd.to_datetime(hist.index).tz_localize(None).normalize()

    df_price = pd.DataFrame(index=hist.index)
    df_price["Close"] = hist["Close"]
    df_price["Dividends"] = 0.0
    if not divs.empty:
      for div_date, div_amount in divs.items():
        valid_dates = df_price.index[df_price.index >= div_date]
        if not valid_dates.empty:
          df_price.loc[next(iter(valid_dates)), "Dividends"] += float(div_amount)
    df_price["Trailing_1Y_Div"] = df_price["Dividends"].rolling("365D").sum()
    df_price["Dividend_Yield"] = (
        df_price["Trailing_1Y_Div"] / df_price["Close"]
    ) * 100

    df_3y = df_price[df_price.index >= three_years_ago].copy()
    if df_3y.empty:
      return None

    max_val = float(df_3y["Dividend_Yield"].max())
    min_val = float(df_3y["Dividend_Yield"].min())
    mean_val = float(df_3y["Dividend_Yield"].mean())
    latest_val = float(df_3y["Dividend_Yield"].tail(1).squeeze())

    past_1y_df = df_3y[df_3y.index <= one_year_ago]
    yield_1y_ago = (
        float(past_1y_df["Dividend_Yield"].tail(1).squeeze())
        if not past_1y_df.empty
        else float(df_3y["Dividend_Yield"].head(1).squeeze())
    )
    past_6m_df = df_3y[df_3y.index <= six_months_ago]
    yield_6m_ago = (
        float(past_6m_df["Dividend_Yield"].tail(1).squeeze())
        if not past_6m_df.empty
        else yield_1y_ago
    )

    diff_3y_mean = latest_val - mean_val
    diff_1y_ago = latest_val - yield_1y_ago
    diff_6m_ago = latest_val - yield_6m_ago

    if (diff_3y_mean > 0) and (diff_1y_ago > 0) and (diff_6m_ago > 0):
      trend_status = "🔥 3段階上昇（強い買い場）"
    elif (diff_3y_mean > 0) and (diff_1y_ago > 0):
      trend_status = "📈 中期上昇（直近一服）"
    elif (diff_3y_mean > 0) and (diff_6m_ago > 0):
      trend_status = "⚡ 直近反転上昇（初動）"
    elif diff_3y_mean > 0:
      trend_status = "➡️ 割安圏横ばい"
    else:
      trend_status = "⬇️ 低下・割高圏"

    range_position = (
        ((latest_val - min_val) / (max_val - min_val)) * 100
        if max_val > min_val
        else 50.0
    )

    # 財務・ROIC推移・FCF
    fin = ticker.financials
    bs = ticker.balance_sheet
    cf = ticker.cashflow
    roic_trend_str = "—"
    equity_ratio = None
    fcf_oku = None

    if not fin.empty and not bs.empty:
      latest_bs = next(iter(bs.columns))
      total_assets = get_first_valid_value(bs, latest_bs, ["Total Assets"])
      equity = get_first_valid_value(
          bs,
          latest_bs,
          ["Stockholders Equity", "Total Equity Gross Minority Interest"],
      )
      if equity and total_assets and total_assets > 0:
        equity_ratio = round((equity / total_assets) * 100, 1)

      common_cols = sorted(list(set(fin.columns).intersection(set(bs.columns))))
      roic_history = []
      for i, col in enumerate(common_cols):
        op_i = get_first_valid_value(
            fin, col, ["Operating Income", "EBIT", "Operating Profit"]
        )
        eq_i = get_first_valid_value(
            bs,
            col,
            ["Stockholders Equity", "Total Equity Gross Minority Interest"],
        )
        db_i = (
            get_first_valid_value(bs, col, ["Total Debt", "Long Term Debt"])
            or 0.0
        )
        if op_i is not None and eq_i is not None and eq_i > 0:
          ic_a = eq_i + db_i
          if i >= 1:
            p_col = common_cols[i - 1]
            p_eq = get_first_valid_value(
                bs,
                p_col,
                ["Stockholders Equity", "Total Equity Gross Minority Interest"],
            )
            p_db = (
                get_first_valid_value(
                    bs, p_col, ["Total Debt", "Long Term Debt"]
                )
                or 0.0
            )
            if p_eq is not None and p_eq > 0:
              ic_a = (ic_a + (p_eq + p_db)) / 2.0
          if ic_a > 0:
            roic_history.append(((op_i * (1 - tax_rate)) / ic_a) * 100)
      if roic_history:
        roic_trend_str = "→".join(f"{r:.1f}%" for r in roic_history)

    if not cf.empty:
      latest_cf = next(iter(cf.columns))
      fcf_raw = get_first_valid_value(cf, latest_cf, ["Free Cash Flow"])
      if fcf_raw is None:
        ocf = get_first_valid_value(
            cf,
            latest_cf,
            [
                "Operating Cash Flow",
                "Cash Flow From Continuing Operating Activities",
            ],
        )
        capex = (
            get_first_valid_value(
                cf, latest_cf, ["Capital Expenditure", "Purchase Of PPE"]
            )
            or 0.0
        )
        if ocf is not None:
          fcf_raw = ocf + capex if capex < 0 else ocf - capex
      if fcf_raw is not None:
        fcf_oku = round(fcf_raw / 100_000_000, 1)

    return {
        "コード": code.replace(".T", ""),
        "銘柄名": stock_name,
        "区分": tag,
        "現在株価(円)": round(current_price, 1),
        "配当利回り(%)": round(div_yield, 2),
        "3年レンジ位置(%)": round(range_position, 1),
        "買い場判定": trend_status,
        "3年平均利回り(%)": round(mean_val, 2),
        "非減配(年)": non_dec_streak,
        "ROIC推移": roic_trend_str,
        "FCF(億円)": fcf_oku,
        "自己資本比率(%)": equity_ratio,
        "_df_3y": df_3y,
        "_mean_val": mean_val,
    }
  except Exception:
    return None


# ====================================================
# 4. スマホアプリの画面UI
# ====================================================
st.title("📈 My株ウォッチリスト診断")
st.caption("保有株 ＋ レアアース・防衛国策銘柄の15秒スピード健康診断")

# サイドバー：銘柄の追加・フィルター機能
with st.sidebar:
  st.header("⚙️ 設定・銘柄追加")
  filter_tag = st.selectbox(
      "表示するカテゴリ",
      [
          "すべて表示",
          "保有・依頼中のみ",
          "レアアースのみ",
          "防衛のみ",
          "監視のみ",
      ],
  )
  st.divider()
  st.subheader("➕ 臨時で銘柄を追加監視")
  extra_code = st.text_input("証券コード（4桁・例: 8056）", "")
  extra_name = st.text_input("銘柄名（例: BIPROGY）", "")
  if "custom_list" not in st.session_state:
    st.session_state.custom_list = []
  if st.button("リストに追加"):
    if extra_code.strip():
      c_fmt = (
          extra_code.strip() + ".T"
          if not extra_code.strip().endswith(".T")
          else extra_code.strip()
      )
      n_fmt = extra_name.strip() if extra_name.strip() else c_fmt
      st.session_state.custom_list.append((c_fmt, n_fmt, "【追加】カスタム"))
      st.success(f"{n_fmt} を追加しました！")

# スキャン実行ボタン
if st.button(
    "🔍 最新データをスキャン（約15秒）",
    type="primary",
    use_container_width=True,
):
  target_list = DEFAULT_WATCHLIST + st.session_state.custom_list
  with st.spinner("Yahoo Financeから最新株価・財務・配当を取得中..."):
    with ThreadPoolExecutor(max_workers=4) as executor:
      raw_results = list(executor.map(analyze_single_stock, target_list))

  valid_results = [r for r in raw_results if r is not None]
  if valid_results:
    df_res = pd.DataFrame(valid_results)
    df_res = df_res.sort_values(
        by="3年レンジ位置(%)", ascending=False
    ).reset_index(drop=True)
    st.session_state.df_res = df_res
  else:
    st.error("データの取得に失敗しました。少し時間をおいて再度お試しください。")

# 結果の表示エリア
if "df_res" in st.session_state:
  df_show = st.session_state.df_res.copy()

  if filter_tag == "保有・依頼中のみ":
    df_show = df_show[
        df_show["区分"].str.contains("保有|依頼中|指値待")
    ].reset_index(drop=True)
  elif filter_tag == "レアアースのみ":
    df_show = df_show[df_show["区分"].str.contains("レアアース")].reset_index(
        drop=True
    )
  elif filter_tag == "防衛のみ":
    df_show = df_show[df_show["区分"].str.contains("防衛")].reset_index(
        drop=True
    )
  elif filter_tag == "監視のみ":
    df_show = df_show[df_show["区分"].str.contains("監視")].reset_index(
        drop=True
    )

  st.subheader(f"📊 割安・買い場ランキング（{len(df_show)}銘柄）")
  st.caption(
      "※「3年レンジ位置(%)」が70%以上の銘柄は、過去3年間で見て割安な買い場水準です。"
  )

  display_cols = [
      "コード",
      "銘柄名",
      "区分",
      "現在株価(円)",
      "配当利回り(%)",
      "3年レンジ位置(%)",
      "買い場判定",
      "3年平均利回り(%)",
      "ROIC推移",
      "FCF(億円)",
      "自己資本比率(%)",
  ]
  st.dataframe(
      df_show[display_cols],
      use_container_width=True,
      hide_index=True,
  )

  # 個別銘柄の過去3年配当利回りチャート表示
  st.divider()
  st.subheader("📉 過去3年間の配当利回りチャート確認")
  selected_stock = st.selectbox(
      "チャートを見たい銘柄を選択",
      df_show["コード"] + " : " + df_show["銘柄名"],
  )
  if selected_stock:
    sel_code = selected_stock.split(" : ")[0]
    row_data = df_show[df_show["コード"] == sel_code].iloc[0]
    df_chart = row_data["_df_3y"]
    mean_v = row_data["_mean_val"]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(
        df_chart.index,
        df_chart["Dividend_Yield"],
        color="#1f77b4",
        linewidth=1.8,
        label="Dividend Yield (%)",
    )
    ax.axhline(
        mean_v,
        color="#ff7f0e",
        linestyle="--",
        linewidth=1.5,
        label=f"3Y Mean ({mean_v:.2f}%)",
    )
    ax.set_ylabel("Yield (%)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    st.pyplot(fig)
