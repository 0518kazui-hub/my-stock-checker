import base64
from concurrent.futures import ThreadPoolExecutor
import json
import time
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib_fontja
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

# ====================================================
# 1. ページ基本設定（スマホ画面に最適化）
# ====================================================
st.set_page_config(
    page_title="My株チェッカー Pro",
    page_icon="📈",
    layout="wide",
)

APP_PASSWORD = "7777"
if "authenticated" not in st.session_state:
  st.session_state.authenticated = False

if not st.session_state.authenticated:
  st.markdown(
      '<div style="font-size: 20px; font-weight: bold; margin-bottom: 10px;">'
      "📈 My株チェッカー（ログイン）</div>",
      unsafe_allow_html=True,
  )
  pwd = st.text_input("パスワードを入力してください", type="password")
  if st.button("ログイン"):
    if pwd == APP_PASSWORD:
      st.session_state.authenticated = True
      st.rerun()
    else:
      st.error("パスワードが違います")
  st.stop()

# ====================================================
# 2. 永続化ウォッチリスト（GitHubへ自動保存・読み込み）
# ====================================================
INITIAL_WATCHLIST = [
    ["9434.T", "ソフトバンク", "【保有】通信インフラ"],
    ["9432.T", "NTT", "【保有】通信インフラ"],
    ["1414.T", "ショーボンドHD", "【保有】国策インフラ補修"],
    ["6200.T", "インソース", "【保有】人材・DX研修"],
    ["5802.T", "住友電気工業", "【保有】電力・通信網"],
    ["6902.T", "デンソー", "【依頼中】自動車・モビリティ"],
    ["4828.T", "ビジネスエンジニアリング", "【依頼中】企業DX・ERP"],
    ["8593.T", "三菱HCキャピタル", "【指値待】連続増配リース"],
    ["5714.T", "DOWAホールディングス", "【レアアース】都市鉱山"],
    ["5471.T", "大同特殊鋼", "【レアアース】省レアアース磁石"],
    ["5711.T", "三菱マテリアル", "【レアアース】リサイクル・非鉄"],
    ["4063.T", "信越化学工業", "【レアアース】磁石・半導体素材"],
    ["6269.T", "三井海洋開発", "【レアアース】深海資源開発"],
    ["7011.T", "三菱重工業", "【防衛】防衛装備・重工"],
    ["7012.T", "川崎重工業", "【防衛】潜水艦・航空機"],
    ["7013.T", "IHI", "【防衛】航空エンジン・宇宙"],
    ["6503.T", "三菱電機", "【防衛】防衛レーダー・電子"],
    ["6701.T", "NEC", "【防衛】防衛通信・サイバー"],
    ["1911.T", "住友林業", "【監視】住宅・資源インフラ"],
    ["5805.T", "ＳＷＣＣ", "【監視】電力・DCケーブル"],
]


def load_watchlist_from_github():
  try:
    token = st.secrets["GITHUB_TOKEN"]
    repo = st.secrets["GITHUB_REPO"]
    domain = ".".join(["api", "github", "com"])
    path = "/".join(["repos", repo, "contents", "watchlist.json"])
    url = f"https://{domain}/{path}"
    headers = {"Authorization": f"token {token}"}
    res = requests.get(url, headers=headers, timeout=5)
    if res.status_code == 200:
      content = base64.b64decode(res.json()["content"]).decode("utf-8")
      return json.loads(content)
  except Exception:
    pass
  return INITIAL_WATCHLIST


def save_watchlist_to_github(new_list):
  try:
    token = st.secrets["GITHUB_TOKEN"].strip()
    repo = st.secrets["GITHUB_REPO"].strip()
    domain = ".".join(["api", "github", "com"])
    path = "/".join(["repos", repo, "contents", "watchlist.json"])
    url = f"https://{domain}/{path}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }
    get_res = requests.get(url, headers=headers, timeout=5)
    sha = get_res.json().get("sha") if get_res.status_code == 200 else None

    json_str = json.dumps(new_list, ensure_ascii=False, indent=2)
    b64_content = base64.b64encode(json_str.encode("utf-8")).decode("utf-8")
    payload = {
        "message": "Update watchlist via MyStockChecker App",
        "content": b64_content,
    }
    if sha:
      payload["sha"] = sha
    put_res = requests.put(url, headers=headers, json=payload, timeout=8)
    if put_res.status_code in (200, 201):
      return True
    else:
      st.error(
          f"GitHubエラー ({put_res.status_code}):"
          f" {put_res.json().get('message')}"
      )
      return False
  except Exception as e:
    st.error(f"通信エラー: {e}")
    return False



if "watchlist" not in st.session_state:
  st.session_state.watchlist = load_watchlist_from_github()


# ====================================================
# 3. テクニカル＆財務計算ヘルパー関数
# ====================================================
def calc_rsi(series, period=14):
  delta = series.diff()
  gain = delta.clip(lower=0)
  loss = -delta.clip(upper=0)
  avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
  avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
  rs = avg_gain / avg_loss.replace(0, np.nan)
  rsi = 100 - (100 / (1 + rs))
  return rsi.fillna(50.0)


def calc_range_pct(series, current_val):
  s_min = float(series.min())
  s_max = float(series.max())
  if s_max <= s_min:
    return 50.0
  return ((current_val - s_min) / (s_max - s_min)) * 100.0


def get_first_valid_value(df, col, candidate_keys):
  for key in candidate_keys:
    if key in df.index:
      val = df.loc[key, col]
      if isinstance(val, pd.Series):
        val = val.dropna().head(1).item() if not val.dropna().empty else None
      if val is not None and pd.notna(val):
        return float(val)
  return None


def analyze_mtf_stock(item):
  code, stock_name, tag = item
  now_norm = pd.Timestamp.now().normalize()
  dt_5y = now_norm - pd.DateOffset(years=5)
  dt_3y = now_norm - pd.DateOffset(years=3)
  dt_1y = now_norm - pd.DateOffset(years=1)
  dt_6m = now_norm - pd.DateOffset(months=6)
  dt_3m = now_norm - pd.DateOffset(months=3)
  dt_1m = now_norm - pd.DateOffset(months=1)
  tax_rate = 0.30

  try:
    time.sleep(0.25)
    ticker = yf.Ticker(code)

    hist = ticker.history(period="6y")
    if hist.empty or len(hist) < 250:
      return None
    hist.index = pd.to_datetime(hist.index).tz_localize(None).normalize()

    df = pd.DataFrame(index=hist.index)
    df["Close"] = hist["Close"]
    df["MA25"] = df["Close"].rolling(25).mean()
    df["MA75"] = df["Close"].rolling(75).mean()
    df["MA200"] = df["Close"].rolling(200).mean()
    df["RSI14"] = calc_rsi(df["Close"], 14)

    divs = ticker.dividends
    df["Dividends"] = 0.0
    if not divs.empty:
      divs.index = pd.to_datetime(divs.index).tz_localize(None).normalize()
      divs = divs.groupby(divs.index).sum()
      for d_date, d_amt in divs.items():
        v_dates = df.index[df.index >= d_date]
        if not v_dates.empty:
          df.loc[next(iter(v_dates)), "Dividends"] += float(d_amt)

    df["Trailing_1Y_Div"] = df["Dividends"].rolling("365D").sum()
    df["Div_Yield"] = (df["Trailing_1Y_Div"] / df["Close"]) * 100.0

    df_5y = df[df.index >= dt_5y].copy()
    df_3y = df[df.index >= dt_3y].copy()
    df_1y = df[df.index >= dt_1y].copy()
    df_6m = df[df.index >= dt_6m].copy()
    df_3m = df[df.index >= dt_3m].copy()
    df_1m = df[df.index >= dt_1m].copy()

    if df_3y.empty or df_1m.empty:
      return None

    latest_close = float(df["Close"].iloc[-1])
    latest_yield = float(df["Div_Yield"].iloc[-1])
    latest_rsi = float(df["RSI14"].iloc[-1])
    ma25 = float(df["MA25"].iloc[-1])
    ma200 = float(df["MA200"].iloc[-1])

    yield_pos_5y = calc_range_pct(df_5y["Div_Yield"], latest_yield)
    yield_pos_3y = calc_range_pct(df_3y["Div_Yield"], latest_yield)
    price_pos_5y = calc_range_pct(df_5y["Close"], latest_close)
    price_pos_1y = calc_range_pct(df_1y["Close"], latest_close)
    price_pos_6m = calc_range_pct(df_6m["Close"], latest_close)
    dev_ma200 = ((latest_close - ma200) / ma200) * 100.0 if ma200 > 0 else 0.0
    dev_ma25 = ((latest_close - ma25) / ma25) * 100.0 if ma25 > 0 else 0.0

    first_1m_close = float(df_1m["Close"].head(1).item())
    ret_1m = (
        ((latest_close - first_1m_close) / first_1m_close) * 100.0
        if len(df_1m) >= 2 and first_1m_close > 0
        else 0.0
    )
    min_1m = float(df_1m["Close"].min())
    rebound_from_1m_low = (
        ((latest_close - min_1m) / min_1m) * 100.0 if min_1m > 0 else 0.0
    )
    # ====================================================
    # 4. 買い場スコア（0〜100点）＆ 売り場・過熱スコア（0〜100点）
    # ====================================================
    b_long = (yield_pos_3y * 0.20) + (yield_pos_5y * 0.15)
    b_mid = ((100.0 - price_pos_1y) * 0.15) + ((100.0 - price_pos_6m) * 0.20)
    b_rsi = min(max(60.0 - latest_rsi, 0.0) / 30.0, 1.0) * 15.0
    b_ma25 = min(max(-dev_ma25, 0.0) / 8.0, 1.0) * 10.0
    b_rebound = 5.0 if (0.5 <= rebound_from_1m_low <= 5.0) else 2.0
    buy_score = round(min(b_long + b_mid + b_rsi + b_ma25 + b_rebound, 100.0), 1)

    s_long = ((100.0 - yield_pos_3y) * 0.20) + (price_pos_5y * 0.15)
    s_ma200 = min(max(dev_ma200, 0.0) / 25.0, 1.0) * 20.0
    s_mid = (price_pos_6m / 100.0) * 15.0
    s_rsi = min(max(latest_rsi - 55.0, 0.0) / 25.0, 1.0) * 20.0
    s_1m = min(max(ret_1m, 0.0) / 15.0, 1.0) * 10.0
    sell_score = round(min(s_long + s_ma200 + s_mid + s_rsi + s_1m, 100.0), 1)

    if buy_score >= 70.0 and sell_score <= 35.0:
      signal_label = "🔥 絶好の買い場（長期割安×短期押し目）"
    elif buy_score >= 58.0:
      signal_label = "📈 買い検討圏（押し目水準）"
    elif sell_score >= 72.0:
      signal_label = "⚠️ 売り場・過熱警戒（一部利確検討）"
    elif sell_score >= 60.0:
      signal_label = "⚡ 高値圏・新規買い控え"
    else:
      signal_label = "➡️ 中立・ホールド継続"
    # 財務・ROIC推移・FCF（クラウド環境で取得エラーが出ても株価・6期間分析は止めない）
    roic_trend_str = "—"
    equity_ratio = None
    fcf_oku = None
    try:
      fin = ticker.financials
      bs = ticker.balance_sheet
      cf = ticker.cashflow
      if fin is not None and not fin.empty and bs is not None and not bs.empty:
        latest_bs = next(iter(bs.columns))
        total_assets = get_first_valid_value(bs, latest_bs, ["Total Assets"])
        equity = get_first_valid_value(
            bs,
            latest_bs,
            ["Stockholders Equity", "Total Equity Gross Minority Interest"],
        )
        if equity and total_assets and total_assets > 0:
          equity_ratio = round((equity / total_assets) * 100, 1)

        common_cols = sorted(
            list(set(fin.columns).intersection(set(bs.columns)))
        )
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
                  [
                      "Stockholders Equity",
                      "Total Equity Gross Minority Interest",
                  ],
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

      if cf is not None and not cf.empty:
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
    except Exception:
      pass

    return {
        "コード": code.replace(".T", ""),
        "銘柄名": stock_name,
        "区分": tag,
        "総合シグナル": signal_label,
        "買い場スコア": buy_score,
        "売り場スコア": sell_score,
        "現在株価(円)": round(latest_close, 1),
        "配当利回り(%)": round(latest_yield, 2),
        "3年利回り位置(%)": round(yield_pos_3y, 1),
        "14日RSI": round(latest_rsi, 1),
        "200日線乖離(%)": round(dev_ma200, 1),
        "ROIC推移": roic_trend_str,
        "FCF(億円)": fcf_oku,
        "自己資本比率(%)": equity_ratio,
        "_dfs": (df_5y, df_3y, df_1y, df_6m, df_3m, df_1m),
    }
  except Exception as e:
    st.warning(f"{stock_name}({code}) スキップ: {e}")
    return None




# ====================================================
# 5. スマホアプリの画面UI（スマホ縦画面・完全最適化版）
# ====================================================
st.markdown(
    """
    <style>
    .app-title { font-size: 20px; font-weight: bold; margin-bottom: 2px; }
    .sec-title { font-size: 16px; font-weight: bold; margin-top: 10px; margin-bottom: 6px; border-left: 4px solid #ff4b4b; padding-left: 8px; }
    .stock-card {
        background-color: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 10px 12px;
        margin-bottom: 8px;
    }
    .card-header { font-size: 15px; font-weight: bold; color: #111; }
    .card-tag { font-size: 11px; color: #555; margin-bottom: 4px; }
    .card-sig { font-size: 13px; font-weight: bold; margin-bottom: 4px; }
    .card-metrics { font-size: 12px; color: #222; line-height: 1.5; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="app-title">📈 My株チェッカー Pro</div>', unsafe_allow_html=True
)
st.caption("5年〜1ヶ月の6期間統合 買い場＆売り場診断")

with st.sidebar:
  st.header("⚙️ ウォッチリスト管理")
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
  sort_mode = st.radio(
      "並べ替え基準",
      ["買い場スコア順（押し目探し）", "売り場スコア順（過熱・利確チェック）"],
  )
  st.divider()

  # --- ① 新規追加・区分変更（自動保存） ---
  st.subheader("➕ 銘柄を追加・区分変更（自動保存）")
  new_code = st.text_input("証券コード（4桁・例: 8056）", "")
  new_name = st.text_input("銘柄名（例: BIPROGY）", "")
  new_tag = st.selectbox(
      "区分・カテゴリ",
      [
          "【保有】主力株",
          "【依頼中】買付待機",
          "【レアアース】国策",
          "【防衛】国策",
          "【監視】押し目待ち",
      ],
  )
  if st.button("追加・更新して保存", use_container_width=True):
    if new_code.strip():
      c_fmt = (
          new_code.strip() + ".T"
          if not new_code.strip().endswith(".T")
          else new_code.strip()
      )
      n_fmt = new_name.strip() if new_name.strip() else c_fmt
      updated = [
          x
          for x in st.session_state.watchlist
          if next(iter(x)) != c_fmt
      ]
      updated.append([c_fmt, n_fmt, new_tag])
      if save_watchlist_to_github(updated):
        st.session_state.watchlist = updated
        st.success(f"{n_fmt} を保存しました！")
      else:
        st.error("保存に失敗しました（Secrets設定を確認してください）")

  st.divider()

  # --- ② 削除（自動保存） ---
  st.subheader("🗑️ 銘柄を削除（自動保存）")
  del_options = [
      f"{x[0]} : {x[1]}" for x in st.session_state.watchlist
  ]
  del_target = st.selectbox("削除する銘柄を選択", del_options)
  if st.button("削除して保存", use_container_width=True):
    if del_target:
      del_code = del_target.split(" : ").pop(0)
      updated = [
          x
          for x in st.session_state.watchlist
          if next(iter(x)) != del_code
      ]
      if save_watchlist_to_github(updated):
        st.session_state.watchlist = updated
        st.success(f"{del_target} を削除しました！")
        st.rerun()
      else:
        st.error("削除の保存に失敗しました")

if st.button(
    "🔍 6期間データ一括スキャン",
    type="primary",
    use_container_width=True,
):
  target_list = st.session_state.watchlist

  valid_results = []
  progress_bar = st.progress(0)
  status_text = st.empty()

  for idx, item in enumerate(target_list):
    status_text.text(f"取得中 ({idx + 1}/{len(target_list)}): {item}")
    res = analyze_mtf_stock(item)
    if res is not None:
      valid_results.append(res)
    progress_bar.progress((idx + 1) / len(target_list))

  status_text.empty()
  progress_bar.empty()

  if valid_results:
    st.session_state.df_res = pd.DataFrame(valid_results)
  else:
    st.error("データの取得に失敗しました。少し時間をおいて再度お試しください。")

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

  if sort_mode == "売り場スコア順（過熱・利確チェック）":
    df_show = df_show.sort_values(
        by="売り場スコア", ascending=False
    ).reset_index(drop=True)
  else:
    df_show = df_show.sort_values(
        by="買い場スコア", ascending=False
    ).reset_index(drop=True)

  st.markdown(
      f'<div class="sec-title">📊 売買タイミング一覧（{len(df_show)}銘柄）</div>',
      unsafe_allow_html=True,
  )

  # タブ切り替え：スマホ用カード表示（横スクロールなし） vs 詳細一覧表
  tab1, tab2 = st.tabs(["📱 スマホカード表示", "📋 詳細テーブル（全項目）"])

  with tab1:
    for idx, row in df_show.iterrows():
      st.markdown(
          f"""
          <div class="stock-card">
            <div class="card-header">{idx + 1}位. [{row['コード']}] {row['銘柄名']}</div>
            <div class="card-tag">{row['区分']}</div>
            <div class="card-sig">{row['総合シグナル']}</div>
            <div class="card-metrics">
              <b>買い場: {row['買い場スコア']}点</b> ／ 売り場: {row['売り場スコア']}点<br>
              株価: <b>{row['現在株価(円)']}円</b>（利回り: <b>{row['配当利回り(%)']}%</b>）<br>
              3年利回り位置: {row['3年利回り位置(%)']}% ／ RSI: {row['14日RSI']} ／ 200日線乖離: {row['200日線乖離(%)']}%
            </div>
          </div>
          """,
          unsafe_allow_html=True,
      )

  with tab2:
    display_cols = [
        "コード",
        "銘柄名",
        "買い場スコア",
        "売り場スコア",
        "現在株価(円)",
        "配当利回り(%)",
        "3年利回り位置(%)",
        "14日RSI",
        "200日線乖離(%)",
        "ROIC推移",
        "総合シグナル",
        "区分",
    ]
    st.dataframe(df_show[display_cols], use_container_width=True, hide_index=True)

  # 6期間マルチタイムフレーム・チャートパネル表示（スマホ画面横幅いっぱい・1列×6段）
  st.divider()
  st.markdown(
      '<div class="sec-title">📉 6期間チャート確認</div>',
      unsafe_allow_html=True,
  )
  selected_stock = st.selectbox(
      "銘柄を選択",
      df_show["コード"] + " : " + df_show["銘柄名"],
  )
  if selected_stock:
    code_str = selected_stock.split(" : ").pop(0)
    row_data = df_show[df_show["コード"] == code_str].head(1).squeeze()
    dfs = row_data["_dfs"]

    titles = [
        "5年",
        "3年",
        "1年",
        "6ヶ月",
        "3ヶ月",
        "1ヶ月",
    ]

    # スマホ画面の横幅いっぱいに1つずつ大きく表示（縦6段）
    for idx, (sub_df, t_str) in enumerate(zip(dfs, titles)):
      fig, ax = plt.subplots(figsize=(6, 2.8))
      ax.plot(
          sub_df.index,
          sub_df["Close"],
          color="#1f77b4",
          linewidth=1.8,
          label="Close",
      )
      if idx <= 2:
        ax.plot(
            sub_df.index,
            sub_df["MA200"],
            color="#d62728",
            linestyle="--",
            linewidth=1.2,
            label="MA200",
        )
        ax.plot(
            sub_df.index,
            sub_df["MA75"],
            color="#ff7f0e",
            linestyle=":",
            linewidth=1.2,
            label="MA75",
        )
      else:
        ax.plot(
            sub_df.index,
            sub_df["MA25"],
            color="#2ca02c",
            linestyle="--",
            linewidth=1.3,
            label="MA25",
        )

      ax.set_title(t_str, fontsize=12, fontweight="bold")
      ax.grid(True, linestyle=":", alpha=0.6)
      ax.legend(loc="upper left", fontsize=8)
      ax.xaxis.set_major_formatter(mdates.DateFormatter("%y/%m"))
      ax.tick_params(axis="x", rotation=20, labelsize=9)
      ax.tick_params(axis="y", labelsize=9)

      plt.tight_layout()
      st.pyplot(fig, use_container_width=True)
      plt.close(fig)

