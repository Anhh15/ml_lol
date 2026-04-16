import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# -- Đường dẫn ---------------------------------------------------------------
ROOT       = Path(__file__).resolve().parents[2]
RAW_DIR    = ROOT / 'data' / 'raw'
PROC_DIR   = ROOT / 'data' / 'processed'
PROC_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH        = RAW_DIR / 'games.csv'
CHAMP_INFO_PATH = RAW_DIR / 'champion_info_2.json'

TRAIN_OUT = PROC_DIR / 't1_train.csv'
TEST_OUT  = PROC_DIR / 't1_test.csv'

# -- Hằng số -----------------------------------------------------------------
RANDOM_STATE      = 42
TEST_SIZE         = 0.2
MIN_DURATION_SEC  = 300   # lọc trận remake / surrender cực sớm

ALL_ROLES = ['Fighter', 'Tank', 'Mage', 'Support', 'Assassin', 'Marksman']

META_COLS = ['gameId', 'creationTime', 'seasonId']

INGAME_COLS = [
    'gameDuration',
    'firstBlood', 'firstTower', 'firstInhibitor',
    'firstBaron', 'firstDragon', 'firstRiftHerald',
    't1_towerKills', 't1_inhibitorKills', 't1_baronKills',
    't1_dragonKills', 't1_riftHeraldKills',
    't2_towerKills', 't2_inhibitorKills', 't2_baronKills',
    't2_dragonKills', 't2_riftHeraldKills',
]


# -- Hàm chính ----------------------------------------------------------------

def load_champion_tags(path):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)['data']
    return {info['id']: info.get('tags', []) for info in data.values()}

def compute_role_counts(df: pd.DataFrame,
                        picks: list[str],
                        id_to_tags: dict[int, list[str]],
                        roles: list[str]):
    role_cols = {}
    for role in roles:
        col_name = role.lower() + '_count'
        role_cols[col_name] = sum(
            df[col].map(lambda cid, r=role: int(r in id_to_tags.get(cid, [])))
            for col in picks
        )
    return pd.DataFrame(role_cols, index=df.index)


def preprocess():
    # -- 1. Load -------------------------------------------------------------
    df = pd.read_csv(CSV_PATH)

    id_to_tags = load_champion_tags(CHAMP_INFO_PATH)

    # -- 2. Lọc trận ngắn  ----------------------------------------------------
    before = len(df)
    df = df[df['gameDuration'] >= MIN_DURATION_SEC].copy()
    removed = before - len(df)

    # -- 3. Drop cột không dùng -----------------------------------------------
    cols_to_drop = META_COLS + INGAME_COLS
    df.drop(columns=cols_to_drop, inplace=True)

    # -- 4. Convert target ----------------------------------------------------
    # winner: 1 → 0 (Team 1 thắng), 2 → 1 (Team 2 thắng)
    df['winner'] = df['winner'].map({1: 0, 2: 1})

    # -- 5. Tính role count (Binary Presence) ---------------------------------

    t1_picks = [f't1_champ{i}id' for i in range(1, 6)]
    t2_picks = [f't2_champ{i}id' for i in range(1, 6)]

    t1_roles = compute_role_counts(df, t1_picks, id_to_tags, ALL_ROLES)
    t2_roles = compute_role_counts(df, t2_picks, id_to_tags, ALL_ROLES)

    # Thêm prefix để phân biệt đội
    t1_roles.columns = ['t1_' + c for c in t1_roles.columns]
    t2_roles.columns = ['t2_' + c for c in t2_roles.columns]

    # Diff features
    diff_roles = pd.DataFrame(index=df.index)
    for role in ALL_ROLES:
        col = role.lower() + '_count'
        diff_roles[f'diff_{col}'] = t1_roles[f't1_{col}'] - t2_roles[f't2_{col}']

    df = pd.concat([df, t1_roles, t2_roles, diff_roles], axis=1)

    # -- 6. Train-test split --------------------------------------------------
    X = df.drop(columns=['winner'])
    y = df['winner']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    train_df = X_train.copy()
    train_df['winner'] = y_train

    test_df = X_test.copy()
    test_df['winner'] = y_test
    # -- 7. Lưu kết quả -------------------------------------------------------
    train_df.to_csv(TRAIN_OUT, index=False)
    test_df.to_csv(TEST_OUT,  index=False)

preprocess()
